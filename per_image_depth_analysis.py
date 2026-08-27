"""Per-image ViT depth trajectories and layer-contribution analysis."""

from __future__ import annotations

import json
import math
import gc
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from PIL import ImageDraw, ImageFilter

from individual_image_analysis import (
    MODEL_SPECS,
    choose_device,
    component_logit_contribution,
    decompose_residual,
    get_final_layernorm,
    get_layer_stack,
    get_readout_heads,
    load_model,
    readout_logits,
)


sns.set_theme(style="whitegrid")

PROPERTY_COLUMNS = [
    "brightness",
    "contrast",
    "luminance_entropy",
    "edge_energy",
    "edge_density",
    "center_edge_fraction",
    "colorfulness",
    "aspect_ratio",
    "megapixels",
]


def chunks(values: Sequence[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def select_model(model_name: str):
    matches = [spec for spec in MODEL_SPECS if spec.name == model_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one known model named {model_name!r}.")
    spec = matches[0]
    if spec.is_distilled:
        raise ValueError("The per-image depth analysis requires a non-distilled model.")
    return spec


def load_balanced_validation(config: dict[str, Any]):
    """Reservoir-sample a class-balanced subset from validation-only parquet files."""
    dataset = load_dataset(
        "parquet",
        data_files={"validation": config["validation_data_files"]},
        split="validation",
        streaming=True,
    )
    samples_per_class = int(config["samples_per_class"])
    rng = np.random.default_rng(int(config["seed"]))
    reservoirs: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    seen_per_class: dict[int, int] = {}
    stream_limit = config.get("stream_limit")

    for dataset_index, example in enumerate(dataset):
        if stream_limit is not None and dataset_index >= int(stream_limit):
            break
        class_id = int(example[config["label_column"]])
        seen_count = seen_per_class.get(class_id, 0) + 1
        seen_per_class[class_id] = seen_count
        reservoir = reservoirs.setdefault(class_id, [])
        candidate = (dataset_index, example)
        if len(reservoir) < samples_per_class:
            reservoir.append(candidate)
        else:
            replacement = int(rng.integers(0, seen_count))
            if replacement < samples_per_class:
                reservoir[replacement] = candidate
    del dataset
    gc.collect()

    available_classes = sorted(reservoirs)
    max_classes = config.get("max_classes")
    if max_classes is not None:
        available_classes = available_classes[: int(max_classes)]

    selected: list[tuple[int, dict[str, Any]]] = []
    for class_id in available_classes:
        reservoir = reservoirs[class_id]
        if len(reservoir) < samples_per_class:
            raise ValueError(
                f"Class {class_id} has {seen_per_class[class_id]} streamed examples; "
                f"need {samples_per_class}."
            )
        selected.extend(reservoir)
    selected.sort(key=lambda item: item[0])

    selection_df = pd.DataFrame(
        {
            "dataset_index": [index for index, _example in selected],
            "label": [
                int(example[config["label_column"]]) for _index, example in selected
            ],
        }
    )
    selection_df["image_id"] = selection_df["dataset_index"].map(
        lambda index: f"imagenet_val_{index:05d}"
    )

    expected_classes = int(config.get("expected_num_classes", len(available_classes)))
    if max_classes is None and len(available_classes) != expected_classes:
        raise AssertionError(
            f"Expected {expected_classes} classes, found {len(available_classes)}."
        )
    return [example for _index, example in selected], selection_df


def apply_condition(image, condition: str):
    image = image.convert("RGB")
    if condition == "clean":
        return image
    if condition == "gaussian_blur_2":
        return image.filter(ImageFilter.GaussianBlur(radius=2.0))
    if condition == "center_occlusion_25":
        transformed = image.copy()
        width, height = transformed.size
        box_width = max(1, round(width * 0.5))
        box_height = max(1, round(height * 0.5))
        left = (width - box_width) // 2
        top = (height - box_height) // 2
        draw = ImageDraw.Draw(transformed)
        draw.rectangle(
            (left, top, left + box_width, top + box_height),
            fill=(128, 128, 128),
        )
        return transformed
    raise ValueError(f"Unknown image condition: {condition}")


def image_properties(image) -> dict[str, float]:
    rgb = np.asarray(image.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    gradient_x = np.diff(gray, axis=1, append=gray[:, -1:])
    gradient_y = np.diff(gray, axis=0, append=gray[-1:, :])
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)

    histogram, _ = np.histogram(gray, bins=64, range=(0.0, 1.0))
    probabilities = histogram / max(1, histogram.sum())
    probabilities = probabilities[probabilities > 0]
    luminance_entropy = float(
        -(probabilities * np.log(probabilities)).sum() / math.log(64)
    )

    center = gradient[32:96, 32:96]
    center_fraction = float(center.sum() / max(1e-12, gradient.sum()))
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colorfulness = float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2)
        + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )
    width, height = image.size
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "luminance_entropy": luminance_entropy,
        "edge_energy": float(gradient.mean()),
        "edge_density": float((gradient > 0.15).mean()),
        "center_edge_fraction": center_fraction,
        "colorfulness": colorfulness,
        "aspect_ratio": float(width / height),
        "megapixels": float(width * height / 1_000_000),
    }


def class_metrics(logits, labels, final_logits):
    batch_size, num_classes = logits.shape
    rows = torch.arange(batch_size, device=logits.device)
    true_logits = logits[rows, labels]
    masked = logits.clone()
    masked[rows, labels] = -torch.inf
    true_competitor_logits, true_competitors = masked.max(dim=-1)
    top_values, top_indices = logits.topk(5, dim=-1)
    log_probs = logits.log_softmax(dim=-1)
    probs = log_probs.exp()
    final_log_probs = final_logits.log_softmax(dim=-1)
    final_top5 = final_logits.topk(5, dim=-1).indices
    top5_overlap = (
        top_indices.unsqueeze(-1).eq(final_top5.unsqueeze(1)).any(dim=-1).sum(dim=-1)
        / 5.0
    )
    return {
        "prediction": top_indices[:, 0],
        "true_rank": logits.gt(true_logits.unsqueeze(-1)).sum(dim=-1) + 1,
        "true_margin": true_logits - true_competitor_logits,
        "top1_margin": top_values[:, 0] - top_values[:, 1],
        "top1_probability": probs[rows, top_indices[:, 0]],
        "true_probability": probs[rows, labels],
        "entropy": -(probs * log_probs).sum(dim=-1) / math.log(num_classes),
        "kl_to_final": (probs * (log_probs - final_log_probs)).sum(dim=-1),
        "top5_overlap": top5_overlap,
        "true_competitor": true_competitors,
    }


@torch.inference_mode()
def collect_trajectory_batch(
    model,
    processor,
    images,
    labels: Sequence[int],
    image_ids: Sequence[str],
    dataset_indices: Sequence[int],
    condition: str,
    device: torch.device,
):
    inputs = processor(images=images, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    label_tensor = torch.tensor(labels, device=device, dtype=torch.long)
    attention, ffn, residual_states = decompose_residual(model, inputs)
    final_layernorm = get_final_layernorm(model)
    heads = get_readout_heads(model)
    final_logits = readout_logits(residual_states[-1], final_layernorm, heads)
    final_prediction = final_logits.argmax(dim=-1)
    final_metrics = class_metrics(final_logits, label_tensor, final_logits)
    true_competitors = final_metrics["true_competitor"]
    rows = torch.arange(len(labels), device=device)

    trajectory_records = []
    for residual_state, hidden_state in enumerate(residual_states):
        logits = readout_logits(hidden_state, final_layernorm, heads)
        metrics = class_metrics(logits, label_tensor, final_logits)
        cpu_metrics = {
            key: value.detach().cpu().numpy()
            for key, value in metrics.items()
            if key != "true_competitor"
        }
        for local_index, image_id in enumerate(image_ids):
            trajectory_records.append(
                {
                    "image_id": image_id,
                    "dataset_index": int(dataset_indices[local_index]),
                    "condition": condition,
                    "label": int(labels[local_index]),
                    "residual_state": residual_state,
                    "prediction": int(cpu_metrics["prediction"][local_index]),
                    "true_rank": int(cpu_metrics["true_rank"][local_index]),
                    "true_margin": float(cpu_metrics["true_margin"][local_index]),
                    "top1_margin": float(cpu_metrics["top1_margin"][local_index]),
                    "top1_probability": float(
                        cpu_metrics["top1_probability"][local_index]
                    ),
                    "true_probability": float(
                        cpu_metrics["true_probability"][local_index]
                    ),
                    "entropy": float(cpu_metrics["entropy"][local_index]),
                    "kl_to_final": float(cpu_metrics["kl_to_final"][local_index]),
                    "top5_overlap": float(cpu_metrics["top5_overlap"][local_index]),
                    "is_correct": int(cpu_metrics["prediction"][local_index])
                    == int(labels[local_index]),
                    "agrees_with_final": int(cpu_metrics["prediction"][local_index])
                    == int(final_prediction[local_index].item()),
                }
            )

    component_records = []
    for layer, layer_components in enumerate(zip(attention, ffn)):
        for component_name, component in zip(("attention", "ffn"), layer_components):
            contribution = component_logit_contribution(
                component,
                residual_states[-1],
                final_layernorm,
                heads,
            )
            centered_norm = (contribution - contribution.mean(dim=-1, keepdim=True)).norm(
                dim=-1
            )
            true_values = contribution[rows, label_tensor]
            final_values = contribution[rows, final_prediction]
            competitor_values = contribution[rows, true_competitors]
            for local_index, image_id in enumerate(image_ids):
                component_records.append(
                    {
                        "image_id": image_id,
                        "dataset_index": int(dataset_indices[local_index]),
                        "condition": condition,
                        "label": int(labels[local_index]),
                        "layer": layer,
                        "component": component_name,
                        "true_class_contribution": float(true_values[local_index].item()),
                        "final_class_contribution": float(final_values[local_index].item()),
                        "true_margin_contribution": float(
                            (true_values[local_index] - competitor_values[local_index]).item()
                        ),
                        "centered_contribution_norm": float(
                            centered_norm[local_index].item()
                        ),
                    }
                )
    return trajectory_records, component_records, inputs, final_logits.detach()


@torch.inference_mode()
def collect_skip_batch(
    model,
    inputs: dict[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    labels: Sequence[int],
    image_ids: Sequence[str],
    dataset_indices: Sequence[int],
):
    label_tensor = torch.tensor(labels, device=baseline_logits.device, dtype=torch.long)
    baseline_metrics = class_metrics(baseline_logits, label_tensor, baseline_logits)
    baseline_probs = baseline_logits.softmax(dim=-1)
    baseline_prediction = baseline_logits.argmax(dim=-1)
    baseline_correct = baseline_prediction.eq(label_tensor)
    records = []

    for layer, block in enumerate(get_layer_stack(model)):
        def skip_hook(_module, module_inputs, module_output):
            skipped_state = module_inputs[0]
            if isinstance(module_output, tuple):
                return (skipped_state, *module_output[1:])
            return skipped_state

        handle = block.register_forward_hook(skip_hook)
        try:
            skipped_logits = model(**inputs).logits
        finally:
            handle.remove()

        skipped_metrics = class_metrics(skipped_logits, label_tensor, baseline_logits)
        skipped_prediction = skipped_logits.argmax(dim=-1)
        skipped_correct = skipped_prediction.eq(label_tensor)
        skipped_log_probs = skipped_logits.log_softmax(dim=-1)
        baseline_log_probs = baseline_logits.log_softmax(dim=-1)
        kl = (baseline_probs * (baseline_log_probs - skipped_log_probs)).sum(dim=-1)

        for local_index, image_id in enumerate(image_ids):
            records.append(
                {
                    "image_id": image_id,
                    "dataset_index": int(dataset_indices[local_index]),
                    "label": int(labels[local_index]),
                    "skipped_layer": layer,
                    "baseline_prediction": int(baseline_prediction[local_index].item()),
                    "skipped_prediction": int(skipped_prediction[local_index].item()),
                    "prediction_changed": bool(
                        baseline_prediction[local_index] != skipped_prediction[local_index]
                    ),
                    "baseline_correct": bool(baseline_correct[local_index].item()),
                    "skipped_correct": bool(skipped_correct[local_index].item()),
                    "correctness_changed": bool(
                        baseline_correct[local_index] != skipped_correct[local_index]
                    ),
                    "change_in_true_margin": float(
                        (
                            skipped_metrics["true_margin"][local_index]
                            - baseline_metrics["true_margin"][local_index]
                        ).item()
                    ),
                    "kl_baseline_to_skipped": float(kl[local_index].item()),
                }
            )
    return records


def earliest_suffix(values: np.ndarray, target: int | bool):
    for index in range(len(values)):
        if np.all(values[index:] == target):
            return float(index)
    return np.nan


def summarize_trajectories(
    trajectory_df: pd.DataFrame,
    property_df: pd.DataFrame,
    late_state: int,
):
    records = []
    for (image_id, condition), group in trajectory_df.groupby(
        ["image_id", "condition"], sort=False
    ):
        group = group.sort_values("residual_state")
        predictions = group["prediction"].to_numpy()
        correct = group["is_correct"].to_numpy(dtype=bool)
        final_prediction = int(predictions[-1])
        state_lookup = group.set_index("residual_state")
        late_row = state_lookup.loc[late_state]
        first_correct = (
            float(group.loc[group["is_correct"], "residual_state"].iloc[0])
            if group["is_correct"].any()
            else np.nan
        )
        records.append(
            {
                "image_id": image_id,
                "dataset_index": int(group["dataset_index"].iloc[0]),
                "condition": condition,
                "label": int(group["label"].iloc[0]),
                "final_prediction": final_prediction,
                "final_correct": bool(correct[-1]),
                "first_correct_depth": first_correct,
                "stable_correct_depth": earliest_suffix(correct, True)
                if correct[-1]
                else np.nan,
                "final_agreement_depth": earliest_suffix(predictions, final_prediction),
                "prediction_flips": int(np.sum(predictions[1:] != predictions[:-1])),
                "correct_to_wrong_flips": int(np.sum(correct[:-1] & ~correct[1:])),
                "wrong_to_correct_flips": int(np.sum(~correct[:-1] & correct[1:])),
                "late_true_margin_gain": float(
                    group["true_margin"].iloc[-1] - late_row["true_margin"]
                ),
                "late_confidence_gain": float(
                    group["top1_probability"].iloc[-1]
                    - late_row["top1_probability"]
                ),
                "final_true_margin": float(group["true_margin"].iloc[-1]),
                "final_entropy": float(group["entropy"].iloc[-1]),
            }
        )
    summary_df = pd.DataFrame(records)
    return summary_df.merge(property_df, on=["image_id", "condition"], how="left")


def add_skip_summaries(summary_df: pd.DataFrame, skip_df: pd.DataFrame, late_layer: int):
    if skip_df.empty:
        return summary_df
    records = []
    for image_id, group in skip_df.groupby("image_id", sort=False):
        late = group[group["skipped_layer"] >= late_layer]
        damage = -group["change_in_true_margin"]
        late_damage = -late["change_in_true_margin"]
        records.append(
            {
                "image_id": image_id,
                "skip_prediction_changes": int(group["prediction_changed"].sum()),
                "skip_correctness_changes": int(group["correctness_changed"].sum()),
                "mean_skip_margin_damage": float(damage.mean()),
                "max_skip_margin_damage": float(damage.max()),
                "late_skip_margin_damage": float(late_damage.mean()),
                "most_sensitive_layer": int(
                    group.loc[damage.idxmax(), "skipped_layer"]
                ),
            }
        )
    skip_summary = pd.DataFrame(records)
    return summary_df.merge(skip_summary, on="image_id", how="left")


def condition_shifts(summary_df: pd.DataFrame):
    clean = summary_df[summary_df["condition"] == "clean"].copy()
    metrics = [
        "final_agreement_depth",
        "prediction_flips",
        "late_true_margin_gain",
        "late_confidence_gain",
        "final_true_margin",
        "final_entropy",
    ]
    records = []
    for condition in sorted(set(summary_df["condition"]) - {"clean"}):
        changed = summary_df[summary_df["condition"] == condition]
        joined = changed.merge(clean, on="image_id", suffixes=("", "_clean"))
        for row in joined.itertuples(index=False):
            record = {
                "image_id": row.image_id,
                "dataset_index": row.dataset_index,
                "condition": condition,
                "final_correct": bool(row.final_correct),
                "final_correct_clean": bool(row.final_correct_clean),
            }
            for metric in metrics:
                record[f"change_in_{metric}"] = float(
                    getattr(row, metric) - getattr(row, f"{metric}_clean")
                )
            records.append(record)
    return pd.DataFrame(records)


def spearman_correlations(summary_df: pd.DataFrame):
    clean = summary_df[summary_df["condition"] == "clean"]
    outcomes = [
        "final_agreement_depth",
        "prediction_flips",
        "late_true_margin_gain",
        "late_skip_margin_damage",
    ]
    records = []
    for property_name in PROPERTY_COLUMNS:
        for outcome in outcomes:
            if outcome not in clean:
                continue
            pair = clean[[property_name, outcome]].dropna()
            correlation = pair[property_name].rank().corr(pair[outcome].rank())
            records.append(
                {
                    "property": property_name,
                    "outcome": outcome,
                    "spearman_correlation": float(correlation),
                    "count": len(pair),
                }
            )
    return pd.DataFrame(records)


def choose_representatives(summary_df: pd.DataFrame):
    clean = summary_df[summary_df["condition"] == "clean"].copy()
    choices = []

    def take(category, frame, column, ascending):
        used = {choice["image_id"] for choice in choices}
        frame = frame[~frame["image_id"].isin(used)].dropna(subset=[column])
        if frame.empty:
            return
        row = frame.sort_values(column, ascending=ascending).iloc[0].to_dict()
        row["category"] = category
        choices.append(row)

    take("early_stable", clean[clean["final_correct"]], "final_agreement_depth", True)
    take("late_corrected", clean[clean["final_correct"]], "stable_correct_depth", False)
    take("most_unstable", clean, "prediction_flips", False)
    take("final_error", clean[~clean["final_correct"]], "prediction_flips", False)
    return pd.DataFrame(choices)


def plot_summary(summary_df, skip_df, shift_df, correlation_df, output_path):
    clean = summary_df[summary_df["condition"] == "clean"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.histplot(
        clean,
        x="final_agreement_depth",
        discrete=True,
        ax=axes[0, 0],
        color="#1f77b4",
    )
    axes[0, 0].set_title("Final-agreement depth on clean images")

    skip_plot = skip_df.assign(margin_damage=-skip_df["change_in_true_margin"])
    sns.lineplot(
        skip_plot,
        x="skipped_layer",
        y="margin_damage",
        errorbar=("ci", 95),
        marker="o",
        ax=axes[0, 1],
    )
    axes[0, 1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[0, 1].set_title("True-margin damage from skipping each layer")

    if not shift_df.empty:
        sns.boxplot(
            shift_df,
            x="condition",
            y="change_in_final_agreement_depth",
            ax=axes[1, 0],
        )
        axes[1, 0].axhline(0, color="gray", linestyle="--", linewidth=1)
        axes[1, 0].set_title("Change in depth demand under controlled difficulty")

    pivot = correlation_df.pivot(
        index="property", columns="outcome", values="spearman_correlation"
    )
    sns.heatmap(pivot, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, ax=axes[1, 1])
    axes[1, 1].set_title("Image-property Spearman correlations")
    fig.suptitle("Per-image depth and layer-contribution analysis", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_skip_heatmap(summary_df, skip_df, output_path):
    clean = summary_df[summary_df["condition"] == "clean"].sort_values(
        ["final_agreement_depth", "prediction_flips"]
    )
    pivot = skip_df.assign(margin_damage=-skip_df["change_in_true_margin"]).pivot(
        index="image_id", columns="skipped_layer", values="margin_damage"
    )
    pivot = pivot.reindex(clean["image_id"])
    limit = float(np.nanpercentile(np.abs(pivot.to_numpy()), 99))
    fig, ax = plt.subplots(figsize=(12, 9))
    image = ax.imshow(
        pivot.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="vlag",
        vmin=-limit,
        vmax=limit,
    )
    ax.set_title("Image-by-layer skip sensitivity, sorted by final-agreement depth")
    ax.set_xlabel("Skipped layer")
    ax.set_ylabel("Clean images")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    fig.colorbar(image, ax=ax, label="True-margin damage")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_representatives(
    selected_images,
    representative_df,
    trajectory_df,
    label_names,
    output_path,
):
    if representative_df.empty:
        return
    fig, axes = plt.subplots(len(representative_df), 2, figsize=(13, 4 * len(representative_df)))
    axes = np.atleast_2d(axes)
    for row_index, row in enumerate(representative_df.itertuples(index=False)):
        image = selected_images[int(row.dataset_index)].convert("RGB")
        axes[row_index, 0].imshow(image)
        axes[row_index, 0].axis("off")
        label_name = label_names[int(row.label)] if label_names else str(row.label)
        prediction_name = (
            label_names[int(row.final_prediction)]
            if label_names
            else str(row.final_prediction)
        )
        axes[row_index, 0].set_title(
            f"{row.category}: label={label_name}\nfinal={prediction_name}"
        )

        path = trajectory_df[
            (trajectory_df["image_id"] == row.image_id)
            & (trajectory_df["condition"] == "clean")
        ].sort_values("residual_state")
        axes[row_index, 1].plot(
            path["residual_state"], path["true_margin"], marker="o", linewidth=2
        )
        axes[row_index, 1].axhline(0, color="gray", linestyle="--", linewidth=1)
        axes[row_index, 1].set_title(
            f"agreement={row.final_agreement_depth:g}, flips={row.prediction_flips:g}"
        )
        axes[row_index, 1].set_xlabel("Residual state")
        axes[row_index, 1].set_ylabel("True-class margin")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_frames(frames: dict[str, pd.DataFrame], results_dir: Path):
    paths = {}
    for name, frame in frames.items():
        path = results_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths


def run_per_image_depth_analysis(config: dict[str, Any]):
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    selected_examples, selection_df = load_balanced_validation(config)

    spec = select_model(config["model_name"])
    device = choose_device()
    print(
        f"Running {spec.name} on {len(selection_df):,} balanced validation images "
        f"with device={device}."
    )
    model, processor = load_model(spec, device)
    label_names = [
        model.config.id2label.get(index, f"class_{index}")
        for index in range(int(model.config.num_labels))
    ]
    selection_df["label_name"] = selection_df["label"].map(
        lambda class_id: label_names[class_id]
    )
    selection_df.to_csv(results_dir / "selection.csv", index=False)
    trajectory_records = []
    component_records = []
    property_records = []
    skip_records = []

    try:
        selected_positions = list(range(len(selected_examples)))
        for batch_number, batch_positions in enumerate(
            chunks(selected_positions, int(config["batch_size"])), start=1
        ):
            batch = [selected_examples[position] for position in batch_positions]
            batch_indices = selection_df.iloc[batch_positions]["dataset_index"].tolist()
            clean_images = [
                example[config["image_column"]].convert("RGB") for example in batch
            ]
            labels = [int(example[config["label_column"]]) for example in batch]
            image_ids = [f"imagenet_val_{index:05d}" for index in batch_indices]
            clean_inputs = None
            clean_logits = None
            for condition in config["conditions"]:
                images = [apply_condition(image, condition) for image in clean_images]
                for image_id, image in zip(image_ids, images):
                    property_records.append(
                        {"image_id": image_id, "condition": condition, **image_properties(image)}
                    )
                trajectory, components, inputs, final_logits = collect_trajectory_batch(
                    model,
                    processor,
                    images,
                    labels,
                    image_ids,
                    batch_indices,
                    condition,
                    device,
                )
                trajectory_records.extend(trajectory)
                component_records.extend(components)
                if condition == "clean":
                    clean_inputs = inputs
                    clean_logits = final_logits

            if config["run_layer_skip"]:
                if clean_inputs is None or clean_logits is None:
                    raise ValueError("The clean condition is required for layer skipping.")
                skip_records.extend(
                    collect_skip_batch(
                        model,
                        clean_inputs,
                        clean_logits,
                        labels,
                        image_ids,
                        batch_indices,
                    )
                )
            if batch_number % int(config["log_every_batches"]) == 0:
                completed = min(
                    batch_number * int(config["batch_size"]), len(selected_examples)
                )
                print(
                    f"Processed {completed:,}/{len(selected_examples):,} images",
                    flush=True,
                )
    finally:
        del model, processor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    trajectory_df = pd.DataFrame(trajectory_records)
    component_df = pd.DataFrame(component_records)
    property_df = pd.DataFrame(property_records)
    skip_df = pd.DataFrame(skip_records)
    summary_df = summarize_trajectories(
        trajectory_df,
        property_df,
        late_state=int(config["late_residual_state"]),
    )
    summary_df = add_skip_summaries(
        summary_df,
        skip_df,
        late_layer=int(config["late_skip_layer"]),
    )
    shift_df = condition_shifts(summary_df)
    correlation_df = spearman_correlations(summary_df)
    representative_df = choose_representatives(summary_df)

    frames = {
        "trajectory": trajectory_df,
        "component_attribution": component_df,
        "image_properties": property_df,
        "layer_skip": skip_df,
        "image_summary": summary_df,
        "condition_shifts": shift_df,
        "correlations": correlation_df,
        "representative_examples": representative_df,
    }
    paths = save_frames(frames, results_dir)
    figure_paths = {
        "summary_report": results_dir / "summary_report.png",
        "skip_heatmap": results_dir / "skip_heatmap.png",
        "representatives": results_dir / "representatives.png",
    }
    plot_summary(summary_df, skip_df, shift_df, correlation_df, figure_paths["summary_report"])
    plot_skip_heatmap(summary_df, skip_df, figure_paths["skip_heatmap"])
    plot_representatives(
        {
            int(row.dataset_index): selected_examples[position][config["image_column"]]
            for position, row in enumerate(selection_df.itertuples(index=False))
        },
        representative_df,
        trajectory_df,
        label_names,
        figure_paths["representatives"],
    )

    clean_summary = summary_df[summary_df["condition"] == "clean"]
    metadata = {
        "model_name": spec.name,
        "repo_id": spec.repo_id,
        "dataset_name": config["dataset_name"],
        "validation_data_files": config["validation_data_files"],
        "split": config["split"],
        "seed": int(config["seed"]),
        "samples_per_class": int(config["samples_per_class"]),
        "num_images": int(len(selection_df)),
        "num_classes": int(selection_df["label"].nunique()),
        "conditions": list(config["conditions"]),
        "clean_final_accuracy": float(clean_summary["final_correct"].mean()),
        "clean_median_final_agreement_depth": float(
            clean_summary["final_agreement_depth"].median()
        ),
        "clean_mean_prediction_flips": float(clean_summary["prediction_flips"].mean()),
        "trajectory_rows": int(len(trajectory_df)),
        "component_rows": int(len(component_df)),
        "skip_rows": int(len(skip_df)),
    }
    metadata_path = results_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    paths["metadata"] = metadata_path
    paths.update(figure_paths)
    return {"metadata": metadata, "frames": frames, "paths": paths}
