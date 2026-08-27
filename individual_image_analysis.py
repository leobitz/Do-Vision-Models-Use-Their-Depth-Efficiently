"""Trace one image through the residual stream of a DeiT classifier."""

from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    DeiTForImageClassificationWithTeacher,
    ViTForImageClassification,
)


sns.set_theme(style="whitegrid")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    imagenet_top1: float
    imagenet_top5: float
    params: str
    is_distilled: bool = False


MODEL_SPECS = [
    ModelSpec("DeiT-tiny", "facebook/deit-tiny-patch16-224", 72.2, 91.1, "5M"),
    ModelSpec("DeiT-small", "facebook/deit-small-patch16-224", 79.9, 95.0, "22M"),
    ModelSpec("DeiT-base", "facebook/deit-base-patch16-224", 81.8, 95.6, "86M"),
    ModelSpec("DeiT-tiny distilled", "facebook/deit-tiny-distilled-patch16-224", 74.5, 91.9, "6M", True),
    ModelSpec("DeiT-small distilled", "facebook/deit-small-distilled-patch16-224", 81.2, 95.4, "22M", True),
    ModelSpec("DeiT-base distilled", "facebook/deit-base-distilled-patch16-224", 83.4, 96.5, "87M", True),
    ModelSpec("DeiT-base 384", "facebook/deit-base-patch16-384", 82.9, 96.2, "87M"),
    ModelSpec("DeiT-base distilled 384", "facebook/deit-base-distilled-patch16-384", 85.2, 97.2, "88M", True),
]


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def select_single_model(selected_names: Sequence[str]) -> ModelSpec:
    selected = [spec for spec in MODEL_SPECS if spec.name in set(selected_names)]
    missing = sorted(set(selected_names) - {spec.name for spec in selected})
    if missing:
        raise ValueError(f"Unknown model names: {missing}")
    if len(selected) != 1:
        raise ValueError("Individual-image analysis requires exactly one selected model.")
    return selected[0]


def get_model_class(spec: ModelSpec):
    if spec.is_distilled:
        return DeiTForImageClassificationWithTeacher
    return ViTForImageClassification


def get_base_model(model):
    prefix = getattr(model, "base_model_prefix", None)
    if prefix and hasattr(model, prefix):
        return getattr(model, prefix)
    for attr in ("deit", "vit", "model"):
        if hasattr(model, attr):
            return getattr(model, attr)
    return model


def get_layer_stack(model):
    base = get_base_model(model)
    encoder = getattr(base, "encoder", None)
    if encoder is not None:
        if hasattr(encoder, "layer"):
            return encoder.layer
        if hasattr(encoder, "layers"):
            return encoder.layers

    candidates = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.ModuleList) and len(module) >= 2
    ]
    if not candidates:
        raise AttributeError("Could not locate the transformer block list.")
    return max(candidates, key=len)


def get_attention_module(block):
    for name, child in block.named_children():
        if "attention" in name.lower() or "attn" in name.lower():
            return child
    raise AttributeError("Could not locate the attention submodule.")


def get_final_layernorm(model):
    base = get_base_model(model)
    for attr in ("layernorm", "layer_norm", "norm", "ln_post"):
        if hasattr(base, attr):
            return getattr(base, attr)
    raise AttributeError("Could not locate the final layer norm.")


def get_readout_heads(model):
    if hasattr(model, "cls_classifier") and hasattr(model, "distillation_classifier"):
        return [(0, model.cls_classifier, 0.5), (1, model.distillation_classifier, 0.5)]
    if hasattr(model, "classifier"):
        return [(0, model.classifier, 1.0)]
    raise AttributeError("Could not locate a classifier head.")


def load_model(spec: ModelSpec, device: torch.device):
    processor = AutoImageProcessor.from_pretrained(spec.repo_id, use_fast=True)
    model = get_model_class(spec).from_pretrained(
        spec.repo_id,
        attn_implementation="eager",
    )
    return model.to(device).eval(), processor


@torch.inference_mode()
def decompose_residual(model, inputs: dict[str, torch.Tensor]):
    layers = get_layer_stack(model)
    attention_outputs: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_index: int):
        def hook(_module, _module_inputs, module_output):
            if isinstance(module_output, (tuple, list)):
                module_output = module_output[0]
            attention_outputs[layer_index] = module_output

        return hook

    for layer_index, block in enumerate(layers):
        handles.append(
            get_attention_module(block).register_forward_hook(make_hook(layer_index))
        )

    try:
        outputs = model(**inputs, output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()

    if outputs.hidden_states is None:
        raise ValueError("Model did not return hidden states.")

    residual_states = list(outputs.hidden_states)
    attention = [attention_outputs[index] for index in range(len(layers))]
    ffn = [
        residual_states[index + 1] - residual_states[index] - attention[index]
        for index in range(len(layers))
    ]
    return attention, ffn, residual_states


def frozen_layernorm_vector(component, final_state, final_layernorm):
    std = torch.sqrt(
        final_state.var(dim=-1, unbiased=False, keepdim=True) + final_layernorm.eps
    )
    centered = component - component.mean(dim=-1, keepdim=True)
    return final_layernorm.weight * centered / std


def component_logit_contribution(
    component,
    final_state,
    final_layernorm,
    heads,
):
    total = None
    for token_index, classifier, head_weight in heads:
        normalized = frozen_layernorm_vector(
            component[:, token_index, :],
            final_state[:, token_index, :],
            final_layernorm,
        )
        contribution = normalized @ classifier.weight.t()
        weighted = head_weight * contribution
        total = weighted if total is None else total + weighted
    return total


def readout_logits(residual_state, final_layernorm, heads):
    total = None
    for token_index, classifier, head_weight in heads:
        logits = classifier(final_layernorm(residual_state[:, token_index, :]))
        weighted = head_weight * logits
        total = weighted if total is None else total + weighted
    return total


def readout_constant(final_layernorm, heads):
    total = None
    for _token_index, classifier, head_weight in heads:
        constant = head_weight * classifier(final_layernorm.bias)
        total = constant if total is None else total + constant
    return total


def readout_token_norm(component, heads) -> float:
    return float(
        sum(
            head_weight * component[0, token_index, :].norm().item()
            for token_index, _classifier, head_weight in heads
        )
    )


def class_name(class_id: int, label_names: Sequence[str]) -> str:
    if 0 <= class_id < len(label_names):
        return label_names[class_id]
    return f"class_{class_id}"


def load_sample(config: dict[str, Any]):
    dataset = load_dataset(
        config["dataset_name"],
        split=config["split"],
        streaming=config["streaming"],
    )
    sample_index = int(config["sample_index"])
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative.")

    if config["streaming"]:
        sample = next(islice(dataset, sample_index, sample_index + 1), None)
    else:
        sample = dataset[sample_index] if sample_index < len(dataset) else None
    if sample is None:
        raise IndexError(f"No sample at index {sample_index} in {config['split']!r}.")

    features = getattr(dataset, "features", None)
    label_feature = features.get(config["label_column"]) if features else None
    label_names = list(getattr(label_feature, "names", None) or [])
    image = sample[config["image_column"]].convert("RGB")
    label = int(sample[config["label_column"]])
    return image, label, label_names


def model_label_names(model, dataset_label_names: Sequence[str]) -> list[str]:
    num_labels = int(model.config.num_labels)
    if len(dataset_label_names) == num_labels:
        return list(dataset_label_names)
    return [
        str(model.config.id2label.get(index, f"class_{index}"))
        for index in range(num_labels)
    ]


@torch.inference_mode()
def analyse_loaded_model(
    model,
    processor,
    image,
    label: int,
    label_names: Sequence[str],
    top_k: int,
    device: torch.device,
):
    inputs = processor(images=[image], return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    attention, ffn, residual_states = decompose_residual(model, inputs)
    final_layernorm = get_final_layernorm(model)
    heads = get_readout_heads(model)

    real_logits = model(**inputs).logits
    lens_logits = readout_logits(residual_states[-1], final_layernorm, heads)
    probabilities = real_logits.softmax(dim=-1)[0]
    requested_top_k = min(max(2, int(top_k)), probabilities.numel())
    top = probabilities.topk(requested_top_k)
    prediction = int(top.indices[0].item())
    alternative = int(top.indices[1].item())

    embedding = component_logit_contribution(
        residual_states[0],
        residual_states[-1],
        final_layernorm,
        heads,
    )
    constant = readout_constant(final_layernorm, heads).unsqueeze(0)
    running_logits = constant + embedding
    component_rows = []

    for layer_index, layer_components in enumerate(zip(attention, ffn)):
        for component_name, component in zip(("attention", "ffn"), layer_components):
            contribution = component_logit_contribution(
                component,
                residual_states[-1],
                final_layernorm,
                heads,
            )
            running_logits = running_logits + contribution
            vector = contribution[0]
            component_rows.append(
                {
                    "layer": layer_index,
                    "component": component_name,
                    "label_id": label,
                    "prediction_id": prediction,
                    "alternative_id": alternative,
                    "label_logit_contribution": float(vector[label].item()),
                    "prediction_logit_contribution": float(vector[prediction].item()),
                    "alternative_logit_contribution": float(vector[alternative].item()),
                    "prediction_margin_contribution": float(
                        (vector[prediction] - vector[alternative]).item()
                    ),
                    "cumulative_prediction_margin": float(
                        (running_logits[0, prediction] - running_logits[0, alternative]).item()
                    ),
                    "centered_contribution_norm": float(
                        (vector - vector.mean()).norm().item()
                    ),
                    "readout_token_update_norm": readout_token_norm(component, heads),
                }
            )

    logit_lens_rows = []
    for residual_index, residual_state in enumerate(residual_states):
        logits = readout_logits(residual_state, final_layernorm, heads)[0]
        probs = logits.softmax(dim=-1)
        lens_prediction = int(logits.argmax().item())
        logit_lens_rows.append(
            {
                "residual_state": residual_index,
                "lens_prediction_id": lens_prediction,
                "lens_prediction_name": class_name(lens_prediction, label_names),
                "lens_prediction_probability": float(probs[lens_prediction].item()),
                "label_probability": float(probs[label].item()),
                "final_prediction_probability": float(probs[prediction].item()),
                "prediction_vs_alternative_margin": float(
                    (logits[prediction] - logits[alternative]).item()
                ),
                "lens_prediction_is_label": lens_prediction == label,
            }
        )

    top_predictions = pd.DataFrame(
        [
            {
                "rank": rank,
                "class_id": int(class_id.item()),
                "class_name": class_name(int(class_id.item()), label_names),
                "probability": float(probability.item()),
            }
            for rank, (probability, class_id) in enumerate(
                zip(top.values, top.indices),
                start=1,
            )
        ]
    )

    lens_gap = float((lens_logits - real_logits).abs().max().item())
    reconstruction_gap = float((running_logits - real_logits).abs().max().item())
    tolerance = 1e-4
    if lens_gap > tolerance or reconstruction_gap > tolerance:
        raise AssertionError(
            "DLA reconstruction failed: "
            f"lens gap={lens_gap:.3e}, reconstruction gap={reconstruction_gap:.3e}"
        )

    metadata = {
        "label_id": label,
        "label_name": class_name(label, label_names),
        "prediction_id": prediction,
        "prediction_name": class_name(prediction, label_names),
        "prediction_probability": float(probabilities[prediction].item()),
        "prediction_is_label": prediction == label,
        "alternative_id": alternative,
        "alternative_name": class_name(alternative, label_names),
        "prediction_vs_alternative_margin": float(
            (real_logits[0, prediction] - real_logits[0, alternative]).item()
        ),
        "baseline_prediction_margin": float(
            ((constant + embedding)[0, prediction] - (constant + embedding)[0, alternative]).item()
        ),
        "lens_max_absolute_error": lens_gap,
        "dla_reconstruction_max_absolute_error": reconstruction_gap,
    }
    return {
        "image": image,
        "metadata": metadata,
        "components": pd.DataFrame(component_rows),
        "logit_lens": pd.DataFrame(logit_lens_rows),
        "top_predictions": top_predictions,
    }


def save_analysis(result: dict[str, Any], spec: ModelSpec, config: dict[str, Any]) -> dict[str, Path]:
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{slugify(spec.name)}_sample_{int(config['sample_index'])}"
    paths = {
        "components": results_dir / f"{prefix}_individual_components.csv",
        "logit_lens": results_dir / f"{prefix}_individual_logit_lens.csv",
        "top_predictions": results_dir / f"{prefix}_individual_top_predictions.csv",
        "metadata": results_dir / f"{prefix}_individual_metadata.json",
        "image": results_dir / f"{prefix}_input.png",
        "report": results_dir / f"{prefix}_report.png",
    }
    result["components"].to_csv(paths["components"], index=False)
    result["logit_lens"].to_csv(paths["logit_lens"], index=False)
    result["top_predictions"].to_csv(paths["top_predictions"], index=False)
    result["image"].save(paths["image"])

    metadata = {
        "model_name": spec.name,
        "repo_id": spec.repo_id,
        "dataset_name": config["dataset_name"],
        "split": config["split"],
        "sample_index": int(config["sample_index"]),
        **result["metadata"],
        "top_predictions": result["top_predictions"].to_dict(orient="records"),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
    result["metadata"] = metadata
    result["paths"] = paths
    return paths


def plot_individual_report(result: dict[str, Any], report_path: Path | None = None):
    metadata = result["metadata"]
    components = result["components"]
    lens = result["logit_lens"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].imshow(result["image"])
    axes[0, 0].axis("off")
    correctness = "correct" if metadata["prediction_is_label"] else "incorrect"
    axes[0, 0].set_title(
        f"Label: {metadata['label_name']}\n"
        f"Prediction: {metadata['prediction_name']} "
        f"({metadata['prediction_probability']:.1%}, {correctness})"
    )

    for component_name, color in (("attention", "#d62728"), ("ffn", "#2ca02c")):
        subset = components[components["component"] == component_name]
        axes[0, 1].plot(
            subset["layer"],
            subset["prediction_margin_contribution"],
            marker="o",
            linewidth=2,
            color=color,
            label=component_name,
        )
    axes[0, 1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Contribution to final prediction margin")
    axes[0, 1].set_xlabel("Layer")
    axes[0, 1].set_ylabel("Predicted logit minus alternative logit")
    axes[0, 1].legend()

    axes[1, 0].plot(
        lens["residual_state"],
        lens["final_prediction_probability"],
        marker="o",
        linewidth=2,
        label="final prediction",
    )
    axes[1, 0].plot(
        lens["residual_state"],
        lens["label_probability"],
        marker="o",
        linewidth=2,
        label="ground-truth label",
    )
    axes[1, 0].set_title("Logit-lens probability by residual state")
    axes[1, 0].set_xlabel("Residual state, 0 is embeddings")
    axes[1, 0].set_ylabel("Probability")
    axes[1, 0].legend()

    order = np.arange(len(components))
    colors = np.where(components["component"].eq("attention"), "#d62728", "#2ca02c")
    axes[1, 1].plot(
        order,
        components["cumulative_prediction_margin"],
        color="#1f77b4",
        linewidth=2,
    )
    axes[1, 1].scatter(
        order,
        components["cumulative_prediction_margin"],
        c=colors,
        s=32,
    )
    axes[1, 1].axhline(0, color="gray", linestyle="--", linewidth=1)
    tick_labels = [
        f"{row.layer}{'A' if row.component == 'attention' else 'F'}"
        for row in components.itertuples()
    ]
    axes[1, 1].set_xticks(order)
    axes[1, 1].set_xticklabels(tick_labels, rotation=60, ha="right")
    axes[1, 1].set_title("Cumulative DLA prediction margin")
    axes[1, 1].set_xlabel("Layer component, A is attention and F is FFN")
    axes[1, 1].set_ylabel("Predicted logit minus alternative logit")

    fig.suptitle(
        f"{metadata['model_name']} on sample {metadata['sample_index']}",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if report_path is not None:
        fig.savefig(report_path, dpi=160, bbox_inches="tight")
    return fig


def run_individual_image_analysis(spec: ModelSpec, config: dict[str, Any]):
    device = choose_device()
    print(f"Using device: {device}")
    image, label, dataset_label_names = load_sample(config)
    model, processor = load_model(spec, device)
    try:
        label_names = model_label_names(model, dataset_label_names)
        result = analyse_loaded_model(
            model,
            processor,
            image,
            label,
            label_names,
            config["top_k"],
            device,
        )
        save_analysis(result, spec, config)
        plot_individual_report(result, result["paths"]["report"])
        return result
    finally:
        del model, processor
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
