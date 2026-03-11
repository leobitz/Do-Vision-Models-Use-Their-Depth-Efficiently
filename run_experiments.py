# run_experiments.py - Main script to run all ViT layer analysis experiments
"""
Run all experiments on DeiT-small model using ImageNet validation subset.
Results are saved to the artifacts directory.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoImageProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Import model and experiments
from modeling_vit import ViTForImageClassification
from experiments import (
    extract_cls_tokens,
    linear_probe_accuracy,
    compute_cka_matrix,
    evaluate_with_layer_skip,
    logit_lens_analysis,
    compute_residual_similarity,
    compute_effective_dimensionality,
    compute_noise_sensitivity,
    decode_positional_info,
    evaluate_with_position_ablation,
    compute_baseline_accuracy,
)

# Configuration
MODEL_NAME = "google/vit-large-patch16-224"
NUM_SAMPLES = 500  # Use a subset for speed
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "results_large"
SEED=42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def prepare_data():
    """Load a RANDOM, FIXED ImageNet-1k validation subset."""
    print("Loading ImageNet validation dataset...")

    # IMPORTANT: select split explicitly
    dataset = load_dataset("pouya-haghi/imagenet-1k", split="train")

    # Shuffle deterministically, then select subset
    dataset = dataset.shuffle(seed=SEED).select(range(NUM_SAMPLES))

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    batches = []
    for i in range(0, len(dataset), BATCH_SIZE):
        batch = dataset[i:i + BATCH_SIZE]

        images = [img.convert("RGB") for img in batch["image"]]
        inputs = processor(images, return_tensors="pt")
        inputs["labels"] = torch.tensor(batch["label"])

        batches.append(inputs)

    return batches


# def prepare_data():
#     """Load ImageNet validation subset."""
#     print("Loading ImageNet validation dataset...")
#     dataset = load_dataset("pouya-haghi/imagenet-1k")
#     dataset = dataset.take(NUM_SAMPLES)
    
#     processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    
#     def transform(examples):
#         images = [img.convert("RGB") for img in examples["image"]]
#         inputs = processor(images, return_tensors="pt")
#         inputs["label"] = torch.tensor(examples["label"])
#         return inputs
    
#     # Convert streaming dataset to list
#     samples = list(tqdm(dataset, total=NUM_SAMPLES, desc="Loading samples"))
    
#     # Create batches manually
#     processed = []
#     for i in range(0, len(samples), BATCH_SIZE):
#         batch = samples[i:i+BATCH_SIZE]
#         images = [s["image"].convert("RGB") for s in batch]
#         inputs = processor(images, return_tensors="pt")
#         inputs["label"] = torch.tensor([s["label"] for s in batch])
#         processed.append(inputs)
    
#     return processed


def run_all_experiments():
    """Run all experiments and save results."""
    
    # Load model
    print(f"Loading model: {MODEL_NAME}")
    model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)
    model.eval()
    
    # Load data
    dataloader = prepare_data()
    
    results = {}
    
    # =========================================================================
    # 1. Baseline Accuracy
    # =========================================================================
    print("\n" + "="*60)
    print("1. Computing Baseline Accuracy")
    print("="*60)
    baseline_acc = compute_baseline_accuracy(model, dataloader, DEVICE)
    results["baseline_accuracy"] = baseline_acc
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    
    # =========================================================================
    # 2. Linear Probing
    # =========================================================================
    print("\n" + "="*60)
    print("2. Linear Probing per Layer")
    print("="*60)
    cls_tokens, labels = extract_cls_tokens(model, dataloader, DEVICE)
    
    probe_acc = {}
    for layer_idx in range(len(cls_tokens)):
        acc = linear_probe_accuracy(cls_tokens[layer_idx], labels)
        probe_acc[layer_idx] = acc
        print(f"  Layer {layer_idx}: {acc:.4f}")
    results["linear_probe_accuracy"] = probe_acc
    
    # Plot
    plt.figure(figsize=(10, 5))
    layers = list(probe_acc.keys())
    accs = list(probe_acc.values())
    plt.plot(layers, accs, 'o-', linewidth=2, markersize=8)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Linear Probe Accuracy per Layer", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/linear_probe.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 3. CKA Similarity Matrix
    # =========================================================================
    print("\n" + "="*60)
    print("3. Computing CKA Similarity Matrix")
    print("="*60)
    cka_matrix = compute_cka_matrix(cls_tokens)
    results["cka_matrix"] = cka_matrix
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cka_matrix, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=range(len(cka_matrix)),
                yticklabels=range(len(cka_matrix)))
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Layer Index", fontsize=12)
    plt.title("CKA Similarity Between Layers", fontsize=14)
    plt.savefig(f"{OUTPUT_DIR}/cka_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  CKA matrix saved.")
    
    # =========================================================================
    # 4. Logit Lens (Early Exit)
    # =========================================================================
    print("\n" + "="*60)
    print("4. Logit Lens Analysis")
    print("="*60)
    logit_acc = logit_lens_analysis(model, dataloader, DEVICE)
    results["logit_lens_accuracy"] = logit_acc
    
    for layer_idx, acc in logit_acc.items():
        print(f"  Layer {layer_idx}: {acc:.4f}")
    
    plt.figure(figsize=(10, 5))
    layers = list(logit_acc.keys())
    accs = list(logit_acc.values())
    plt.plot(layers, accs, 's-', linewidth=2, markersize=8, color='orange')
    plt.axhline(y=baseline_acc, color='red', linestyle='--', label='Final Accuracy')
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Logit Lens: Early Exit Accuracy", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/logit_lens.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 5. Residual Similarity
    # =========================================================================
    print("\n" + "="*60)
    print("5. Residual Similarity Analysis")
    print("="*60)
    cos_sims, norm_ratios = compute_residual_similarity(model, dataloader, DEVICE)
    results["cosine_similarity"] = cos_sims
    results["norm_ratio"] = norm_ratios
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cosine similarity
    layers = list(cos_sims.keys())
    axes[0].plot(layers, [cos_sims[i] for i in layers], 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel("Layer Index", fontsize=12)
    axes[0].set_ylabel("Cosine Similarity", fontsize=12)
    axes[0].set_title("Input-Output Cosine Similarity", fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Norm ratio
    axes[1].plot(layers, [norm_ratios[i] for i in layers], 'o-', linewidth=2, markersize=8, color='green')
    axes[1].set_xlabel("Layer Index", fontsize=12)
    axes[1].set_ylabel("||f(x)|| / ||x||", fontsize=12)
    axes[1].set_title("Residual Norm Ratio", fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/residual_similarity.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 6. Effective Dimensionality
    # =========================================================================
    print("\n" + "="*60)
    print("6. Effective Dimensionality (PCA)")
    print("="*60)
    eff_dims = compute_effective_dimensionality(cls_tokens)
    results["effective_dimensionality"] = eff_dims
    
    for layer_idx, dim in eff_dims.items():
        print(f"  Layer {layer_idx}: {dim} components for 99% variance")
    
    plt.figure(figsize=(10, 5))
    layers = list(eff_dims.keys())
    dims = [eff_dims[i] for i in layers]
    plt.bar(layers, dims, color='purple', alpha=0.7)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Effective Dimensionality", fontsize=12)
    plt.title("Effective Dimensionality per Layer (99% variance)", fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(f"{OUTPUT_DIR}/effective_dim.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 7. Layer Skipping
    # =========================================================================
    print("\n" + "="*60)
    print("7. Layer Skipping Ablation")
    print("="*60)
    skip_acc = {}
    for layer_idx in range(len(model.vit.encoder.layer)):
        acc = evaluate_with_layer_skip(model, dataloader, layer_idx, DEVICE)
        skip_acc[layer_idx] = acc
        delta = acc - baseline_acc
        print(f"  Skip Layer {layer_idx}: {acc:.4f} (Δ = {delta:+.4f})")
    results["layer_skip_accuracy"] = skip_acc
    
    plt.figure(figsize=(10, 5))
    layers = list(skip_acc.keys())
    deltas = [skip_acc[i] - baseline_acc for i in layers]
    colors = ['red' if d < -0.01 else 'green' if d > 0.01 else 'gray' for d in deltas]
    plt.bar(layers, deltas, color=colors, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel("Skipped Layer Index", fontsize=12)
    plt.ylabel("Δ Accuracy", fontsize=12)
    plt.title("Accuracy Change When Skipping Each Layer", fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(f"{OUTPUT_DIR}/layer_skip.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 8. Positional Information Decoding
    # =========================================================================
    print("\n" + "="*60)
    print("8. Positional Information Decoding")
    print("="*60)
    r2_scores = decode_positional_info(model, dataloader, DEVICE)
    results["positional_r2"] = r2_scores
    
    for layer_idx, r2 in r2_scores.items():
        print(f"  Layer {layer_idx}: R² = {r2:.4f}")
    
    plt.figure(figsize=(10, 5))
    layers = list(r2_scores.keys())
    r2s = [r2_scores[i] for i in layers]
    plt.plot(layers, r2s, 'o-', linewidth=2, markersize=8, color='brown')
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("R² Score", fontsize=12)
    plt.title("Positional Information Decoding (Coordinate Regression)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/positional_decoding.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 9. Positional Ablation
    # =========================================================================
    print("\n" + "="*60)
    print("9. Positional Embedding Ablation")
    print("="*60)
    zero_acc = evaluate_with_position_ablation(model, dataloader, DEVICE, 'zero')
    shuffle_acc = evaluate_with_position_ablation(model, dataloader, DEVICE, 'shuffle')
    results["pos_zero_accuracy"] = zero_acc
    results["pos_shuffle_accuracy"] = shuffle_acc
    
    print(f"  Zero Position Embeddings: {zero_acc:.4f} (Δ = {zero_acc - baseline_acc:+.4f})")
    print(f"  Shuffle Position Embeddings: {shuffle_acc:.4f} (Δ = {shuffle_acc - baseline_acc:+.4f})")
    
    # =========================================================================
    # Save all results
    # =========================================================================
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
        return obj
    
    import json
    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    print(f"Results saved to {OUTPUT_DIR}/")
    print("\nExperiments complete!")
    
    return results


if __name__ == "__main__":
    results = run_all_experiments()
