
import torch
print("DEBUG: Torch imported")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoImageProcessor
from datasets import load_dataset
import os
import json
print("DEBUG: Standard imports done")

from modeling_vit import ViTForImageClassification
print("DEBUG: modeling_vit imported")
from run_experiments import prepare_data, compute_baseline_accuracy
print("DEBUG: run_experiments imported")
from experiments_v2 import (
    patch_shuffle_at_depth,
    evaluate_with_rope,
    compute_attention_distance
)
print("DEBUG: experiments_v2 imported")

# Configuration
MODEL_NAME = "facebook/deit-small-patch16-224"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "results_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_v2_experiments():
    print(f"Loading model: {MODEL_NAME}")
    model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)
    model.eval()
    
    dataloader = prepare_data()
    
    results = {}
    
    # Baseline
    baseline = compute_baseline_accuracy(model, dataloader, DEVICE)
    results['baseline'] = baseline
    print(f"Baseline: {baseline:.4f}")
    
    # 1. Patch Shuffle
    print("\n" + "="*60)
    print("1. Patch Shuffle at Depth K")
    print("="*60)
    depths = range(12)
    shuffle_res = patch_shuffle_at_depth(model, dataloader, depths, DEVICE)
    results['patch_shuffle'] = shuffle_res
    
    plt.figure(figsize=(10, 5))
    keys = list(shuffle_res.keys())
    vals = list(shuffle_res.values())
    plt.plot(keys, vals, 'o-', label='Shuffle Accuracy')
    plt.axhline(baseline, color='red', linestyle='--', label='Baseline')
    plt.xlabel('Shuffle Layer Depth')
    plt.ylabel('Accuracy')
    plt.title('Accuracy when Shuffling Patches After Layer K')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/patch_shuffle.png")
    plt.close()
    
    # 2. RoPE
    print("\n" + "="*60)
    print("2. RoPE vs Absolute Pos Emb")
    print("="*60)
    rope_acc = evaluate_with_rope(model, dataloader, DEVICE)
    results['rope_accuracy'] = rope_acc
    print(f"RoPE Accuracy (Zero Abs): {rope_acc:.4f}")
    
    # 3. Attention Distance
    print("\n" + "="*60)
    print("3. Attention Distance vs Depth")
    print("="*60)
    attn_dists = compute_attention_distance(model, dataloader, DEVICE)
    
    # Convert to list for JSON
    serializable_dists = {k: v.tolist() for k, v in attn_dists.items()}
    results['attention_distance'] = serializable_dists
    
    # Plot Mean distance per layer
    mean_dists = [np.mean(attn_dists[i]) for i in range(12)]
    plt.figure(figsize=(10, 5))
    plt.plot(range(12), mean_dists, 'o-', color='purple')
    plt.xlabel('Layer Index')
    plt.ylabel('Mean Attention Distance (pixels)')
    plt.title('Mean Attention Distance vs Depth')
    plt.savefig(f"{OUTPUT_DIR}/attention_distance.png")
    plt.close()
    
    # Save results
    with open(f"{OUTPUT_DIR}/results_v2.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_v2_experiments()
