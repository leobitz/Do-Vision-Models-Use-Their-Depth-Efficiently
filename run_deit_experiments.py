import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import traceback
from transformers import AutoImageProcessor, AutoConfig
from datasets import load_dataset
from modeling_vit import ViTForImageClassification
from experiments import extract_cls_tokens, linear_probe_accuracy, logit_lens_analysis

MODELS = {
    'DeiT-tiny': 'facebook/deit-tiny-patch16-224',
    'DeiT-small': 'facebook/deit-small-patch16-224',
    'DeiT-base': 'facebook/deit-base-patch16-224',
    'DeiT-tiny distilled': 'facebook/deit-tiny-distilled-patch16-224',
    'DeiT-small distilled': 'facebook/deit-small-distilled-patch16-224',
    'DeiT-base distilled': 'facebook/deit-base-distilled-patch16-224',
    'DeiT-base 384': 'facebook/deit-base-patch16-384',
    'DeiT-base distilled 384': 'facebook/deit-base-distilled-patch16-384'
}

NUM_SAMPLES = 500
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = 'results_deit'
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cache to store dataloaders by their resolution/processor to avoid reprocessing
_DATALOADER_CACHE = {}
_RAW_DATASET = None


def get_dataloader(model_name):
    global _RAW_DATASET

    processor = AutoImageProcessor.from_pretrained(model_name)

    # We use the processor's image size as the cache key 
    # (since the dataset is the same, only the crop size changes)
    size_key = str(processor.size)

    if size_key in _DATALOADER_CACHE:
        print(f"Using cached dataloader for resolution {size_key}")
        return _DATALOADER_CACHE[size_key]

    print(f"Processing data for resolution {size_key} (Model: {model_name})...")

    if _RAW_DATASET is None:
        print("Loading raw ImageNet dataset from disk...")
        dataset = load_dataset("pouya-haghi/imagenet-1k", split="train")
        _RAW_DATASET = dataset.shuffle(seed=SEED).select(range(NUM_SAMPLES))

    batches = []
    for i in range(0, len(_RAW_DATASET), BATCH_SIZE):
        batch = _RAW_DATASET[i:i + BATCH_SIZE]
        images = [img.convert("RGB") for img in batch["image"]]
        inputs = processor(images, return_tensors="pt")
        inputs["labels"] = torch.tensor(batch["label"])
        batches.append(inputs)

    _DATALOADER_CACHE[size_key] = batches
    return batches


def main():

    # Fix 3: Set random seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    results = {}

    for display_name, hf_id in MODELS.items():
        print(f"\n{'='*60}\nEvaluating: {display_name} ({hf_id})\n{'='*60}")
        try:
            model = ViTForImageClassification.from_pretrained(
                hf_id,
                ignore_mismatched_sizes=True
            )

            model.to(DEVICE)
            model.eval()

            dataloader = get_dataloader(hf_id)

            print("1. Linear Probing...")

            # Fix 2: Use no_grad
            with torch.no_grad():
                cls_tokens, labels = extract_cls_tokens(model, dataloader, DEVICE)

            probe_acc = {}
            for layer_idx in range(len(cls_tokens)):
                acc = linear_probe_accuracy(cls_tokens[layer_idx], labels)
                probe_acc[layer_idx] = acc

            print(f"Linear Probe Final Layer Acc: {probe_acc[len(cls_tokens)-1]:.4f}")

            print("2. Logit Lens Analysis...")

            # Fix 2: Use no_grad
            with torch.no_grad():
                logit_acc = logit_lens_analysis(model, dataloader, DEVICE)

            print(f"Logit Lens Final Layer Acc: {logit_acc[len(logit_acc)-1]:.4f}")

            results[display_name] = {
                'linear_probe': probe_acc,
                'logit_lens': logit_acc
            }

        except Exception as e:
            print(f"Failed to evaluate {display_name}: {e}")
            traceback.print_exc()

        # Fix 1: Only one cleanup block
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

    # Save results
    with open(f"{OUTPUT_DIR}/results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Plot Linear Probe
    plt.figure(figsize=(12, 8))
    for name, res in results.items():
        if 'linear_probe' in res:
            layers = list(res['linear_probe'].keys())
            accs = list(res['linear_probe'].values())

            sorted_idx = sorted([int(l) for l in layers])
            sorted_accs = [
                res['linear_probe'][str(i) if str(i) in layers else i]
                for i in sorted_idx
            ]

            plt.plot(sorted_idx, sorted_accs, 'o-', label=name)

    plt.xlabel('Layer Index')
    plt.ylabel('Accuracy')
    plt.title('Linear Probe Accuracy Across Layers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/linear_probe_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot Logit Lens
    plt.figure(figsize=(12, 8))
    for name, res in results.items():
        if 'logit_lens' in res:
            layers = list(res['logit_lens'].keys())
            accs = list(res['logit_lens'].values())

            sorted_idx = sorted([int(l) for l in layers])
            sorted_accs = [
                res['logit_lens'][str(i) if str(i) in layers else i]
                for i in sorted_idx
            ]

            plt.plot(sorted_idx, sorted_accs, 's-', label=name)

    plt.xlabel('Layer Index')
    plt.ylabel('Accuracy')
    plt.title('Logit Lens Accuracy Across Layers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/logit_lens_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("\nExperiments complete and plots saved.")


if __name__ == '__main__':
    main()