import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, DeiTForImageClassificationWithTeacher
from datasets import load_dataset
from modeling_vit import ViTForImageClassification

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configurations
models_to_test = [
    {"name": "DeiT-tiny", "repo": "facebook/deit-tiny-patch16-224", "top1": 72.2, "top5": 91.1, "params": "5M"},
    {"name": "DeiT-small", "repo": "facebook/deit-small-patch16-224", "top1": 79.9, "top5": 95.0, "params": "22M"},
    {"name": "DeiT-base", "repo": "facebook/deit-base-patch16-224", "top1": 81.8, "top5": 95.6, "params": "86M"},
    {"name": "DeiT-tiny distilled", "repo": "facebook/deit-tiny-distilled-patch16-224", "top1": 74.5, "top5": 91.9, "params": "6M"},
    
    {"name": "DeiT-small distilled", "repo": "facebook/deit-small-distilled-patch16-224", "top1": 81.2, "top5": 95.4, "params": "22M"},
    {"name": "DeiT-base distilled", "repo": "facebook/deit-base-distilled-patch16-224", "top1": 83.4, "top5": 96.5, "params": "87M"},
    {"name": "DeiT-base 384", "repo": "facebook/deit-base-patch16-384", "top1": 82.9, "top5": 96.2, "params": "87M"},
    {"name": "DeiT-base distilled 384", "repo": "facebook/deit-base-distilled-patch16-384", "top1": 85.2, "top5": 97.2, "params": "88M"},
]

# Load dataset once
ds = load_dataset("pouya-haghi/imagenet-1k")

def original(module, input, output) -> torch.Tensor:
    hidden_states, head_mask = input
    hidden_states_norm = module.layernorm_before(hidden_states)
    attention_output = module.attention(hidden_states_norm, head_mask)
    # first residual connection
    hidden_states = attention_output + hidden_states
    # in ViT, layernorm is also applied after self-attention
    layer_output = module.layernorm_after(hidden_states)
    layer_output = module.intermediate(layer_output)
    # second residual connection is done here
    layer_output = module.output(layer_output, hidden_states) + hidden_states

    return layer_output


# Hook functions from notebook
def skip_layer_atten(module, input, output):
    hidden_states, head_mask = input
    hidden_states_norm = module.layernorm_before(hidden_states)
    # attention_output = module.attention(hidden_states_norm, head_mask)
    # first residual connection
    # hidden_states = attention_output + hidden_states
    # in ViT, layernorm is also applied after self-attention
    layer_output = module.layernorm_after(hidden_states)
    layer_output = module.intermediate(layer_output)
    # second residual connection is done here
    layer_output = module.output(layer_output, hidden_states) + hidden_states

    return layer_output

def skip_layer_ffn(module, input, output):
    hidden_states, head_mask = input
    hidden_states_norm = module.layernorm_before(hidden_states)
    attention_output = module.attention(hidden_states_norm, head_mask)
    # first residual connection
    hidden_states = attention_output + hidden_states
    # in ViT, layernorm is also applied after self-attention
    # layer_output = module.layernorm_after(hidden_states)
    # layer_output = module.intermediate(layer_output)
    # # second residual connection is done here
    # layer_output = module.output(layer_output, hidden_states) + hidden_states

    return hidden_states

def skip_layer_attention_residual(module, input, output):
    hidden_states, head_mask = input
    hidden_states_norm = module.layernorm_before(hidden_states)
    attention_output = module.attention(hidden_states_norm, head_mask)
    # first residual connection
    hidden_states = attention_output #+ hidden_states
    # in ViT, layernorm is also applied after self-attention
    layer_output = module.layernorm_after(hidden_states)
    layer_output = module.intermediate(layer_output)
    # second residual connection is done here
    layer_output = module.output(layer_output, hidden_states) + hidden_states

    return layer_output

def skip_layer_ffn_residual(module, input, output):
    hidden_states, head_mask = input
    hidden_states_norm = module.layernorm_before(hidden_states)
    attention_output = module.attention(hidden_states_norm, head_mask)
    # first residual connection
    hidden_states = attention_output + hidden_states
    # in ViT, layernorm is also applied after self-attention
    layer_output = module.layernorm_after(hidden_states)
    layer_output = module.intermediate(layer_output)
    # second residual connection is done here
    layer_output = module.output(layer_output, hidden_states) #+ hidden_states

    return layer_output

def skip_layer_ffn_residual_no_residual_attention(module, input, output):
    hidden_states, head_mask = input
    hidden_states_norm = module.layernorm_before(hidden_states)
    attention_output = module.attention(hidden_states_norm, head_mask)
    # first residual connection
    hidden_states = attention_output #+ hidden_states
    # in ViT, layernorm is also applied after self-attention
    layer_output = module.layernorm_after(hidden_states)
    layer_output = module.intermediate(layer_output)
    # second residual connection is done here
    layer_output = module.output(layer_output, hidden_states) #+ hidden_states

    return layer_output

def full_layer_skip(module, input, output) -> torch.Tensor:
    hidden_states, head_mask = input
    # hidden_states_norm = module.layernorm_before(hidden_states)
    # attention_output = module.attention(hidden_states_norm, head_mask)
    # # first residual connection
    # hidden_states = attention_output + hidden_states
    # # in ViT, layernorm is also applied after self-attention
    # layer_output = module.layernorm_after(hidden_states)
    # layer_output = module.intermediate(layer_output)
    # # second residual connection is done here
    # layer_output = module.output(layer_output, hidden_states) + hidden_states

    return hidden_states

def evaluate(model, processor, batch_size=16):
    correct = 0
    train_ds = ds['train']
    for i in range(0, len(train_ds), batch_size):
        batch = train_ds[i : i + batch_size]
        images = [img.convert("RGB") for img in batch['image']]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = outputs.logits.argmax(-1).cpu()
        labels = torch.tensor(batch['label']).cpu()
        correct += (predictions == labels).sum().item()
    return np.round((correct / len(train_ds)) * 100, 2)

hook_fns = [
    skip_layer_atten, 
    skip_layer_attention_residual, 
    skip_layer_ffn, 
    skip_layer_ffn_residual, 
    skip_layer_ffn_residual_no_residual_attention,
    full_layer_skip
]

all_results = []
batch_size = 32

for model_info in models_to_test:
    print(f"Testing model: {model_info['name']}")
    processor = AutoImageProcessor.from_pretrained(model_info['repo'], use_fast=True)
    MODEL_CLASS = DeiTForImageClassificationWithTeacher if "distilled" in model_info['repo'] else ViTForImageClassification
    model = MODEL_CLASS.from_pretrained(model_info['repo']).to(device).eval()
    vit = model.deit if "distilled" in model_info['repo'] else model.vit
    baseline_acc = evaluate(model, processor, batch_size=batch_size)
    print(f"Baseline Accuracy: {baseline_acc}%")

    num_layers = model.config.num_hidden_layers
    for layer_idx in range(num_layers):
        for hook_fn in hook_fns:
            handle = vit.encoder.layer[layer_idx].register_forward_hook(hook_fn)
            acc = evaluate(model, processor, batch_size=batch_size)
            handle.remove()
            
            all_results.append({
                'model_name': model_info['name'],
                'repo': model_info['repo'],
                'imagenet_top1': model_info['top1'],
                'imagenet_top5': model_info['top5'],
                'params': model_info['params'],
                'layer': layer_idx,
                'hook': hook_fn.__name__,
                'accuracy': acc,
                'baseline_accuracy': baseline_acc
            })
        print(f" -> Completed layer {layer_idx}/{num_layers}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("experiment_results.csv", index=False)

print(f"Finished testing model: {model_info['name']}")