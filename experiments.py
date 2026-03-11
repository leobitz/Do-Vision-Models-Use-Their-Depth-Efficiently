# experiments.py - Utility functions for ViT layer analysis experiments
"""
This module contains helper functions for analyzing Vision Transformer layers.
Experiments include:
1. Linear Probing (per-layer classification accuracy)
2. CKA (Centered Kernel Alignment) for representation similarity
3. Layer Skipping (ablation)
4. Logit Lens (early exit confidence)
5. Residual Similarity (input/output cosine similarity)
6. Attention Analysis (distance, entropy)
7. Effective Dimensionality (PCA)
8. Noise Sensitivity
9. Positional Information Decoding
10. Positional Ablation / Shuffling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.metrics import accuracy_score
from tqdm import tqdm


# ============================================================================
# 1. LINEAR PROBING
# ============================================================================
def extract_cls_tokens(model, dataloader, device='cuda'):
    """
    Extract CLS token representations from each layer for all images in dataloader.
    Returns: dict with layer_idx -> (N, D) tensor of CLS tokens, and labels (N,)
    """
    model.eval()
    all_cls = {i: [] for i in range(len(model.vit.encoder.layer))}
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting CLS tokens"):
            images = batch['pixel_values'].to(device)
            labels = batch['labels']
            
            outputs = model(images)
            bags = outputs.bags
            
            for i, bag in enumerate(bags):
                # layer_output is the output after FFN + residual
                cls_token = bag['layer_output'][:, 0, :].cpu()
                all_cls[i].append(cls_token)
            
            all_labels.append(labels)
    
    # Concatenate all batches
    for i in all_cls:
        all_cls[i] = torch.cat(all_cls[i], dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    return all_cls, all_labels


def linear_probe_accuracy(cls_tokens, labels, train_ratio=0.8):
    """
    Train a linear classifier on CLS tokens and return accuracy.
    """
    n = len(labels)
    n_train = int(n * train_ratio)
    
    # Shuffle
    perm = np.random.permutation(n)
    X = cls_tokens[perm]
    y = labels[perm]
    
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    return accuracy_score(y_test, y_pred)


# ============================================================================
# 2. CKA (Centered Kernel Alignment)
# ============================================================================
def centering(K):
    """Center a kernel matrix."""
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n
    return H @ K @ H


def linear_kernel(X):
    """Compute linear kernel."""
    return X @ X.T


def rbf_kernel(X, sigma=None):
    """Compute RBF kernel."""
    GX = X @ X.T
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = np.sqrt(mdist)
    KX = np.exp(-KX / (2 * sigma ** 2))
    return KX


def linear_CKA(X, Y):
    """
    Compute Linear CKA between two representation matrices.
    X, Y: (N, D) numpy arrays
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    hsic_xy = np.linalg.norm(X.T @ Y, ord='fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, ord='fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, ord='fro') ** 2
    
    return hsic_xy / (np.sqrt(hsic_xx) * np.sqrt(hsic_yy) + 1e-10)


def compute_cka_matrix(representations):
    """
    Compute CKA matrix for all layer pairs.
    representations: dict with layer_idx -> (N, D) numpy array
    Returns: (num_layers, num_layers) CKA matrix
    """
    num_layers = len(representations)
    cka_matrix = np.zeros((num_layers, num_layers))
    
    for i in range(num_layers):
        for j in range(num_layers):
            cka_matrix[i, j] = linear_CKA(representations[i], representations[j])
    
    return cka_matrix


# ============================================================================
# 3. LAYER SKIPPING (Ablation)
# ============================================================================
def evaluate_with_layer_skip(model, dataloader, skip_layer_idx, device='cuda'):
    """
    Evaluate model accuracy when skipping a specific layer.
    This requires modifying the forward pass.
    """
    model.eval()
    correct = 0
    total = 0
    
    # Store original forward
    original_forward = model.vit.encoder.layer[skip_layer_idx].forward
    
    # Define skip forward (identity)
    def skip_forward(hidden_states, head_mask=None):
        bag = {'hidden_states': hidden_states, 'layer_output': hidden_states}
        return hidden_states, bag
    
    # Replace forward
    model.vit.encoder.layer[skip_layer_idx].forward = skip_forward
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Skipping layer {skip_layer_idx}"):
            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(images)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    # Restore original forward
    model.vit.encoder.layer[skip_layer_idx].forward = original_forward
    
    return correct / total


# ============================================================================
# 4. LOGIT LENS (Early Exit Confidence)
# ============================================================================
def logit_lens_analysis(model, dataloader, device='cuda'):
    """
    Apply final classifier to intermediate layer outputs.
    Returns: dict with layer_idx -> accuracy at that layer
    """
    model.eval()
    num_layers = len(model.vit.encoder.layer)
    
    # Get final layernorm and classifier
    layernorm = model.vit.layernorm
    classifier = model.classifier
    
    layer_correct = {i: 0 for i in range(num_layers)}
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Logit Lens"):
            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(images)
            bags = outputs.bags
            
            for i, bag in enumerate(bags):
                hidden = bag['layer_output']
                hidden_norm = layernorm(hidden)
                cls_token = hidden_norm[:, 0, :]
                logits = classifier(cls_token)
                preds = logits.argmax(dim=-1)
                layer_correct[i] += (preds == labels).sum().item()
            
            total += labels.size(0)
    
    return {i: layer_correct[i] / total for i in range(num_layers)}


# ============================================================================
# 5. RESIDUAL SIMILARITY
# ============================================================================
def compute_residual_similarity(model, dataloader, device='cuda'):
    """
    Compute cosine similarity between input and output of each layer.
    Also compute the norm ratio ||f(x)|| / ||x||.
    """
    model.eval()
    num_layers = len(model.vit.encoder.layer)
    
    cos_sims = {i: [] for i in range(num_layers)}
    norm_ratios = {i: [] for i in range(num_layers)}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Residual Similarity"):
            images = batch['pixel_values'].to(device)
            outputs = model(images)
            bags = outputs.bags
            
            for i, bag in enumerate(bags):
                x_in = bag['hidden_states']  # Input to the block
                x_out = bag['layer_output']  # Output of the block
                f_x = x_out - x_in  # The residual contribution
                
                # Compute cosine similarity (per token, then average)
                cos = F.cosine_similarity(x_in, x_out, dim=-1).mean().item()
                cos_sims[i].append(cos)
                
                # Compute norm ratio
                f_norm = f_x.norm(dim=-1).mean().item()
                x_norm = x_in.norm(dim=-1).mean().item()
                norm_ratios[i].append(f_norm / (x_norm + 1e-10))
    
    # Average over all batches
    avg_cos = {i: np.mean(cos_sims[i]) for i in range(num_layers)}
    avg_norm = {i: np.mean(norm_ratios[i]) for i in range(num_layers)}
    
    return avg_cos, avg_norm


# ============================================================================
# 6. ATTENTION ANALYSIS
# ============================================================================
def compute_attention_metrics(model, dataloader, device='cuda', num_patches_per_side=14):
    """
    Compute mean attention distance and attention entropy per layer.
    """
    model.eval()
    num_layers = len(model.vit.encoder.layer)
    
    # Create distance matrix for patches (excluding CLS)
    # Patches are arranged in a grid
    coords = np.stack(np.meshgrid(range(num_patches_per_side), range(num_patches_per_side)), axis=-1)
    coords = coords.reshape(-1, 2)  # (196, 2)
    
    # Add CLS token at position (0, 0) or treat separately
    # For simplicity, we'll compute distance only for patch-to-patch attention
    
    # Compute pairwise distances
    dist_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1))
    dist_matrix = torch.tensor(dist_matrix, device=device).float()
    
    mean_distances = {i: [] for i in range(num_layers)}
    entropies = {i: [] for i in range(num_layers)}
    
    # We need to hook into attention to get attention weights
    # For now, assume the model returns attention weights in bags
    # If not, we need to modify the model or use hooks
    
    # Note: The current model doesn't return attention weights in bags
    # We would need to add hooks or modify ViTSelfAttention to return them
    # For this implementation, we'll skip this if attention weights aren't available
    
    print("Note: Attention metrics require attention weights to be saved. "
          "Please ensure the model is modified to return attention_probs in bags.")
    
    return mean_distances, entropies


# ============================================================================
# 7. EFFECTIVE DIMENSIONALITY (PCA)
# ============================================================================
def compute_effective_dimensionality(representations, variance_threshold=0.99):
    """
    Compute effective dimensionality (number of PCs to explain 99% variance).
    representations: dict with layer_idx -> (N, D) numpy array
    """
    from sklearn.decomposition import PCA
    
    num_layers = len(representations)
    eff_dims = {}
    
    for i in range(num_layers):
        X = representations[i]
        X = X - X.mean(axis=0)  # Center
        
        pca = PCA(n_components=min(X.shape[0], X.shape[1]))
        pca.fit(X)
        
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        eff_dim = np.searchsorted(cumvar, variance_threshold) + 1
        eff_dims[i] = eff_dim
    
    return eff_dims


# ============================================================================
# 8. NOISE SENSITIVITY
# ============================================================================
def compute_noise_sensitivity(model, dataloader, device='cuda', noise_scale=0.1):
    """
    Inject noise at each layer and measure KL divergence from original prediction.
    """
    model.eval()
    num_layers = len(model.vit.encoder.layer)
    
    kl_divs = {i: [] for i in range(num_layers)}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Noise Sensitivity"):
            images = batch['pixel_values'].to(device)
            
            # Get original predictions
            orig_outputs = model(images)
            orig_logits = orig_outputs.logits
            orig_probs = F.softmax(orig_logits, dim=-1)
            
            # For each layer, inject noise
            for layer_idx in range(num_layers):
                # We need to hook into the layer to inject noise
                # For simplicity, we'll use a forward hook
                
                def make_noise_hook(scale):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            noisy = output[0] + torch.randn_like(output[0]) * scale
                            return (noisy,) + output[1:]
                        return output + torch.randn_like(output) * scale
                    return hook
                
                # Register hook
                handle = model.vit.encoder.layer[layer_idx].register_forward_hook(
                    make_noise_hook(noise_scale)
                )
                
                # Forward with noise
                noisy_outputs = model(images)
                noisy_logits = noisy_outputs.logits
                noisy_probs = F.softmax(noisy_logits, dim=-1)
                
                # Compute KL divergence
                kl = F.kl_div(noisy_probs.log(), orig_probs, reduction='batchmean').item()
                kl_divs[layer_idx].append(kl)
                
                # Remove hook
                handle.remove()
    
    # Average KL divergence per layer
    avg_kl = {i: np.mean(kl_divs[i]) for i in range(num_layers)}
    return avg_kl


# ============================================================================
# 9. POSITIONAL INFORMATION DECODING
# ============================================================================
def decode_positional_info(model, dataloader, device='cuda', num_patches_per_side=14):
    """
    Train a linear regressor to predict (x, y) coordinates from patch tokens.
    Returns R^2 score per layer.
    """
    model.eval()
    num_layers = len(model.vit.encoder.layer)
    num_patches = num_patches_per_side ** 2
    
    # Create target coordinates (normalized to [0, 1])
    coords = np.stack(np.meshgrid(
        np.linspace(0, 1, num_patches_per_side),
        np.linspace(0, 1, num_patches_per_side)
    ), axis=-1).reshape(-1, 2)
    
    # Collect patch tokens per layer
    patch_tokens = {i: [] for i in range(num_layers)}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting patch tokens"):
            images = batch['pixel_values'].to(device)
            outputs = model(images)
            bags = outputs.bags
            
            for i, bag in enumerate(bags):
                # Exclude CLS token (index 0)
                tokens = bag['layer_output'][:, 1:, :].cpu().numpy()  # (B, 196, D)
                patch_tokens[i].append(tokens)
    
    # Concatenate and train regressor
    r2_scores = {}
    for i in range(num_layers):
        X = np.concatenate(patch_tokens[i], axis=0)  # (B*196, D)
        X = X.reshape(-1, X.shape[-1])  # Flatten batch and patches
        
        # Repeat coordinates for each image
        num_images = X.shape[0] // num_patches
        y = np.tile(coords, (num_images, 1))
        
        # Split
        n = len(y)
        n_train = int(n * 0.8)
        perm = np.random.permutation(n)
        X, y = X[perm], y[perm]
        
        reg = Ridge(alpha=1.0)
        reg.fit(X[:n_train], y[:n_train])
        r2_scores[i] = reg.score(X[n_train:], y[n_train:])
    
    return r2_scores


# ============================================================================
# 10. POSITIONAL ABLATION / SHUFFLING
# ============================================================================
def evaluate_with_position_ablation(model, dataloader, device='cuda', ablation_type='zero'):
    """
    Evaluate model with positional embeddings zeroed or shuffled.
    ablation_type: 'zero' or 'shuffle'
    """
    model.eval()
    
    # Store original position embeddings
    original_pos_embed = model.vit.embeddings.position_embeddings.data.clone()
    
    if ablation_type == 'zero':
        model.vit.embeddings.position_embeddings.data.zero_()
    elif ablation_type == 'shuffle':
        # Shuffle patch positions (keep CLS at index 0)
        perm = torch.randperm(original_pos_embed.shape[1] - 1) + 1
        perm = torch.cat([torch.tensor([0]), perm])
        model.vit.embeddings.position_embeddings.data = original_pos_embed[:, perm, :]
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Position {ablation_type}"):
            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(images)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    # Restore original
    model.vit.embeddings.position_embeddings.data = original_pos_embed
    
    return correct / total


# ============================================================================
# BASELINE ACCURACY
# ============================================================================
def compute_baseline_accuracy(model, dataloader, device='cuda'):
    """Compute baseline accuracy without any modifications."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Baseline"):
            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(images)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total
