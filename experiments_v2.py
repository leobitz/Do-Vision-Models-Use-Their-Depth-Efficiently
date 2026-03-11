
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from modeling_vit import ViTSelfAttention

# ============================================================================
# 1. PATCH SHUFFLE AT DIFFERENT DEPTHS
# ============================================================================
def patch_shuffle_at_depth(model, dataloader, depths_to_test, device='cuda'):
    """
    Measure accuracy when shuffling patches after layer k.
    This effectively destroys spatial information at that depth.
    """
    model.eval()
    results = {}

    # Wrapper to perform shuffling during forward pass
    class ShuffleHook:
        def __init__(self, layer_idx):
            self.layer_idx = layer_idx
            
        def __call__(self, module, input, output):
            # output is (hidden_states, bag)
            # hidden_states: (B, N, D)
            hidden_states = output[0]
            
            # Keep CLS token (index 0) fixed, shuffle others
            # We want to shuffle the patches *within* each image independently effectively?
            # Or just shuffle the sequence.
            # Standard patch shuffle shuffles the sequence dimension 1..N
            
            B, N, D = hidden_states.shape
            # Create a random permutation for patches (indices 1 to N-1)
            # We use the same permutation for all batch elements or different?
            # Usually different is stronger, but same is easier to implement vectorized.
            # Let's do per-sample shuffle for rigor.
            
            # Efficient implementation:
            # Generate random noise and argsort
            perm = torch.rand(B, N-1, device=hidden_states.device).argsort(dim=1) + 1
            
            # Combine with CLS index (0)
            cls_idx = torch.zeros(B, 1, dtype=torch.long, device=hidden_states.device)
            full_perm = torch.cat([cls_idx, perm], dim=1)
            
            # Gather
            # hidden_states: (B, N, D)
            # full_perm: (B, N) -> expand to (B, N, D)
            shuffled_states = torch.gather(hidden_states, 1, full_perm.unsqueeze(-1).expand(-1, -1, D))
            
            # Update the tuple
            return (shuffled_states,) + output[1:]

    for depth in depths_to_test:
        # Register hook at 'depth'
        # The hook should be applied *after* the layer processing
        # We can hook onto the layer module itself
        hook_handle = model.vit.encoder.layer[depth].register_forward_hook(ShuffleHook(depth))
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Shuffle @ Layer {depth}"):
                images = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(images)
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        hook_handle.remove()
        results[depth] = correct / total
        
    return results

# ============================================================================
# 2. RELATIVE POSITIONAL ENCODING / RoPE SWAP
# ============================================================================
class RoPESelfAttention(nn.Module):
    """
    Self-Attention with Rotary Positional Embeddings (RoPE).
    Replaces ViTSelfAttention.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.original_attn = ViTSelfAttention(config) # Use logic from original
        # We need to replicate init because we are basically wrapping or monkey patching
        # Actually easier to just subclass or rewrite forward.
        # Let's perform a runtime check to see if we can reuse the weights
        
    def forward(self, hidden_states, head_mask=None):
        # We need the weights from the original attention mechanism
        # This wrapper expects to be assigned the existing weights
        pass
        
def apply_rotary_emb(x, freqs_cis):
    # x: (B, H, L, D)
    # freqs_cis: (L, D/2) (complex)
    # This is a simplified version; normally we need to handle shapes carefully
    # Implementing full RoPE from scratch is complex.
    # We can use a simplified relative bias instead as the prompt suggested "Relative vs Absolute"
    pass

def evaluate_with_rope(model, dataloader, device='cuda'):
    """
    Replace Absolute Positional Embeddings with Rotary Embeddings (RoPE).
    """
    model.eval()
    
    # 1. Zero out absolute positional embeddings
    original_pos_embed = model.vit.embeddings.position_embeddings.data.clone()
    model.vit.embeddings.position_embeddings.data.zero_()
    
    # 2. Monkey patch SelfAttention to use RoPE
    # We define a function that applies RoPE to Q and K before attention
    
    def rope_attention_forward_hook(module, args, output):
        # We can't easily hook *inside* the forward to change Q, K without rewriting the class.
        # We have to replace the `forward` method of the ViTSelfAttention class or instances.
        pass

    # Better approach: Define a custom forward function and bind it to the instances
    from transformers.models.vit.modeling_vit import eager_attention_forward # Helper
    
    # Create simple 1D RoPE frequencies
    # Dim d, Seq Len N
    # theta_i = 10000^(-2(i-1)/d)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    seq_len = 197 # 14x14 + 1
    
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim)).to(device)
    t = torch.arange(seq_len, device=device).type_as(inv_freq)
    freqs = torch.einsum('i,j->ij', t, inv_freq) # (Seq, Dim/2)
    emb = torch.cat((freqs, freqs), dim=-1) # (Seq, Dim)
    # cos, sin
    cos_cached = emb.cos()[None, None, :, :] # (1, 1, Seq, Dim)
    sin_cached = emb.sin()[None, None, :, :]
    
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(q, k, cos, sin):
        # q, k: (B, H, L, D)
        q_rope = (q * cos) + (rotate_half(q) * sin)
        k_rope = (k * cos) + (rotate_half(k) * sin)
        return q_rope, k_rope

    # Save original forward methods
    original_forwards = {}
    for i, layer in enumerate(model.vit.encoder.layer):
        original_forwards[i] = layer.attention.attention.forward
    
    # Define new forward
    def new_forward(self, hidden_states, head_mask=None):
        # Copy from original ViTSelfAttention.forward but add RoPE
        batch_size = hidden_states.shape[0]
        new_shape = batch_size, -1, self.num_attention_heads, self.attention_head_size
        
        key_layer = self.key(hidden_states).view(*new_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*new_shape).transpose(1, 2)
        query_layer = self.query(hidden_states).view(*new_shape).transpose(1, 2)
        
        # --- Apply RoPE ---
        query_layer, key_layer = apply_rope(query_layer, key_layer, cos_cached, sin_cached)
        # ------------------

        # Proceed with standard attention
        # We can call the attention_interface directly, or copy the code
        # Since we are monkey patching a specific model instance, we can rely on eager attention
        
        dropout_prob = 0.0 if not self.training else self.dropout_prob
        
        # Manual Attention
        attn_weights = torch.matmul(query_layer, key_layer.transpose(-1, -2)) * self.scaling
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout_prob, training=self.training)
        
        if head_mask is not None:
             attn_weights = attn_weights * head_mask
             
        context_layer = torch.matmul(attn_weights, value_layer)
        context_layer = context_layer.transpose(1, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)
        
        return context_layer, attn_weights

    # Patch modules
    for i, layer in enumerate(model.vit.encoder.layer):
        # We bind the new method to the instance
        # Use simple assignment of bound method
        layer.attention.attention.forward = new_forward.__get__(layer.attention.attention, ViTSelfAttention)

    # Eval
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="RoPE Evaluation"):
            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(images)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # Restore
    model.vit.embeddings.position_embeddings.data = original_pos_embed
    for i, layer in enumerate(model.vit.encoder.layer):
        layer.attention.attention.forward = original_forwards[i]
        
    return correct / total

# ============================================================================
# 3. ATTENTION DISTANCE VS DEPTH
# ============================================================================
def compute_attention_distance(model, dataloader, device='cuda', num_patches_per_side=14):
    """
    Compute mean attention distance per head per layer.
    """
    model.eval()
    num_layers = len(model.vit.encoder.layer)
    num_heads = model.vit.encoder.layer[0].attention.attention.num_attention_heads
    
    # Precompute distance matrix
    # Coords: (14, 14) grid
    coords = np.stack(np.meshgrid(range(num_patches_per_side), range(num_patches_per_side)), axis=-1)
    coords = coords.reshape(-1, 2)
    # Add CLS (-1, -1) or handle separately. 
    # Standard practice: Ignore CLS for distance, or treat as global.
    # Let's compute distance between patches only.
    
    # Distance matrix (196, 196)
    dist_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1))
    dist_matrix = torch.tensor(dist_matrix, device=device).float()
    
    layer_dists = {i: [] for i in range(num_layers)}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Attention Distance"):
            images = batch['pixel_values'].to(device)
            
            outputs = model(images)
            bags = outputs.bags
            
            for i, bag in enumerate(bags):
                # attention_probs: (B, H, N, N)
                # N = 197 (CLS + 196 patches)
                attn = bag['attention_probs']
                
                # Slice out patch-to-patch attention
                # (B, H, 196, 196)
                patch_attn = attn[:, :, 1:, 1:]
                
                # Normalize because we removed CLS
                patch_attn = patch_attn / (patch_attn.sum(dim=-1, keepdim=True) + 1e-10)
                
                # Compute weighted average distance
                # (B, H, 196, 196) * (196, 196) -> sum over last dim
                # expected_dist per query token: (B, H, 196)
                avg_dist = (patch_attn * dist_matrix).sum(dim=-1)
                
                # Mean over patches and batch
                mean_dist = avg_dist.mean(dim=(0, 2)) # (H,)
                layer_dists[i].append(mean_dist.cpu().numpy())
    
    # Average over batches
    final_dists = {}
    for i in range(num_layers):
        # Stack all batches: (M, H)
        dists = np.stack(layer_dists[i], axis=0)
        final_dists[i] = dists.mean(axis=0) # (H,)
        
    return final_dists
