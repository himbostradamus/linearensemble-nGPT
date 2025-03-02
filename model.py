import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


def apply_rotary_position_embeddings(sinusoidal_pos, q, k):
    # Split the sinusoidal_pos into sin and cos parts
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
    # Apply the rotary embeddings to the query and key
    q_rot = torch.stack((-q[..., 1::2], q[..., ::2]), dim=-1)
    k_rot = torch.stack((-k[..., 1::2], k[..., ::2]), dim=-1)
    q_rot = torch.reshape(q_rot, q.shape[:-1] + (q.shape[-1]//2, 2)) * torch.stack((cos, sin), dim=-1)
    k_rot = torch.reshape(k_rot, k.shape[:-1] + (k.shape[-1]//2, 2)) * torch.stack((cos, sin), dim=-1)
    q_rot = torch.reshape(q_rot, q.shape)
    k_rot = torch.reshape(k_rot, k.shape)
    return q_rot, k_rot

def get_sinusoidal_embeddings(n_positions, dim):
    """Generate sinusoidal positional embeddings."""
    position = torch.arange(n_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    sinusoidal_emb = torch.zeros((n_positions, dim))
    sinusoidal_emb[:, 0::2] = torch.sin(position * div_term)
    sinusoidal_emb[:, 1::2] = torch.cos(position * div_term)
    return sinusoidal_emb


class RMSNorm(torch.nn.Module):
    def __init__(self, embdim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(embdim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = torch.mean(x * x, dim=-1, keepdim=True)
        xnorm = x * torch.rsqrt(norm + self.eps)
        xnorm = xnorm.to(dtype=dtype)
        return xnorm * self.weight


class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x shape: [B, H, T, D] (batch, heads, sequence_length, head_dim)
        # We want to normalize each head independently
        B, H, T, D = x.shape
        x = x.reshape(B * H, T, D)
        
        # Compute mean and var per head
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), unbiased=False, keepdim=True)
        
        # Normalize
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply weight and bias if affine
        if self.affine:
            x = x * self.weight.view(1, 1, -1) + self.bias.view(1, 1, -1)
        
        # Reshape back
        x = x.reshape(B, H, T, D)
        return x


class LinearEnsembleBlock(nn.Module):
    """
    Transformer block with linear ensemble attention and normalization on the hypersphere
    """
    def __init__(self, config, iblock):
        super().__init__()
        self.config = config
        self.iblock = iblock
        self.num_maps = config.num_maps  # Number of attention maps in the ensemble
        
        # Create attention projections for each map
        self.query_projs = nn.ModuleList([
            nn.Linear(config.n_embd, config.n_embd // config.num_maps, bias=config.bias, dtype=torch.bfloat16)
            for _ in range(config.num_maps)
        ])
        self.key_projs = nn.ModuleList([
            nn.Linear(config.n_embd, config.n_embd // config.num_maps, bias=config.bias, dtype=torch.bfloat16)
            for _ in range(config.num_maps)
        ])
        
        # Single value projection (shared across maps)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        self.att_c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        
        # Raw attention weights with learnable initialization
        self.raw_map_weights = nn.Parameter(torch.zeros(config.num_maps, dtype=torch.float32))
        # Scale for controlling weight impact
        self.weight_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
        
        # Initialize with varying weights
        with torch.no_grad():
            for i in range(config.num_maps):
                # Initialize with alternating positive/negative small values
                self.raw_map_weights[i] = 0.1 * ((-1) ** i)
        
        # Group normalization for attention heads
        self.group_norm = GroupNorm(config.n_head, config.n_embd // config.n_head * config.n_head)
        
        # Feed-forward network
        self.c_fc = nn.Linear(config.n_embd, 2 * 4 * config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        self.silu = nn.SiLU()
        self.mlp_c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)

        # nGPT hyperparameters
        if config.use_nGPT == 1:
            # Attention eigen learning rate
            self.attn_alpha_init_value = 0.05
            self.attn_alpha_init_scaling = config.base_scale
            self.attn_alpha = torch.nn.Parameter(self.attn_alpha_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32))

            # MLP eigen learning rate
            self.mlp_alpha_init_value = 0.05
            self.mlp_alpha_init_scaling = config.base_scale
            self.mlp_alpha = torch.nn.Parameter(self.mlp_alpha_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32))

            # Scaling factors for query-key computation
            self.sqk_init_value = 1.0       
            self.sqk_init_scaling = config.base_scale
            self.sqk = torch.nn.Parameter(self.sqk_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32))

            # Scaling factors for MLP
            self.suv_init_value = 1.0
            self.suv_init_scaling = 1.0
            self.suv = torch.nn.Parameter(self.suv_init_scaling * torch.ones(2 * 4 * config.n_embd, dtype=torch.float32))

    def justnorm(self, x):
        """Normalize a vector to unit norm."""
        res = x / x.norm(p=2, dim=-1, keepdim=True)
        return res

    def forward(self, h):
        B, T, C = h.size()

        # Input for attention computation
        hin = h
        
        # Process each map
        all_attentions = []
        
        # Calculate attention map weights
        map_weights = torch.tanh(self.raw_map_weights) * self.weight_scale
        
        # Apply tanh to get weights in range [-1, 1] and multiply by scale
        # Calculate head dimension based on num_maps
        head_dim = self.config.n_embd // (self.config.num_maps * self.config.n_head)
        
        # Project value (shared across maps)
        v = self.value(hin)
        v = v.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)
        v = v.transpose(1, 2)  # [B, H, T, D]
        
        # Process queries and keys for each map
        for i in range(self.config.num_maps):
            q_i = self.query_projs[i](hin)
            k_i = self.key_projs[i](hin)
            
            # Reshape for multi-head attention
            q_i = q_i.view(B, T, self.config.n_head, head_dim)
            k_i = k_i.view(B, T, self.config.n_head, head_dim)
            
            # Apply rotary position embeddings
            sinusoidal_pos = get_sinusoidal_embeddings(T, head_dim).to(device=q_i.device)
            q_i, k_i = apply_rotary_position_embeddings(sinusoidal_pos, q_i.transpose(1, 2), k_i.transpose(1, 2))
            q_i = q_i.transpose(2, 1)  # [B, H, T, D]
            k_i = k_i.transpose(2, 1)  # [B, H, T, D]
            
            # Apply normalization and scaling to queries and keys if using nGPT
            if self.config.use_nGPT == 1:
                sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(1, 1, self.config.n_head, head_dim)
                q_i = sqk * self.justnorm(q_i)
                k_i = sqk * self.justnorm(k_i)
            
            # Calculate attention scores
            sqrt_head_dim = math.sqrt(head_dim)
            softmax_scale = sqrt_head_dim  # Following nGPT scaling approach
            
            # Compute attention scores
            attn_scores = torch.matmul(q_i, k_i.transpose(-1, -2)) * softmax_scale
            
            # Apply causal mask
            causal_mask = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
            
            # Apply softmax
            attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).type_as(attn_scores)
            
            # Store attention for this map
            all_attentions.append(attn_weights)
        
        # Combine attentions with learned weights
        combined_attn = torch.zeros_like(all_attentions[0])
        for i, attn in enumerate(all_attentions):
            combined_attn = combined_attn + map_weights[i] * attn
        
        # Apply attention to values
        y = torch.matmul(combined_attn, v)
        
        # Apply group normalization
        y = self.group_norm(y)
        
        # Reshape to [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Project to output
        h_att = self.att_c_proj(y)

        # Update with eigen learning rates for nGPT
        if self.config.use_nGPT == 1:
            lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
            lr = torch.abs(lr)
            
            A_norm = self.justnorm(h)
            B_norm = self.justnorm(h_att)
            
            res = A_norm + lr * (B_norm - A_norm)
            h = self.justnorm(res)
        else:
            h = h + h_att

        # Feed-forward network
        hin = h
        
        # MLP computation
        uv = self.c_fc(hin)
        if self.config.use_nGPT == 1:
            suv = (self.suv * ((self.suv_init_value / self.suv_init_scaling) * (self.config.n_embd ** 0.5)))
            uv = suv * uv
        u, v = torch.chunk(uv, 2, dim=-1)
        x_mlp = u * self.silu(v)
        h_mlp = self.mlp_c_proj(x_mlp)

        # Update with eigen learning rates for MLP in nGPT
        if self.config.use_nGPT == 1:
            lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
            lr = torch.abs(lr)

            A_norm = self.justnorm(h)
            B_norm = self.justnorm(h_mlp)
            
            res = A_norm + lr * (B_norm - A_norm)
            h = self.justnorm(res)
        else:
            h = h + h_mlp

        return h


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 1024
    num_maps: int = 3  # Number of attention maps in the linear ensemble
    base_scale: float = 1.0 / (1024.0 ** 0.5)    # 1 / sqrt(n_embd)
    use_nGPT: int = 0
    dropout: float = 0.0
    bias: bool = False 


class LinearEnsembleNGPT(nn.Module):
    """
    Linear Ensemble nGPT: Combining normalized transformer with linear ensemble attention
    """
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([LinearEnsembleBlock(config, il) for il in range(config.n_layer)])
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Init all weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=config.base_scale/math.sqrt(2 * config.n_layer))
        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
        if config.use_nGPT == 1:
            self.sz_init_value = 1.00
            self.sz_init_scaling = config.base_scale
            self.sz = torch.nn.Parameter(self.sz_init_scaling * torch.ones(config.vocab_size, dtype=torch.float32))

        if config.use_nGPT == 0:
            self.rmsnorm_f = RMSNorm(config.n_embd)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()

        # Get token embeddings
        tok_emb = self.transformer.wte(idx)  # [b, t, n_embd]
        
        # Apply dropout
        x = self.transformer.drop(tok_emb)
        
        # Apply transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Apply final normalization if not using nGPT
        if self.config.use_nGPT == 0:
            x = self.rmsnorm_f(x)

        # Loss computation
        if targets is not None:
            # Get logits
            logits = self.lm_head(x)
            
            # Apply scaling in nGPT
            if self.config.use_nGPT == 1:
                sz = self.sz * (self.sz_init_value / self.sz_init_scaling)
                logits = sz * logits
                
            # Compute loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference optimization: only compute logits for the last position
            logits = self.lm_head(x[:, [-1], :])
            
            # Apply scaling in nGPT
            if self.config.use_nGPT == 1:
                sz = self.sz * (self.sz_init_value / self.sz_init_scaling)
                logits = sz * logits
                
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Create optimization groups
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Weight decay: apply to 2D tensors only (weights)
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # Print parameter stats
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Create AdamW optimizer
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        
        return optimizer
    
    def justnorm(self, x, idim=-1):
        """Normalize a vector to unit norm, used for normalization during training."""
        dtype = x.dtype
        x = x.float()
        res = (x / x.norm(p=2, dim=idim, keepdim=True)).to(dtype=dtype) 
        return res

    def normalize_matrices(self):
        """Normalize all matrices to unit norm as in nGPT."""
        if self.config.use_nGPT != 1:
            return
            
        # Normalize embedding and output matrices
        self.transformer.wte.weight.data.copy_(self.justnorm(self.transformer.wte.weight.data, 1))
        self.lm_head.weight.data.copy_(self.justnorm(self.lm_head.weight.data, 1))
        
        # Normalize each layer's matrices
        for layer_idx in range(0, self.config.n_layer):
            block = self.transformer.h[layer_idx]
            
            # Normalize query and key matrices for each map
            for i in range(self.config.num_maps):
                block.query_projs[i].weight.data.copy_(self.justnorm(block.query_projs[i].weight.data, 1))
                block.key_projs[i].weight.data.copy_(self.justnorm(block.key_projs[i].weight.data, 1))
            
            # Normalize value and projection matrices
            block.value.weight.data.copy_(self.justnorm(block.value.weight.data, 1))
            block.att_c_proj.weight.data.copy_(self.justnorm(block.att_c_proj.weight.data, 0))
            
            # Normalize MLP matrices
            block.c_fc.weight.data.copy_(self.justnorm(block.c_fc.weight.data, 1))
            block.mlp_c_proj.weight.data.copy_(self.justnorm(block.mlp_c_proj.weight.data, 0))
