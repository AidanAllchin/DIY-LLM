from typing import Dict

import torch
from torch import nn

from src.attention import MultiHeadAttention

class GELU(nn.Module):
    """
    Gaussian Error Linear Unit.

    Leads to better optimization through nuanced adjustments to parameters.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * ( 1 + torch.tanh(  # an approximation of x * \Phi(x)
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    """
    Feed forward network.
    """
    def __init__(self, embed_dim: int):
        super().__init__()

        # Input enters and leaves as (batch_size, number_of_tokens, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            GELU(),  # dim=2 is 4x larger than embed_dim
            nn.Linear(4 * embed_dim, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg: Dict[str, int | bool]):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["embed_dim"],
            d_out=cfg["embed_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=bool(cfg["qkv_bias"])
        )
        self.ff = FeedForward(cfg["embed_dim"])
        self.norm1 = LayerNorm(cfg["embed_dim"])
        self.norm2 = LayerNorm(cfg["embed_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shortcut for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = shortcut + x

        # Shortcut for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = shortcut + x
        return x

class LayerNorm(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.eps = 1e-5  # Added to var to avoid division by zero
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        std = torch.sqrt(var + self.eps)
        return self.scale * (x - mean) / std + self.shift

class GPT2Ripoff(nn.Module):
    def __init__(self, cfg: Dict[str, int | bool]):
        super().__init__()

        self.tok_emb  = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"])
        self.pos_emb  = nn.Embedding(cfg["context_length"], cfg["embed_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["embed_dim"])
        self.out_head   = nn.Linear(cfg["embed_dim"], cfg["vocab_size"], bias=False)
        
    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape
        token_embeds = self.tok_emb(in_idx)

        # Add positional embeddings
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = token_embeds + pos_embeds

        # Apply dropout to embeddings
        x = self.drop_emb(x)

        # Pass through transformer blocks
        x = self.trf_blocks(x)

        # Final layer normalization
        x = self.final_norm(x)

        # Project to vocab size
        logits = self.out_head(x)
        return logits
