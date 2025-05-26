from torch import nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_in: int,
            d_out: int,
            context_length: int,
            dropout: float,
            num_heads: int,
            qkv_bias: bool = False
        ):
        super().__init__()

        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_projection = nn.Linear(d_out, d_in)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = x.shape

        # Linear projection
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)
        
        # Reshape for multi-head attention
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.transpose(1, 2)
        
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        keys    = keys.transpose(1, 2)
        
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim)
        values  = values.transpose(1, 2)
        
        # Compute attention scores
        scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # type: ignore
        scores = scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(
            scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        
        # Get the context vector by taking the weighted sum of values
        context = (attn_weights @ values).transpose(1, 2)
        context = context.contiguous().view(b, num_tokens, self.d_out)
        
        return self.out_projection(context)
    