import numpy as np
from minnak_gpt.engine.tensor import Tensor
from minnak_gpt.nn.layers import (
    Module,
    LinearLayer, 
    LayerNormLayer, 
    DropoutLayer, 
    SoftmaxLayer
)


class MultiHeadAttention(Module):
    """Computes parallel attention over multiple heads."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = LinearLayer(embed_dim, embed_dim)
        self.k_proj = LinearLayer(embed_dim, embed_dim)
        self.v_proj = LinearLayer(embed_dim, embed_dim)
        self.out_proj = LinearLayer(embed_dim, embed_dim)
        
        self.softmax = SoftmaxLayer(axis=-1)
        self.dropout = DropoutLayer(dropout)

    # --- FORWARD: Computes scaled dot-product attention across heads ---
    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.data.shape
        
        q_vec = self.q_proj(q)
        k_vec = self.k_proj(k)
        v_vec = self.v_proj(v)

        q_vec = q_vec.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose()
        k_vec = k_vec.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose()
        v_vec = v_vec.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose()

        scores = q_vec.matmul(k_vec.transpose()) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores.data = np.where(mask.data == 0, -1e9, scores.data)

        attn = self.softmax(scores)
        context = attn.matmul(v_vec).transpose().reshape(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(self.dropout(context))


class FeedForward(Module):
    """Two-layer MLP applied to each position independently."""
    
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        self.linear1 = LinearLayer(embed_dim, ff_dim)
        self.linear2 = LinearLayer(ff_dim, embed_dim)
        self.dropout = DropoutLayer(dropout)

    # --- FORWARD: Linear -> ReLU -> Linear -> Dropout ---
    def forward(self, x):
        from minnak_gpt.nn import functional as F
        return self.dropout(self.linear2(F.relu(self.linear1(x))))


class EncoderBlock(Module):
    """Single transformer encoder layer with self-attention."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = LayerNormLayer(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2 = LayerNormLayer(embed_dim)

    # --- FORWARD: Self-Attention + FFN with residual connections ---
    def forward(self, x, mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, mask))
        x = self.norm2(x + self.ff(x))
        return x


class DecoderBlock(Module):
    """Single transformer decoder layer with self and cross attention."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = LayerNormLayer(embed_dim)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = LayerNormLayer(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.norm3 = LayerNormLayer(embed_dim)

    # --- FORWARD: Self-Attention + Cross-Attention + FFN ---
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
        x = self.norm2(x + self.cross_attn(x, enc_output, enc_output, src_mask))
        x = self.norm3(x + self.ff(x))
        return x
