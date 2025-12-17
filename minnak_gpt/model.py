from minnak_gpt.nn.layers import (
    Module,
    EmbeddingLayer,
    LinearLayer,
    PositionalEncodingLayer,
    SoftmaxLayer
)
from minnak_gpt.nn.blocks import EncoderBlock, DecoderBlock


class Transformer(Module):
    """Complete encoder-decoder transformer architecture."""
    
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_len, dropout=0.1):
        self.src_embedding = EmbeddingLayer(vocab_size, embed_dim)
        self.tgt_embedding = EmbeddingLayer(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncodingLayer(seq_len, embed_dim)

        self.encoder_layers = [
            EncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ]

        self.decoder_layers = [
            DecoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ]

        self.final_linear = LinearLayer(embed_dim, vocab_size)
        self.softmax = SoftmaxLayer(axis=-1)

    # --- FORWARD: Encodes source, decodes target, returns logits ---
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encoder: embed + position + N encoder layers
        src_emb = self.pos_encoding(self.src_embedding(src))
        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)

        # Decoder: embed + position + N decoder layers
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt))
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)

        # Output projection to vocabulary
        logits = self.final_linear(dec_output)
        return logits
