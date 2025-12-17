import numpy as np
from minnak_gpt.engine.tensor import Tensor
from minnak_gpt.nn import functional as F


class Module:
    """Base class for all neural network layers."""
    
    # --- PARAMETERS: Collects all trainable tensors recursively ---
    def parameters(self):
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, Tensor) and attr.requires_grad:
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class EmbeddingLayer(Module):
    """Maps token IDs to dense vectors."""
    
    def __init__(self, vocab_size, embed_dim):
        self.weight = Tensor(np.random.randn(vocab_size, embed_dim) * 0.01, label="embed_w")

    # --- FORWARD: Looks up embeddings for input token IDs ---
    def forward(self, input_ids):
        input_ids = np.array(input_ids)
        out_data = self.weight.data[input_ids]
        out = Tensor(out_data, _children=(self.weight,), _op='embedding')

        def _backward():
            if self.weight.requires_grad:
                grad_weight = np.zeros_like(self.weight.data)
                np.add.at(grad_weight, input_ids, out.grad)
                self.weight.grad += grad_weight
        out._backward = _backward
        return out


class PositionalEncodingLayer(Module):
    """Adds sinusoidal position information to embeddings."""
    
    def __init__(self, seq_len, embed_dim):
        pe = np.zeros((seq_len, embed_dim))
        position = np.arange(0, seq_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = Tensor(pe, requires_grad=False, label="pos_encoding")

    # --- FORWARD: Adds positional encoding to input ---
    def forward(self, x):
        return x + self.pe


class LinearLayer(Module):
    """Fully connected layer: y = xW + b"""
    
    def __init__(self, in_features, out_features, bias=True):
        self.weight = Tensor(np.random.randn(in_features, out_features) * 0.01, label="linear_w")
        self.bias = Tensor(np.zeros(out_features), label="linear_b") if bias else None

    # --- FORWARD: Applies linear transformation ---
    def forward(self, x):
        out = x.matmul(self.weight)
        if self.bias:
            out = out + self.bias
        return out


class LayerNormLayer(Module):
    """Normalizes activations across the last dimension."""
    
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = Tensor(np.ones(dim), label="ln_gamma")
        self.beta = Tensor(np.zeros(dim), label="ln_beta")

    # --- FORWARD: Normalizes and scales input ---
    def forward(self, x):
        mean = x.sum(axis=-1, keepdims=True) / x.data.shape[-1]
        var = ((x - mean)**2).sum(axis=-1, keepdims=True) / x.data.shape[-1]
        x_hat = (x - mean) / (var + self.eps)**0.5
        return x_hat * self.gamma + self.beta


class DropoutLayer(Module):
    """Randomly zeros activations for regularization."""
    
    def __init__(self, p=0.1):
        self.p = p

    # --- FORWARD: Applies dropout mask during training ---
    def forward(self, x, training=True):
        if not training: 
            return x
        mask = (np.random.rand(*x.data.shape) > self.p) / (1 - self.p)
        out = Tensor(x.data * mask, (x,), 'dropout')
        
        def _backward():
            x.grad += mask * out.grad
        out._backward = _backward
        return out


class ReLULayer(Module):
    """ReLU activation layer."""
    
    def forward(self, x):
        return F.relu(x)


class SoftmaxLayer(Module):
    """Softmax activation layer."""
    
    def __init__(self, axis=-1):
        self.axis = axis

    def forward(self, x):
        return F.softmax(x, axis=self.axis)
