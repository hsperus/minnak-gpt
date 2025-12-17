import numpy as np
from minnak_gpt.engine.tensor import Tensor


# --- RELU: Returns max(0, x) with gradient support ---
def relu(x):
    out = Tensor(np.maximum(0, x.data), (x,), 'relu')
    
    def _backward():
        x.grad += (x.data > 0) * out.grad
    out._backward = _backward
    return out


# --- SOFTMAX: Converts values to probability distribution ---
def softmax(x, axis=-1):
    shifted_data = x.data - np.max(x.data, axis=axis, keepdims=True)
    exp_tensor = x.exp()
    sum_exp = exp_tensor.sum(axis=axis, keepdims=True)
    return exp_tensor / sum_exp
