import numpy as np


class Tensor:
    """Core data structure with automatic differentiation support."""
    
    def __init__(self, data, _children=(), _op='', label='', requires_grad=True):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op 
        self.label = label

    # --- BROADCASTING HANDLER: Fixes gradient shapes after broadcast operations ---
    @staticmethod
    def _handle_broadcasting(tensor, grad):
        while grad.ndim > tensor.data.ndim:
            grad = grad.sum(axis=0)
        for i, dim in enumerate(tensor.data.shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    # --- ADDITION: Element-wise add with gradient tracking ---
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            if self.requires_grad:
                self.grad += self._handle_broadcasting(self, out.grad)
            if other.requires_grad:
                other.grad += self._handle_broadcasting(other, out.grad)
        out._backward = _backward
        return out

    # --- MULTIPLICATION: Element-wise multiply with gradient tracking ---
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            if self.requires_grad:
                grad_self = other.data * out.grad
                self.grad += self._handle_broadcasting(self, grad_self)
            if other.requires_grad:
                grad_other = self.data * out.grad
                other.grad += self._handle_broadcasting(other, grad_other)
        out._backward = _backward
        return out

    # --- MATMUL: Matrix multiplication for linear layers and attention ---
    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data, other.data), (self, other), 'matmul')

        def _backward():
            if self.requires_grad:
                self.grad += np.matmul(out.grad, other.data.swapaxes(-1, -2))
            if other.requires_grad:
                other.grad += np.matmul(self.data.swapaxes(-1, -2), out.grad)
        out._backward = _backward
        return out

    # --- SUM: Reduces tensor along specified axis ---
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')

        def _backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    # --- EXP: Exponential function for softmax ---
    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')

        def _backward():
            if self.requires_grad:
                self.grad += out.data * out.grad
        out._backward = _backward
        return out

    # --- POWER: Raises tensor to a power ---
    def __pow__(self, n):
        out = Tensor(self.data**n, (self,), f'pow_{n}')

        def _backward():
            if self.requires_grad:
                self.grad += (n * self.data**(n-1)) * out.grad
        out._backward = _backward
        return out

    # --- RESHAPE: Changes tensor shape for multi-head splitting ---
    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), 'reshape')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    # --- TRANSPOSE: Swaps axes for attention computation ---
    def transpose(self, axis1=-1, axis2=-2):
        out = Tensor(self.data.swapaxes(axis1, axis2), (self,), 'transpose')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.swapaxes(axis1, axis2)
        out._backward = _backward
        return out

    # --- NEGATION: Returns negative of tensor ---
    def __neg__(self): 
        return self * -1

    # --- SUBTRACTION: Element-wise subtract ---
    def __sub__(self, other):
        return self + (-other)

    # --- DIVISION: Element-wise divide ---
    def __truediv__(self, other):
        return self * (other**-1)

    # --- BACKWARD: Runs backpropagation through computational graph ---
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = np.ones_like(self.data)
        
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op})"
