"""
Tensor Engine Tests
"""
import numpy as np
from minnak_gpt.engine.tensor import Tensor


def test_tensor_creation():
    """Tests basic tensor creation."""
    t = Tensor([1.0, 2.0, 3.0])
    assert t.data.shape == (3,)
    assert t.requires_grad == True
    print("✓ test_tensor_creation passed")


def test_addition():
    """Tests addition and gradient computation."""
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = a + b
    c.backward()
    
    assert np.allclose(c.data, [5.0, 7.0, 9.0])
    assert np.allclose(a.grad, [1.0, 1.0, 1.0])
    print("✓ test_addition passed")


def test_multiplication():
    """Tests multiplication and gradient computation."""
    a = Tensor([2.0, 3.0])
    b = Tensor([4.0, 5.0])
    c = a * b
    c = c.sum()
    c.backward()
    
    assert np.allclose(a.grad, [4.0, 5.0])
    assert np.allclose(b.grad, [2.0, 3.0])
    print("✓ test_multiplication passed")


def test_matmul():
    """Tests matrix multiplication."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    c = a.matmul(b)
    
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    assert np.allclose(c.data, expected)
    print("✓ test_matmul passed")


def test_backward():
    """Tests backpropagation chain."""
    x = Tensor([2.0])
    y = x * x  # y = x^2
    z = y * y  # z = x^4
    z.backward()
    
    # dz/dx = 4x^3 = 4 * 8 = 32
    assert np.allclose(x.grad, [32.0])
    print("✓ test_backward passed")


if __name__ == "__main__":
    test_tensor_creation()
    test_addition()
    test_multiplication()
    test_matmul()
    test_backward()
    print("\nAll tests passed!")
