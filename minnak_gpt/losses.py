import numpy as np
from minnak_gpt.engine.tensor import Tensor


class Loss:
    """Base class for loss functions."""
    
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

    def forward(self, y_pred, y_true):
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    """Cross-entropy loss for language modeling."""
    
    # --- FORWARD: Computes NLL loss from logits and target indices ---
    def forward(self, logits, target):
        batch_size, seq_len, vocab_size = logits.data.shape
        logits_flat = logits.data.reshape(-1, vocab_size)
        target_flat = target.reshape(-1)
        
        logits_max = np.max(logits_flat, axis=1, keepdims=True)
        exp_logits = np.exp(logits_flat - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        nll = -np.log(probs[np.arange(len(target_flat)), target_flat] + 1e-9)
        loss_val = np.mean(nll)
        
        out = Tensor(loss_val, (logits,), 'cross_entropy')

        def _backward():
            d_logits = probs.copy()
            d_logits[np.arange(len(target_flat)), target_flat] -= 1
            d_logits /= len(target_flat)
            logits.grad += d_logits.reshape(batch_size, seq_len, vocab_size) * out.grad
            
        out._backward = _backward
        return out
