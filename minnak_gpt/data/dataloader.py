import numpy as np


class DataLoader:
    """Manages batching and iteration over training data."""
    
    def __init__(self, data, batch_size, seq_len, shuffle=True):
        self.data = np.array(data)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.n_batches = len(self.data) // (batch_size * seq_len)

    # --- ITER: Prepares data for new epoch ---
    def __iter__(self):
        n = self.batch_size * self.seq_len
        clipped_data = self.data[:self.n_batches * n]
        self.batches = clipped_data.reshape(self.batch_size, -1)
        self.current_pos = 0
        return self

    # --- NEXT: Returns next source-target batch pair ---
    def __next__(self):
        if self.current_pos + self.seq_len >= self.batches.shape[1]:
            raise StopIteration
            
        src = self.batches[:, self.current_pos : self.current_pos + self.seq_len]
        tgt = self.batches[:, self.current_pos + 1 : self.current_pos + self.seq_len + 1]
        
        self.current_pos += self.seq_len
        return src, tgt

    def __len__(self):
        return self.n_batches
