"""
Minnak-GPT Training Script
"""
import numpy as np

from minnak_gpt.model import Transformer
from minnak_gpt.losses import CrossEntropyLoss
from minnak_gpt.optim import Adam
from minnak_gpt.data import DataLoader


# --- CONFIGURATION ---
VOCAB_SIZE = 100
EMBED_DIM = 128
NUM_HEADS = 8
FF_DIM = 512
NUM_LAYERS = 4
SEQ_LEN = 32
BATCH_SIZE = 16
LR = 0.001
EPOCHS = 10


def main():
    # --- DATA PREPARATION: Random dummy data for demo ---
    data = np.random.randint(0, VOCAB_SIZE, (10000,))
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)

    # --- MODEL SETUP ---
    model = Transformer(
        vocab_size=VOCAB_SIZE, 
        embed_dim=EMBED_DIM, 
        num_heads=NUM_HEADS, 
        ff_dim=FF_DIM, 
        num_layers=NUM_LAYERS, 
        seq_len=SEQ_LEN
    )
    
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    # --- TRAINING LOOP ---
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            optimizer.zero_grad()
            logits = model(src, tgt)
            loss = loss_fn(logits, tgt)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.data
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.data:.4f}")
                
        print(f"Epoch {epoch+1} completed. Avg Loss: {epoch_loss / len(dataloader):.4f}")

    print("Training finished!")


if __name__ == "__main__":
    main()
