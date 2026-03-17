import os

# Model Architecture
n_layers = 4
n_heads = 4
d_model = 256
d_ff = 1024
max_seq_len = 512
vocab_size = 5000  # Target BPE vocab size

# Training Hyperparameters
batch_size = 8
lr = 3e-4
epochs = 5
grad_accum_steps = 4
dropout = 0.1
early_stopping_patience = 3

# Data & Checkpoints
data_dir = "data"
processed_data_dir = os.path.join(data_dir, "processed")
checkpoint_dir = "checkpoints"
tokenizer_dir = "tokenizer"
best_model_path = os.path.join(checkpoint_dir, "best.pt")

# Directories to ensure exist
REQUIRED_DIRS = [data_dir, processed_data_dir, checkpoint_dir, tokenizer_dir]

def ensure_dirs():
    for d in REQUIRED_DIRS:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")

if __name__ == "__main__":
    ensure_dirs()
