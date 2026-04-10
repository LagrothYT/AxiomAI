import torch

def safe_load(path, map_location=None):
    """Load a checkpoint, preferring weights_only=True for security.
    Falls back to weights_only=False with a warning if the checkpoint
    was saved by an older PyTorch version or contains non-standard objects.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        print(f"[!] Warning: Safe load failed for {path}. Falling back to full unpickle. "
              "Only load checkpoints you trust.")
        return torch.load(path, map_location=map_location, weights_only=False)
