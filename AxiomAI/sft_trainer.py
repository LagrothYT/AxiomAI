"""
Supervised Fine-Tuning is now natively integrated into trainer.py via the `is_sft` toggle.
This file serves strictly as a command-line wrapper for backward compatibility.
"""
import sys
from trainer import train_model

if __name__ == "__main__":
    train_model(is_sft=True)