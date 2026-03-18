# TinyGPT

A from-scratch GPT-style language model optimized for CPU systems.

## Features
- **Custom BPE Tokenizer**: Built from scratch for efficient text encoding.
- **Transformer Architecture**: Causal multi-head attention with weight tying.
- **Dual Mode Support**: Infrastructure for both Pretraining and Supervised Fine-Tuning (SFT).
- **Optimization**: Tailored for 16GB RAM systems and CPU-only inference.

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Interactive CLI**:
   The entire pipeline is managed via a single entry point:
   ```bash
   python main.py
   ```

## CLI Menu Options

| Option | Action | Description |
| :--- | :--- | :--- |
| 1 | Train Tokenizer | Trains the BPE model on `data/*.jsonl` |
| 2 | Pretrain Preprocessing | Prepares raw text for causal modeling |
| 3 | Pretrain Model | Starts the main pretraining loop |
| 4 | SFT Preprocessing | Prepares conversation data for instruction tuning |
| 5 | SFT Fine-tune | Runs the Supervised Fine-Tuning phase |
| 6 | Interactive Chat | Launch chat with custom temperature/top-p |
| 7 | Exit | Close the tool |

## Project Structure

- `main.py`: Interactive central controller.
- `train.py`: Core training logic for both phases.
- `chat.py`: Inference and conversation interface.
- `config.py`: Configuration management and defaults.
- `model/`: Transformer architecture implementation.
- `tokenizer/`: BPE tokenizer logic and training.
- `data/`: Data loading and preprocessing scripts.

## Hardware Support
- **Default**: PyTorch CPU.
- **ROCm**: Enable with `export USE_ROCM=1`.
