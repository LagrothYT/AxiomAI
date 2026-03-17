# TinyGPT

A from-scratch GPT-style language model optimized for CPU-only systems (like the Ryzen 7 7840HS).

## Features
- Custom BPE Tokenizer from scratch.
- Causal Transformer architecture with multi-head attention.
- Support for Pretraining and SFT (Supervised Fine-Tuning).
- ShareGPT data format support.
- Low-memory footprint for 16GB RAM systems.

## Unified CLI
You can manage the entire pipeline through `main.py`:

```bash
# 1. Train Tokenizer
python main.py train-tokenizer

# 2. Pretrain Phase
python main.py prepare --mode pretrain
python main.py train --mode pretrain

# 3. SFT Phase
python main.py prepare --mode sft
python main.py train --mode sft

# 4. Chat
python main.py chat
```

## Individual Script Usage (Advanced)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Drop your `.jsonl` files (ShareGPT format) into the `data/` directory.

### 3. Train Tokenizer
```bash
python tokenizer/train_tokenizer.py
```

### 4. Preprocess
```bash
python data/prepare_data.py --mode pretrain
# or
python data/prepare_data.py --mode sft
```

### 5. Train
```bash
python train.py --mode pretrain
# or
python train.py --mode sft
```

### 6. Chat
```bash
python chat.py
```
Options:
- `--temperature`: Control randomness (default: 0.8)
- `--top_p`: Nucleus sampling (default: 0.9)
- `--json_output`: Output JSON instead of raw text


## Hardware Support
- Uses PyTorch CPU by default.
- Support for ROCm: `export USE_ROCM=1` if available.
