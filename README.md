# AxiomAI (TinyGPT)

A fully from-scratch AI studio built on PyTorch. Train a text chatbot, generate images from prompts, and generate short video clips — all from a single interactive menu.

## Features

- **Custom BPE Tokenizer** — Built from scratch for efficient text encoding.
- **Transformer (Text)** — Causal multi-head attention with RoPE and KV-cache, optimised for CPU.
- **Dual Training Modes** — Pretraining and Supervised Fine-Tuning (SFT).
- **Image Generation** — Latent Diffusion with a VAE encoder, DDPM/DDIM scheduler, and CFG.
- **Video Generation** — Spatiotemporal Latent Diffusion using TemporalAttention layers inserted into the image UNet. Shares the image VAE and tokenizer — no re-training of those required.
- **Hardware** — Default: CPU + 16 GB RAM. GPU: CUDA is auto-detected. ROCm: `export USE_ROCM=1`.

---

## Quick Start

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Launch the control center:**
```bash
python main.py
```

---

## CLI Menu

| Option | Action | Description |
| :--- | :--- | :--- |
| 1 | Train Tokenizer | Trains the BPE model on `data/*.jsonl` |
| 2 | Pretrain Preprocessing | Prepares raw text for causal modelling |
| 3 | Pretrain Model | Starts the main pretraining loop |
| 4 | SFT Preprocessing | Prepares conversation data for instruction tuning |
| 5 | SFT Fine-tune | Runs the Supervised Fine-Tuning phase |
| 6 | Train Image VAE | Trains the VAE encoder/decoder on your photo dataset |
| 7 | Train Image Diffusion | Trains the latent diffusion UNet on your photo dataset |
| 9 | Train Video Diffusion | Trains the video diffusion UNet on your video clip dataset |
| 10 | Launch Chat | Interactive chat with `/imagine` and `/animate` commands |
| 11 | Exit | Close the studio |

Prerequisite chains are enforced automatically — the menu will tell you exactly what is missing before it lets you run a step.

---

## Chat Commands

Once in chat (option 10):

| Command | Action |
| :--- | :--- |
| `/imagine <prompt>` | Generate a single image from a text prompt |
| `/animate <prompt>` | Generate a short video clip from a text prompt |
| `/clear` | Clear conversation history |
| `/help` | Show all commands and current settings |
| `/quit` | Exit the chat |

---

## Pipeline Overview

```
Text Pipeline
─────────────
data/pretrain/*.jsonl  →  [1] Tokenizer  →  [2] Preprocess  →  [3] Pretrain  →  [4/5] SFT  →  chat

Image Pipeline
──────────────
Photos/  →  [6] VAE (The Eyes)  →  [7] Diffusion (The Brain)  →  /imagine in chat

Video Pipeline
──────────────
Videos/                      →  [9] Video Diffusion  ──┐
image_gen_model/vae_best.pt  ──────── (shared frozen) ─┘  →  /animate in chat
```

The video pipeline reuses the image VAE and your BPE tokenizer. You do not retrain them.
Option 9 trains only the temporal attention layers inside the video UNet when
`freeze_spatial = True` in `video_config.ini` (the default). This means the model
learns inter-frame coherence while the spatial image knowledge is preserved.

---

## Project Structure

```
tiny_gpt/
├── main.py                          Central control menu
├── train.py                         Text model training (pretrain + SFT)
├── chat.py                          Interactive chat + /imagine + /animate
├── config.py                        Text model config loader
├── utils.py                         Shared utilities
│
├── config/
│   ├── config.ini                   Text model settings
│   ├── image_config.ini             Image generation settings
│   └── video_config.ini             Video generation settings
│
├── model/
│   └── transformer.py               TinyGPT transformer architecture
│
├── tokenizer/
│   ├── bpe.py                       BPE tokenizer implementation
│   └── train_tokenizer.py           Tokenizer training script
│
├── data/
│   ├── pretrain/                    Raw pretrain .jsonl files
│   ├── sft/                         SFT .jsonl files
│   ├── processed/                   Compiled .npy arrays
│   └── prepare_data.py              Data preprocessing script
│
├── image_generation/
│   ├── config.py                    ImageModelConfig
│   ├── scheduler.py                 DDPM + DDIM noise scheduler
│   ├── image_loader.py              ImageDataset (single-frame)
│   ├── bridge.py                    ModelBridge (/imagine routing)
│   ├── train.py                     VAE + Diffusion training
│   └── model/
│       ├── vae.py                   Variational Autoencoder
│       ├── diffusion.py             LatentDiffusion UNet
│       └── text_encoder.py          Transformer text encoder
│
├── video_generation/
│   ├── config.py                    VideoModelConfig
│   ├── video_loader.py              VideoDataset (frame sequences)
│   ├── bridge.py                    VideoBridge (/animate routing)
│   ├── train.py                     Video diffusion training
│   └── model/
│       ├── temporal_attention.py    TemporalAttention (T-dim self-attn)
│       └── video_unet.py            VideoUNetBlock + VideoLatentDiffusion
│
├── Photos/                          Image training data
├── Videos/                          Video training data (clips)
├── image_gen_model/                 Saved image model weights (.pt)
├── video_gen_model/                 Saved video model weights (.pt)
├── out_image/                       Generated images
└── out_video/                       Generated video clips (GIF + MP4)
```

---

## Hardware Notes

| Scenario | Recommendation |
| :--- | :--- |
| CPU only | `num_frames = 8`, `batch_size = 1`, resolution ≤ 144×96 |
| Consumer GPU 8 GB | `num_frames = 8–16`, `batch_size = 2–4`, resolution up to 256×256 |
| `freeze_spatial = True` | Strongly recommended. Trains ~2–5% of video UNet params. Much less data needed. |
| `freeze_spatial = False` | Full training from scratch. Needs a large clip dataset and a GPU. |

---

## ROCm / CUDA

```bash
# ROCm
export USE_ROCM=1
python main.py

# CUDA is detected automatically — no flag needed.
```
