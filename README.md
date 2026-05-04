# AxiomAI

AxiomAI is a local, CPU-first AI training playground built around two connected systems:

- A small autoregressive text model for pretraining, SFT, and chat.
- A small text-conditioned video generation pipeline built from a Video VAE, a text encoder, and a video autoregressive transformer.

This project is designed for personal experimentation on PC and high-end Android/Pydroid-style environments. It is not pretending to be a giant production model stack. The goal is to make the full training loop understandable, editable, and runnable locally.

The most important rule in this project is simple:

```text
Bad data makes bad models.
Bad VAE reconstructions make bad video generation.
Bad captions make text-conditioned video impossible.
```

## Project Map

```text
AxiomAI/
  main.py                         Main console menu
  trainer.py                      Base text pretraining and SFT training loop
  chat.py                         Chat interface and /video command
  preprocess.py                   Pretrain data parser
  sft_preprocess.py               SFT data parser
  model.py                        Text transformer model
  overfit_monitor.py              Train/validation overfit warnings

  configs/
    config.ini                    Text model config
    sft_config.ini                SFT config, if present
    video_config.ini              Video VAE/text/video model config

  data/
    pretrain/                     Raw text pretraining files
    sft/                          SFT JSONL files
    Videos/                       Video .mp4 + .txt caption pairs
    processed/                    Tokenized/cached arrays and video latent cache

  model/
    best_model.pth                Best base text model
    sft_best_model.pth            Best SFT text model
    video_model/
      vae_checkpoint.pth          Best Video VAE checkpoint
      text_encoder_checkpoint.pth Video text encoder checkpoint
      video_checkpoint.pth        Video AR checkpoint
      previews/                   VAE reconstruction preview strips

  video_model/                    Video neural network modules
  video_generation/               Dataset loader and video generation code

  DATAGUIDE.md                    Text pretrain/SFT dataset guide
  VIDEO_GUIDE.md                  Video dataset/training/generation guide
```

## Quick Start

From the project root:

```bash
python main.py
```

The dashboard menu is the main workflow:

```text
1  Train Shared Tokenizer
2  Parse PRETRAIN Data
3  Parse FINE-TUNE Data
4  Train BASE Text Model
5  Train SFT Text Model
6  Chat with Model
7  Train Video VAE
8  Pre-Train Text Encoder
9  Train Video Model
10 Export Models to GGUF
```

Recommended text order:

```text
1. Add raw text to data/pretrain/
2. Train tokenizer
3. Parse pretrain data
4. Train base text model
5. Add SFT JSONL to data/sft/
6. Parse SFT data
7. Train SFT model
8. Chat
```

Recommended video order:

```text
1. Add .mp4 + .txt caption pairs to data/Videos/
2. Train Video VAE
3. Check VAE preview images and PSNR
4. Pre-train Video Text Encoder
5. Train Video Model
6. Use chat command: /video your prompt here
```

Do not skip the VAE quality check. The video model learns compressed VAE tokens. If the VAE cannot reconstruct the input clips, the video model cannot generate good output.

## Text Pipeline

The text model has two stages.

Pretraining teaches general language structure:

```text
data/pretrain/*.txt
data/pretrain/*.jsonl
```

SFT teaches response behavior:

```text
data/sft/*.jsonl
```

Pretraining is not the same thing as SFT. Pretraining is "learn text." SFT is "learn how to answer."

See `DATAGUIDE.md` for exact examples and data rules.

## Video Pipeline

The video pipeline has three trained pieces.

```text
Video VAE:
  pixels -> discrete video tokens -> reconstructed pixels

Text Encoder:
  caption text -> conditioning embeddings

Video AR Transformer:
  caption embeddings + previous VAE tokens -> next VAE tokens
```

The Video VAE is the foundation. AxiomAI now includes:

- stricter video file validation
- train/validation split for VAE
- PSNR reporting
- codebook usage reporting
- reconstruction preview image strips
- stronger VAE loss with L1, MSE, temporal motion, spatial edge, and VQ terms
- EMA codebook updates
- VAE quality gate before Video AR training
- latent cache for faster AR training

See `VIDEO_GUIDE.md` before training video. That guide is the source of truth for clip format, captions, VAE quality targets, failure modes, and scaling.

## Configuration

Text config:

```text
configs/config.ini
```

Video config:

```text
configs/video_config.ini
```

Important text settings:

```ini
d_model = 64
n_layers = 3
n_heads = 4
max_seq_len = 256
batch_size = 8
gradient_accumulation_steps = 4
lr = 3e-4
sequence_stride = 0
enable_torch_compile = False
gradient_checkpointing = False
```

Important video settings:

```ini
width = 144
height = 128
fps = 8.0
duration = 2.0
base_channels = 48
latent_channels = 8
res_blocks_per_stage = 2
use_ema_quantizer = True
use_latent_cache = True
enforce_vae_quality_gate = True
```

If you change VAE architecture settings such as `base_channels`, `latent_channels`, `res_blocks_per_stage`, or `codebook_size`, retrain the VAE. Old VAE checkpoints may no longer match the new model shape.

## What Good Looks Like

For the text model:

- Training loss should go down.
- Validation loss should go down or flatten.
- If training loss improves while validation loss worsens, the model is overfitting.
- Tiny datasets produce broken text even when the code is correct.

For the Video VAE:

- Preview strips should visibly resemble the original clips.
- Motion should not smear into static mush.
- PSNR should climb over training.
- Codebook usage should not stay near zero.
- Validation loss should not drift far above training loss.

For Video AR:

- It should only train after the VAE passes the quality gate.
- Latent cache should build once, then reuse.
- Captions must match the visual action.
- Repeated caption concepts need repeated clips.

## Hard Limits

AxiomAI can train real models, but scale matters.

On CPU/mobile-class hardware:

- Small text models can train.
- Tiny video models can train.
- Quality depends heavily on clean data and enough examples.
- High-resolution video generation is not realistic in this setup.
- The current video generator is not diffusion. It generates VAE token sequences autoregressively.

That does not make it fake. It means the project is closer to a small research trainer than a commercial video model.

## Common Failures

```text
Chat outputs nonsense:
  Not enough pretraining/SFT data, weak tokenizer, or too-small model.

SFT loss looks worse than pretrain:
  Different dataset distribution. SFT validation is not directly comparable to pretrain validation.

Video AR refuses to train:
  Missing VAE checkpoint or VAE quality gate failed.

VAE preview looks bad:
  Train longer, improve video data, lower resolution, or increase VAE capacity carefully.

Video prompt does not control output:
  Captions are too weak/inconsistent, text encoder is weak, or video AR needs more paired examples.

Latent cache seems stale:
  Set rebuild_latent_cache = True for one run, then set it back to False.
```

## Guides

Read these in order:

```text
README.md       Project overview and workflow
DATAGUIDE.md    Text pretrain and SFT data guide
VIDEO_GUIDE.md  Video data, VAE, AR training, and generation guide
```

The guides are intentionally blunt. They are there to prevent wasted training runs.
