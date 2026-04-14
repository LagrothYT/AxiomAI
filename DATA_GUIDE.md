# AxiomAI Data Formatting Guide

This guide details exactly how to format training data for every pipeline in AxiomAI.
The neural networks depend on these structures to learn correctly.

---

## 1. Text Pretraining (`data/pretrain/`)

Pretraining teaches the AI grammar, facts, and raw knowledge from scratch. It is not a
chatbot yet — it is a raw text predictor.

**Where to put it:**
Place all files inside `data/pretrain/`. They must end in `.jsonl` (JSON Lines).

**Format:**
Every line must be a valid JSON object. Only the `value` fields are extracted.

```json
{"conversations": [{"value": "The mitochondria is the powerhouse of the cell."}]}
{"conversations": [{"value": "def hello_world():\n    print('Hello World')"}]}
```

**Best practices:**
- Feed it large blocks of clean text (Wikipedia articles, books, code repositories).
- No conversational formatting here. Just raw knowledge.

---

## 2. Supervised Fine-Tuning (`data/sft/`)

SFT teaches the pretrained brain discipline — how to behave as a chatbot, follow
instructions, and hold conversations.

**Where to put it:**
Place all `.jsonl` files inside `data/sft/`.

**Format:**
The `"from"` tag tells the training loop how to apply the loss mask.

```json
{"conversations": [{"from": "system", "value": "You are a helpful assistant."}, {"from": "human", "value": "What is 2+2?"}, {"from": "gpt", "value": "It is 4."}]}
```

**The 3 core roles:**

1. **`"system"`** — Optional. Sets the model's persona or master-prompt. Loss is not applied here.
2. **`"human"`** — Mandatory. The user's message. Loss mask = `0` (model is not trained to predict this).
3. **`"gpt"`** — Mandatory. The model's response. Loss mask = `1`. This is the only text the model is actively trained to predict.

**Best practices:**
- Multi-turn conversations are allowed: `system → human → gpt → human → gpt`.
- Always ensure `"gpt"` is doing the heavy lifting — it is the only supervised signal.

---

## 3. Image Generation (`Photos/`)

Trains the VAE (The Eyes) and Latent Diffusion (The Brain) models on a photo dataset.

**Where to put it:**
Place images inside the directory set in `config/image_config.ini` → `data_dir`
(default: `Photos/`).

**No JSON files here.** The `image_loader.py` engine scans for images and resolves captions
from the filesystem.

### Option A — Text-pairing method (highest quality)
Place a `.txt` file with the exact same filename next to the image.

```
Photos/
  ├── golden_retriever.jpg
  ├── golden_retriever.txt    ← "A golden retriever wearing sunglasses on a beach."
  ├── red_car.png
  └── red_car.txt             ← "A shiny red sports car on a neon highway."
```

### Option B — Directory/filename fallback (fastest)
If no `.txt` file exists, the filename or parent directory name becomes the caption.

```
Photos/
  ├── retro_city_skyline.jpg       ← Caption: "retro_city_skyline"
  └── cyberpunk_cars/
      ├── car1.jpg                 ← Caption: "cyberpunk_cars"
      └── car2.jpg                 ← Caption: "cyberpunk_cars"
```

**Best practices:**
- Keep descriptions visually precise.
- Corrupted images are automatically skipped.
- Resolution is controlled by `image_config.ini` → `image_width` / `image_height`. Images are
  resized to fit. Both dimensions must be divisible by `vae_downsample_factor` (default 4).

---

## 4. Video Generation (`Videos/`)

Trains the VideoLatentDiffusion model on a dataset of video clips.

**Where to put it:**
Place clips inside the directory set in `config/video_config.ini` → `data_dir`
(default: `Videos/`).

Two clip formats are supported.

---

### Format A — Frame folders (recommended, highest compatibility)

Each clip is a subdirectory containing sequential image files. Files are sorted
lexicographically, so zero-pad your filenames for correct ordering.

```
Videos/
  ├── sunset_beach/
  │   ├── frame_0001.jpg
  │   ├── frame_0002.jpg
  │   ├── frame_0003.jpg
  │   └── ...
  ├── sunset_beach.txt          ← Caption: "A slow sunset over a calm beach."
  │
  ├── running_dog/
  │   ├── 0001.png
  │   ├── 0002.png
  │   └── ...
  └── running_dog.txt           ← Caption: "A dog running through a field of grass."
```

Caption resolution priority for frame folders:
1. Sibling `.txt` file: `Videos/clip_name.txt` — highest priority, use this.
2. Inner `caption.txt`: `Videos/clip_name/caption.txt`
3. Folder name as fallback: `"clip_name"`

---

### Format B — Video files (mp4, avi, mov, mkv, gif)

Place video files directly inside the `Videos/` directory. Frames are extracted
at load time using `torchvision.io` (preferred) or `opencv-python` (fallback).

```
Videos/
  ├── sunset_beach.mp4
  ├── sunset_beach.txt          ← Caption: "A slow sunset over a calm beach."
  ├── running_dog.avi
  └── running_dog.txt           ← Caption: "A dog running through a field of grass."
```

Caption resolution for video files:
1. Sibling `.txt` file: `Videos/filename.txt` — use this.
2. Filename stem as fallback: `"filename"`

> **Note:** Format A (frame folders) is more reliable across all platforms and does not
> require `opencv-python`. Use Format B only if you already have raw video files.

---

### Frame sampling

During training, `num_frames` frames are sampled from each clip per batch:
- **Training:** a random contiguous window is selected (temporal augmentation).
- **Validation:** frames are evenly spaced (deterministic).
- **Short clips** (fewer frames than `num_frames`): the last frame is repeated to reach `num_frames`.

The `num_frames` setting is in `config/video_config.ini`. Start with `8` on CPU.

---

### Minimum dataset size

For `freeze_spatial = True` (default): 50–200 clips is enough to teach temporal coherence.
For `freeze_spatial = False` (full training): thousands of clips are needed.

---

### Resolution

`frame_width` and `frame_height` in `video_config.ini` must match `image_width` and
`image_height` in `image_config.ini`. The shared VAE was trained on one resolution —
using a different resolution for video will produce incoherent latents.

Both dimensions must be divisible by `vae_downsample_factor` (default 4).

---

## 5. Output locations

| Pipeline | Generated output |
| :--- | :--- |
| Image (`/imagine`) | `out_image/latest_generation.jpg` |
| Video (`/animate`) | `out_video/latest_generation.gif` and `out_video/latest_generation.mp4` (if imageio is installed) |
