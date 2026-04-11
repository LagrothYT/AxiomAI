# TinyGPT Data Formatting Guide

This guide explicitly details how you must format your training data for all modules of the TinyGPT architecture. The neural networks depend on these exact structures to learn correctly without mathematically corrupting.

---

## 1. Text Pretraining (`data/pretrain/`)

Pretraining is the phase where you teach the AI grammar, math, facts, and raw knowledge from scratch. It is not a chatbot yet; it is simply a raw text predictor.

**Where to put it:**
Place all your files inside `data/pretrain/`. They must end in `.jsonl` (JSON Lines).

**How it must look:**
Every line in the file must be a valid JSON object. For pretraining, we strictly extract only the `value` fields. You do not need roles like "human" or "gpt", just raw text blocks.

```json
{"conversations": [{"value": "The mitochondria is the powerhouse of the cell. It produces energy..."}]}
{"conversations": [{"value": "def hello_world():\n    print('Hello World')"}]}
```

**Best Practices:**
- Feed it large blocks of clean text (Wikipedia articles, books, code repositories).
- There is no conversational formatting here. Just raw knowledge. Do not try to teach it formatting during pretraining.

---

## 2. Supervised Fine-Tuning (SFT) (`data/sft/`)

SFT is where you take your smart, pre-trained brain and teach it discipline. You teach it how to behave as a chatbot, how to obey constraints, and what a "conversation" looks like.

**Where to put it:**
Place all `.jsonl` files inside `data/sft/`.

**How it must look:**
The data compiler reads the `"from"` tags to figure out exactly how to map penalties and rewards to the neural network.

```json
{"conversations": [{"from": "system", "value": "You are a sarcastic AI."}, {"from": "human", "value": "What is 2+2?"}, {"from": "gpt", "value": "It is four. Obviously."}]}
```

### The 3 Core Roles:
1. **`"system"`**: Optional. Tells the model its overarching master-prompt or persona. The neural network's loss logic is hardcoded to *obey* this, but it will never be punished for not memorizing it word-for-word.
2. **`"human"`**: Mandatory. The prompt. The network applies a `0` loss mask here, meaning it knows it shouldn't try to auto-generate the human's queries.
3. **`"gpt"`**: Mandatory. The answer. The network applies a `1` loss mask here. This is the **only** text the AI is actively trained and forced to predict.

**Best Practices:**
- Always ensure "gpt" is the one doing the heavy lifting.
- Multi-turn conversations are allowed! Just chain them: `system -> human -> gpt -> human -> gpt`.

---

## 3. Image Generation Pipeline (`Photos/` or `image_generation/data/`)

To train the VAE (The Eyes) and Diffusion (The Brain) models, you need a collection of images paired with descriptive text (captions). 

**Where to put it:**
Place your images inside the data directory defined in your `image_config.ini` (typically `image_generation/data` or `Photos/`).

**How to format it:**
There are no JSON files here. The `image_loader.py` engine scans the raw directory for images (PNG, JPG, JPEG) and smartly grabs captions based on the filesystem. You have two options:

### Option A: The Text-Pairing Method (Highest Quality)
Place a `.txt` file with the **exact same name** directly next to the image. The loader will extract the text inside.
```text
Photos/
  |- a_cool_dog.jpg
  |- a_cool_dog.txt     <-- Contains: "A golden retriever wearing sunglasses on a beach."
  |- red_car.png
  |- red_car.txt        <-- Contains: "A shiny red sports car driving down a neon highway."
```

### Option B: The Directory/Filename Fallback (Fastest)
If no `.txt` file exists, the loader uses the filename or directory name as the caption fallback.
```text
Photos/
  |- retro_city_skyline.jpg       <-- Caption becomes: "retro_city_skyline"
  |- cyberpunk_cars/
      |- car1.jpg                 <-- Caption becomes: "cyberpunk_cars"
      |- car2.jpg                 <-- Caption becomes: "cyberpunk_cars"
```

**Best Practices for Images:**
- Keep descriptions extremely precise and visually descriptive.
- Ensure all images are clean. Corrupted images will be automatically skipped by the loader to prevent crashing the PyTorch queue.
- Aspect ratio matters. The custom `image_config.ini` enforces a specific width/height scale (which will squash or stretch your images if they are drastically mismatched).
