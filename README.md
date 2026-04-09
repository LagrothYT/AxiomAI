# TinyGPT - Your Own Tiny AI Chatbot + Image Generator

A complete from-scratch GPT-style language model plus a small image generator that runs on a normal laptop.

TinyGPT lets you train a real AI on your own chat data, talk to it like a friend, and even generate images by typing `/imagine` in the same chat.

It is small, educational, fun, and 100 percent yours. No cloud. No giant downloads.

---

## What You Can Do

- Chat with a friendly, personality-filled AI
- Type `/imagine a red phone on a wooden table` and it generates a real image
- Train everything yourself (text model + image model)
- Run on CPU or AMD GPU with ROCm

---

## Quick Start (5 minutes)

1. Open the `AxiomAI/tiny_gpt` folder
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the menu:
   ```bash
   python main.py
   ```

That is it. Use the numbered menu to train and chat.

---

## Project Structure

```
tiny_gpt/
  data/                  conversations (data.jsonl)
  Photos/                images + captions for image training
  model/                 trained models are saved here
  tokenizer/             tokenizer files
  out_image/             generated images appear here
  main.py                main menu
  chat.py                the chat interface
  config/                text model settings
  image_config.json      image generation settings
  image_generation/      image VAE + diffusion code
```

---

## The Chatbot

### Data Format (`data/data.jsonl`)

Each line is one conversation. Example:

```json
{"conversations": [
  {"from": "human", "value": "it's so hot today I genuinely cannot think straight"},
  {"from": "gpt", "value": "Heat does that. Your brain prioritizes cooling over everything else..."}
]}
```

Add as many natural conversations as you want. More data gives the model better personality.

### How to Chat

1. Train the model using Options 1–5 in the main menu
2. Choose **Option 8: Chat**
3. Type normally
4. Special commands:
   ```
   /imagine your prompt here   -> generates an image
   /clear                      -> clears chat history
   /quit                       -> exit
   ```

The chat shows live streaming, tokens per second, perplexity, and remaining context.

---

## Image Generation

TinyGPT can create images directly from the chat.

### How to make good images

**1. Add photos**

Put them in `Photos/` (you can use subfolders like `Photos/phones/`)  
Aim for 150–200+ images for decent results.

**2. Add caption files**

For every image `image.jpg` you MUST create a file named `image.txt` next to it.

Example content of `image.txt`:
```
A modern black smartphone lying on a wooden desk, natural lighting, close up
```

Keep captions short and descriptive.

**3. Train the image models (in this exact order)**

```
Option 6 -> Train Image VAE (The Eyes)
Option 7 -> Train Image Diffusion (The Brain)
```

**4. Generate images**

In the chat just type:
```
/imagine a red phone floating in space with stars around it
```

The image is saved as `out_image/latest_generation.jpg`

---

## Training Order (Important!)

Always follow this sequence in the main menu:

1. Train Tokenizer
2. Preprocess Data (Pretrain)
3. Pretrain Model
4. Preprocess Data (SFT)
5. SFT Fine-tune
6. Train Image VAE
7. Train Image Diffusion
8. Chat

---

## Configuration Tips

Edit `image_config.json` after adding more images:

```json
image_width: 256
image_height: 192
epochs: 300
batch_size: 4
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Tokenizer not found | Run Option 1 |
| Model checkpoint not found | Run Options 3 and 5 |
| Images look like random blobs | Need 150–200+ images then retrain VAE and Diffusion |
| Out of memory during image training | Lower `batch_size` to 2 or 4 in `image_config.json` |
| Generation is slow | Normal on CPU, just wait |

---

## Best Practices

- For the chatbot: more casual human-like conversations = more personality
- For images: variety (different angles, lighting, backgrounds) is more important than quantity
- Always train VAE before Diffusion
- You can run everything on a normal laptop

---

## Final Note

TinyGPT is your little AI. You trained it, you talk to it, and you can keep improving it forever.

Have fun, add your own data, make it weird, and enjoy watching it grow.

**Now go train it and say hi to your creation!**
