# AxiomAI Data Guide

This guide explains the text data side of AxiomAI: tokenizer training, base pretraining, and supervised fine-tuning.

Video data is different. For video clips, captions, VAE quality, and text-conditioned video generation, read `VIDEO_GUIDE.md`.

## The Big Difference

AxiomAI has two text training stages:

```text
Pretraining:
  Teaches the model language patterns.
  Input is mostly raw text.

SFT:
  Teaches the model how to answer like an assistant.
  Input is structured conversation JSONL.
```

Do not mix up their purpose.

Pretraining is not "chat behavior." It is the model learning grammar, words, facts, style, and continuation.

SFT is not "general intelligence." It is the model learning that a human message should be followed by an assistant response.

## Folder Layout

Recommended layout:

```text
data/
  pretrain/
    alice.txt
    aesop_fables.txt
    grimms_fairy_tales.txt
    notes.txt

  sft/
    assistant_examples.jsonl
    coding_examples.jsonl
    personality_examples.jsonl

  processed/
    pretrain_train.npy
    pretrain_val.npy
    sft_train.npy
    sft_val.npy
```

The exact paths are controlled in:

```text
configs/config.ini
configs/sft_config.ini
```

## Tokenizer Training

Run menu option:

```text
1. Train Shared Tokenizer
```

The tokenizer should be trained before parsing pretrain or SFT data.

The tokenizer learns the vocabulary used by the model. If you train it on too little text, the model sees awkward token splits and learning gets harder.

Good tokenizer source:

```text
Raw books
Notes
Instruction examples
Conversation examples
Domain text you actually want the model to understand
```

Bad tokenizer source:

```text
One tiny file
Only SFT examples
Only weird symbols
Mostly corrupted text
```

If you completely change your dataset style, retraining the tokenizer can help. If you retrain the tokenizer, you should usually reprocess data and retrain models that depend on it.

## Pretraining Data

Run menu option:

```text
2. Parse PRETRAIN Data
4. Train BASE Text Model
```

Pretraining can use `.txt` and supported `.jsonl` content.

Best simple format:

```text
data/pretrain/my_book.txt
```

The file can be plain raw text:

```text
Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do. Once or twice she had peeped into the book her
sister was reading, but it had no pictures or conversations in it.
```

That is valid pretraining data. No labels. No prompt. No assistant tags. Just text.

### Good Pretraining Data

Use clean, readable text:

```text
Books
Articles
Manuals
Public-domain literature
Your own notes
Small encyclopedic text
Carefully cleaned scraped text
```

For a tiny model, variety helps, but chaos hurts. A model trained only on old fairy tales will sound like old fairy tales. A model trained on mixed clean modern text will learn more useful language.

### Bad Pretraining Data

Avoid:

```text
HTML menus
Navigation bars
Repeated copyright footers
Broken OCR
Duplicate pages
Random Unicode garbage
Extremely long tables
Unlabeled chat logs
Huge files with no cleanup
```

The model does not know which parts are junk. It learns all of it.

## SFT Data

Run menu option:

```text
3. Parse FINE-TUNE Data
5. Train SFT Text Model
```

SFT data should be JSONL. JSONL means one complete JSON object per line.

Each line should look like this:

```jsonl
{"conversations":[{"from":"system","value":"You are Axiom, a helpful local AI assistant."},{"from":"human","value":"What is pretraining?"},{"from":"gpt","value":"Pretraining teaches a model general language patterns by predicting text from large amounts of raw data."}]}
{"conversations":[{"from":"human","value":"Give me a short Python loop."},{"from":"gpt","value":"```python\nfor i in range(5):\n    print(i)\n```"}]}
```

The supported roles are:

```text
system
human
gpt
```

Use `human` for the user and `gpt` for the assistant.

### Good SFT Examples

Good SFT examples are clear and direct:

```jsonl
{"conversations":[{"from":"human","value":"Explain validation loss in simple terms."},{"from":"gpt","value":"Validation loss measures how well the model performs on data it did not train on. If training loss improves but validation loss gets worse, the model is probably memorizing."}]}
```

Good SFT teaches:

```text
How to answer questions
How long answers should be
How to format code
How to refuse impossible requests
How to follow your preferred tone
How to use project-specific words
```

### Bad SFT Examples

Avoid examples where:

```text
The assistant answer is wrong.
The assistant answer is empty.
The user message is missing.
The assistant rambles for no reason.
The JSON is invalid.
Different examples use wildly different styles.
The model is trained to hallucinate certainty.
```

The model copies behavior. If the SFT examples are messy, the model becomes messy.

## Pretrain JSONL vs SFT JSONL

Pretraining JSONL is used as text to learn language.

SFT JSONL is used as role-structured assistant behavior.

If you are unsure where something belongs:

```text
Is it a book/article/reference/plain text?
  Put it in pretraining.

Is it a user question plus ideal assistant answer?
  Put it in SFT.

Is it a random chat transcript with no cleanup?
  Clean it first or do not use it.
```

## Minimum Useful Dataset Sizes

These are rough practical targets, not laws.

```text
Tokenizer:
  At least a few MB of mixed text is better than one tiny file.

Pretraining:
  Under 1 MB: pipeline test only.
  1-10 MB: model starts learning visible language patterns.
  10-100 MB: much more useful for small local models.
  100 MB+: better, but train time rises fast on CPU.

SFT:
  50 examples: tiny behavior test.
  500 examples: starts shaping response style.
  5,000 examples: much better.
  50,000 examples: serious small-model SFT territory.
```

For your current tiny model, data quality matters more than raw size. A smaller clean dataset usually beats a larger polluted one.

## Validation Split

Validation data is held out from training. The model does not update weights on it.

Use validation to answer:

```text
Is the model actually learning?
Is it memorizing?
Did the latest config make generalization worse?
```

Healthy training:

```text
Training loss down
Validation loss down or stable
Gap between them not exploding
```

Overfitting:

```text
Training loss down
Validation loss up
Generated text becomes narrow/repetitive
```

## Perplexity

Perplexity, or PPL, is a rough measure of how surprised the model is by the text.

Lower is better, but PPL is only comparable on similar data.

Do not directly compare:

```text
Pretrain PPL on books
SFT PPL on assistant JSONL
Chat prompt PPL during generation
```

They are different distributions.

Useful rough interpretation:

```text
PPL 1000+:
  Very confused.

PPL 200-800:
  Learning basic local patterns, still weak.

PPL 50-200:
  Much more coherent for small models.

PPL under 50:
  Stronger fit for this dataset, but watch for overfitting.
```

Tiny models can still generate weird text even with better PPL. Good data, enough training, and SFT all matter.

## Cleaning Checklist

Before parsing data:

```text
Remove duplicate pages.
Remove HTML.
Remove menus/navigation.
Remove broken encoding.
Remove empty files.
Remove repeated boilerplate.
Keep paragraphs readable.
Keep SFT answers correct.
Validate JSONL.
```

For JSONL, every line must be valid JSON. One broken line can kill preprocessing.

## Example SFT File

Save as:

```text
data/sft/basic_assistant.jsonl
```

Content:

```jsonl
{"conversations":[{"from":"system","value":"You are Axiom, a concise and helpful assistant."},{"from":"human","value":"What is the difference between pretraining and SFT?"},{"from":"gpt","value":"Pretraining teaches the model general language patterns from raw text. SFT teaches it how to respond to user instructions using curated conversation examples."}]}
{"conversations":[{"from":"human","value":"Why is validation loss important?"},{"from":"gpt","value":"Validation loss shows whether the model is improving on held-out data. If validation loss rises while training loss falls, the model is likely overfitting."}]}
{"conversations":[{"from":"human","value":"Give me a JSONL example."},{"from":"gpt","value":"```jsonl\n{\"conversations\":[{\"from\":\"human\",\"value\":\"Hello\"},{\"from\":\"gpt\",\"value\":\"Hi. How can I help?\"}]}\n```"}]}
```

## Troubleshooting

```text
Tokenizer fails:
  Check raw_data_path and make sure files exist.

Preprocess finds no data:
  Check configs/config.ini paths.

SFT parser fails:
  Validate JSONL. Every line must be a full JSON object.

Model talks like old books:
  You pretrained mostly on old books. Add modern text and better SFT.

Model cannot answer:
  It needs more pretraining, better SFT, or a larger model.

SFT seems worse than base:
  The SFT dataset may be too tiny, too different, or badly formatted.

Training improves but chat is still weird:
  Loss is not the whole story. Data quality and generation sampling matter.
```

## Final Rule

Treat data like source code. If the data is sloppy, the model is sloppy. Clean input is not optional.
