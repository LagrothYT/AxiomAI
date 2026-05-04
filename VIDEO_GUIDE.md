# AxiomAI Video Guide

This is the practical guide for AxiomAI video generation.

Read this before training video. The video pipeline is less forgiving than text. If the clips, captions, VAE, or config are weak, the final generation will be weak.

## The Video Pipeline

AxiomAI video generation is built from three pieces:

```text
1. Video VAE
   Compresses real video frames into discrete tokens.
   Also reconstructs those tokens back into pixels.

2. Text Encoder
   Converts captions/prompts into embeddings.

3. Video AR Transformer
   Learns to predict VAE video tokens conditioned on text.
```

Generation looks like this:

```text
prompt text
  -> text encoder
  -> video AR predicts VAE tokens
  -> VAE decoder turns tokens into frames
  -> mp4 output
```

The VAE is the foundation. If the VAE reconstruction is bad, video generation cannot be good.

## Hard Truth

AxiomAI does not magically know what "dog running" means.

It learns that from paired examples:

```text
dog_run_001.mp4
dog_run_001.txt -> "a dog running on grass"

dog_run_002.mp4
dog_run_002.txt -> "a dog running across a yard"

dog_walk_001.mp4
dog_walk_001.txt -> "a dog walking on grass"
```

The model connects words to motion only when your dataset repeatedly shows the connection.

No paired captions means no prompt control.

## Required Video Dataset Layout

Put video clips in:

```text
data/Videos/
```

Every `.mp4` must have a matching `.txt` file with the exact same base name.

Correct:

```text
data/Videos/
  dog_running_0001.mp4
  dog_running_0001.txt
  dog_running_0002.mp4
  dog_running_0002.txt
  car_driving_0001.mp4
  car_driving_0001.txt
```

Wrong:

```text
dog_running_0001.mp4
dog_running.txt
```

Wrong:

```text
dog_running_0001.mp4
dog_running_0001.txt.txt
```

Wrong:

```text
dog_running_0001.mp4
```

Missing captions are skipped or rejected depending on strict loading.

## Current Recommended Clip Settings

The current config is designed for CPU-first training:

```ini
width = 144
height = 128
fps = 8.0
duration = 2.0
```

That means each training sample is:

```text
16 frames
144x128 resolution
2 seconds long
8 frames per second
```

This is intentional. More clean low-resolution clips are better than a few high-resolution clips.

Recommended resolution ladder:

```text
64x64:
  Fast pipeline test only.

128x128:
  Good starter size.

144x128:
  Current recommended CPU setting.

192x192:
  Heavier, only after the pipeline works.

256x256:
  Expensive for CPU-only training.

512x512:
  Not realistic for this setup.
```

## Caption Rules

Captions should be short, visual, and consistent.

Good:

```text
a brown dog running on grass
a black car driving down a road
a person walking across a room
water flowing in a river
clouds moving across the sky
```

Bad:

```text
this is a beautiful cinematic scene that gives an emotional feeling
random clip
video 123
dog
movement
```

Very short captions like `dog` are usually too weak. The model needs object plus action plus scene when possible.

Best pattern:

```text
a [subject] [action] [place/background]
```

Examples:

```text
a dog running on grass
a dog walking on grass
a dog sitting on grass
a cat walking across a floor
a car driving on a city street
a fire burning in a fireplace
```

Consistency matters. If you use ten different phrases for the same concept, the tiny model has to learn all ten.

Use boring captions first. Fancy captions later.

## Dataset Size Targets

Rough practical targets:

```text
10 clips:
  Pipeline smoke test.

50 clips:
  VAE can start learning the visual domain.

200 clips:
  Better VAE training and early prompt/action patterns.

1,000 clips:
  Much better for repeated object/action concepts.

10,000 clips:
  Serious small video dataset territory.
```

For prompt control, each concept needs repeated examples:

```text
dog running:
  50+ clips is much better than 3 clips.

car driving:
  50+ clips is much better than 3 clips.

fire burning:
  50+ clips is much better than 3 clips.
```

The model cannot learn a visual concept from one example.

## Where To Get Video Data

Use data you have rights to use.

Good sources:

```text
Your own phone videos
Your own screen recordings
Public-domain video
Open-license datasets
Synthetic clips you generate yourself
Small curated clips from sources that permit your use
```

Be careful with:

```text
Random social media clips
Movie scenes
TV clips
Copyrighted YouTube videos
Private videos
Videos with identifiable people if you do not have permission
```

For a personal local experiment, you still want clean, legal, repeatable data. Dirty data wastes training time.

## Best Way To Build A Dataset

Start with folders by concept:

```text
raw_video_sources/
  dog_running/
    source1.mp4
    source2.mp4

  dog_walking/
    source1.mp4

  car_driving/
    source1.mp4

  water_flowing/
    source1.mp4
```

Then cut them into 2-second clips and output:

```text
data/Videos/
  dog_running_0001.mp4
  dog_running_0001.txt
  dog_running_0002.mp4
  dog_running_0002.txt
```

Captions can be generated from folder names at first:

```text
dog_running -> a dog running
car_driving -> a car driving
water_flowing -> water flowing
```

Then improve the captions by hand:

```text
a dog running on grass
a car driving on a wet road
water flowing over rocks
```

## Training Order

Run:

```bash
python main.py
```

Then use:

```text
7. Train Video VAE
8. Pre-Train Text Encoder
9. Train Video Model
```

Do not train Video AR before the VAE is decent.

Recommended video order:

```text
1. Prepare video/caption pairs.
2. Train Video VAE.
3. Inspect previews in model/video_model/previews/.
4. Confirm PSNR and codebook usage are acceptable.
5. Train Video Text Encoder.
6. Train Video AR.
7. Generate with /video prompt.
```

## Video VAE

The VAE compresses video frames into discrete tokens.

It must learn:

```text
Color
Shape
Edges
Frame-to-frame motion
How to reconstruct the original clip
How to use the codebook instead of collapsing to a few tokens
```

Current stronger VAE config:

```ini
base_channels = 48
latent_channels = 8
codebook_size = 4096
res_blocks_per_stage = 2
residual_scale = 0.707
use_ema_quantizer = True
ema_decay = 0.99
```

Current VAE loss:

```ini
vae_l1_weight = 1.0
vae_mse_weight = 0.5
vae_vq_weight = 1.0
vae_temporal_weight = 0.1
vae_spatial_weight = 0.05
vae_grad_clip = 1.0
vae_cosine_schedule = True
```

What those mean:

```text
L1:
  Helps sharper reconstructions.

MSE:
  Supports stable reconstruction and PSNR tracking.

VQ:
  Keeps the discrete codebook training.

Temporal:
  Penalizes bad frame-to-frame motion differences.

Spatial:
  Penalizes bad edge/structure differences.

EMA quantizer:
  Updates codebook entries more stably than direct gradient-only embedding training.
```

## VAE Quality Metrics

During VAE training, AxiomAI reports:

```text
T Loss:
  Training loss.

V Loss:
  Validation loss.

PSNR:
  Reconstruction quality estimate from MSE.

Codebook used:
  Percent of VAE codebook tokens used during validation.
```

Rough PSNR interpretation:

```text
Under 12 dB:
  Bad. Reconstructions are likely not usable.

12-16 dB:
  Weak. May show rough blobs/motion, but not enough for serious AR.

16-20 dB:
  Usable starting point for tiny local video experiments.

20-25 dB:
  Much better for this project.

25+ dB:
  Strong for a tiny VAE, but still inspect previews.
```

PSNR is not everything. Always inspect preview images.

Preview path:

```text
model/video_model/previews/
```

Each preview strip shows original and reconstructed frames. If the preview is bad, the VAE is bad. Do not move to AR training yet.

## Codebook Usage

The VAE codebook is the token vocabulary for video.

If codebook usage is near zero, the VAE is collapsing. That means it is using too few visual tokens and the AR model will learn poor video language.

Current quality gate:

```ini
min_vae_psnr_for_ar = 16.0
min_vae_code_usage_for_ar = 0.005
enforce_vae_quality_gate = True
```

`0.005` means 0.5 percent of the 4096-code codebook. That is a low minimum gate, not a final quality target.

For serious improvement, you want usage to climb meaningfully over training.

If codebook usage stays low:

```text
Train longer.
Use more diverse clips.
Lower ema_decay slightly, for example 0.98.
Enable dead_code_refresh only if needed.
Reduce codebook_size if the dataset is tiny.
Increase dataset variety.
```

## Video Text Encoder

The text encoder learns caption language for the video pipeline.

It trains from `.txt` captions in:

```text
data/Videos/
```

If captions are weak, prompt control will be weak.

Caption quality matters more than caption length.

Good:

```text
a dog running across grass
```

Weak:

```text
dog
```

Bad:

```text
cool video lol
```

## Video AR Transformer

The Video AR model learns to predict VAE tokens conditioned on text.

It does not see raw pixels directly. It sees VAE token grids.

This means:

```text
Bad VAE -> bad AR training.
Good VAE + bad captions -> weak prompt control.
Good VAE + good captions + enough clips -> real learning.
```

The AR model uses a two-scale setup:

```text
Scale 0:
  Coarse low-resolution structure.

Scale 1:
  Full latent resolution refinement conditioned on the coarse structure.
```

This is designed to reduce CPU/RAM cost compared with predicting only a full large token grid.

## Latent Cache

VAE encoding is expensive. AxiomAI can cache VAE tokens:

```ini
use_latent_cache = True
latent_cache_path = data/processed/video_latent_cache.pt
rebuild_latent_cache = False
```

First AR run:

```text
Builds latent cache.
```

Later AR runs:

```text
Loads cached VAE tokens.
Skips repeated video decode and VAE encode.
Trains faster and more consistently.
```

Rebuild the cache when:

```text
You add/remove video clips.
You change VAE checkpoint.
You change video width/height/fps/duration.
You change caption max length/tokenizer.
You suspect stale cache.
```

To force rebuild:

```ini
rebuild_latent_cache = True
```

Set it back after one run:

```ini
rebuild_latent_cache = False
```

## Generation

Start chat:

```text
6. Chat with Model
```

Use:

```text
/video a dog running on grass
```

The generated video is written to:

```text
out_video/
```

Generation quality depends on:

```text
VAE reconstruction quality
Caption consistency
Text encoder quality
Video AR training quality
Dataset size
Sampling behavior
```

If prompts do not control output, the model has not learned the caption-to-video relationship strongly enough.

## Overfitting

AxiomAI warns when train and validation behavior diverge.

For VAE:

```text
Train loss falls
Validation loss rises
Preview gets worse on validation clips
```

That means the VAE is memorizing training clips instead of learning useful compression.

For Video AR:

```text
Train loss falls
Validation loss rises
Generated clips become repetitive
```

That means AR is memorizing token patterns.

Fixes:

```text
More data
Cleaner validation split
Lower learning rate
Fewer epochs
More varied clips
Less aggressive model size
```

## Config Change Rules

Changing these requires retraining the VAE:

```text
base_channels
latent_channels
codebook_size
res_blocks_per_stage
spatial_downsample
temporal_downsample
width
height
fps
duration
```

Changing these usually requires rebuilding latent cache:

```text
video files
captions
VAE checkpoint
width
height
fps
duration
tokenizer
caption max length
```

Changing these affects AR training:

```text
d_model
n_layers
n_heads
max_seq_len
fine_tune_text_encoder
use_latent_cache
```

## Recommended Starting Config

For CPU-first realistic experiments:

```ini
[VAE]
base_channels = 48
latent_channels = 8
codebook_size = 4096
res_blocks_per_stage = 2
use_ema_quantizer = True

[TRAINING]
width = 144
height = 128
batch_size = 2
lr = 1e-4
epochs = 50
val_split = 0.2
strict_video_loading = True
use_latent_cache = True
save_vae_recon_preview = True
enforce_vae_quality_gate = True

[DATA]
fps = 8.0
duration = 2.0
```

If training is too slow:

```text
Lower width/height to 128x128.
Lower base_channels to 32.
Use fewer clips while testing.
Keep use_latent_cache = True for AR.
```

If quality is too poor:

```text
Use cleaner clips.
Train VAE longer.
Increase dataset size.
Keep captions consistent.
Inspect previews every few epochs.
Consider base_channels 64 on a strong PC CPU.
```

## Data Validation

AxiomAI validates video files before training.

It checks:

```text
Can OpenCV open the mp4?
Does it have readable frames?
Does it have enough frames for fps * duration?
Does the matching caption exist?
Is the caption non-empty?
```

With strict loading:

```ini
strict_video_loading = True
```

Fatal data problems stop training. This is good. Silent blank samples poison the model.

## Common Failure Modes

```text
VAE preview is blurry:
  Train longer, clean data, lower resolution, or increase VAE capacity.

VAE preview has wrong motion:
  Temporal loss may need more training/data. Check clip frame rate and duration.

Codebook usage stays near zero:
  More varied data, longer training, lower ema_decay, or smaller codebook.

AR refuses to train:
  VAE checkpoint missing, old, incompatible, or below quality gate.

Generation outputs noise:
  VAE is poor, AR is undertrained, or dataset is too small.

Prompt ignored:
  Captions are inconsistent, text encoder is weak, or too few examples connect that prompt to motion.

Training crashes on old checkpoint:
  VAE architecture changed. Retrain VAE.

Latent cache mismatch:
  Rebuild latent cache.
```

## What To Aim For First

Do not aim for cinematic video first.

Aim for this order:

```text
1. VAE reconstructs clips visibly.
2. VAE PSNR passes the gate.
3. Codebook usage is not collapsed.
4. AR validation loss improves.
5. Generated video has consistent colors/shapes.
6. Generated video has basic motion.
7. Prompt controls object/action.
8. Then increase dataset size and quality.
```

Trying to skip to high-quality generation before step 1 works is wasted time.

## Practical Dataset Plan

Start with 5 visual concepts:

```text
dog running
dog walking
car driving
water flowing
clouds moving
```

Make 50 clips per concept if possible:

```text
250 total clips
2 seconds each
144x128
8 fps
consistent captions
```

Then train:

```text
VAE until previews are recognizable.
Text encoder until caption loss improves.
AR until validation loss improves.
```

Then test prompts:

```text
/video a dog running on grass
/video a car driving on a road
/video clouds moving across the sky
```

Only add more concepts after the first concepts work.

## Final Rule

Video generation quality comes from the whole chain:

```text
clean paired clips
consistent captions
strong enough VAE
healthy codebook
text encoder that understands captions
AR model trained on enough examples
```

If any link is bad, the output is bad. Fix the earliest broken link first.
