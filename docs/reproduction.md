# Reproducing Experiments with raw-bench

This guide will help you reproduce our audio watermarking experiments reported in our [paper](https://arxiv.org/abs/2505.19663) using the `raw-bench` framework.  
Follow the steps below for a smooth setup and evaluation experience.

---

## 1. Install raw-bench and Clone the Repository

First, clone the repository and install all dependencies:

```bash
git clone https://github.com/SonyResearch/raw_bench.git
cd raw_bench
pip install .
```

---

## 2. Download Required Datasets

Please follow the [dataset instructions](./datasets.md) to download and prepare the required datasets.

---

## 3. Download or Prepare FFmpeg

Some experiments require a custom FFmpeg binary for codec attacks.  
See [FFmpeg instructions](./ffmpeg.md) for details.

---

## 4. Reproduce Experiments for Each Model

Below are step-by-step instructions for each supported watermarking model.

---

### AudioSeal

1. **Download the AudioSeal checkpoint**  
   See [AudioSeal instructions](./wm_models/audioseal.md) to set up the environment.

2. **Run evaluation (strict attacks):**
   ```bash
   python scripts/eval.py \
       run_dir=runs/audioseal/pretrained_on_strict \
       ffmpeg4codecs=ffmpeg/ffmpeg-7.0.2-amd64-static/ffmpeg \
       --config-name=../configs/audioseal/eval_strict.yaml
   ```

3. **Run evaluation (loose attacks):**
   ```bash
   python scripts/eval.py \
       run_dir=runs/audioseal/pretrained_on_loose \
       ffmpeg4codecs=ffmpeg/ffmpeg-7.0.2-amd64-static/ffmpeg \
       --config-name=../configs/audioseal/eval_loose.yaml
   ```

---

### SilentCipher

1. **Download the SilentCipher checkpoint**  
   See [SilentCipher instructions](./wm_models/silentcipher.md) to set up the environment.

2. **Run evaluation (strict attacks):**
   ```bash
   python scripts/eval.py \
       run_dir=runs/silentcipher/pretrained_on_strict \
       ffmpeg4codecs=ffmpeg/ffmpeg-7.0.2-amd64-static/ffmpeg \
       --config-name=../configs/silentcipher/eval_strict.yaml
   ```

3. **Run evaluation (loose attacks):**
   ```bash
   python scripts/eval.py \
       run_dir=runs/silentcipher/pretrained_on_loose \
       ffmpeg4codecs=ffmpeg/ffmpeg-7.0.2-amd64-static/ffmpeg \
       --config-name=../configs/silentcipher/eval_loose.yaml
   ```

---

### Timbre

1. **Download the Timbre checkpoint**  
   See [Timbre instructions](./wm_models/timbre.md) for the download link.

2. **Run evaluation (strict attacks):**
   ```bash
   python scripts/eval.py \
       run_dir=runs/timbre/pretrained_on_strict \
       ffmpeg4codecs=ffmpeg/ffmpeg-7.0.2-amd64-static/ffmpeg \
       --config-name=../configs/timbre/eval_strict.yaml
   ```

3. **Run evaluation (loose attacks):**
   ```bash
   python scripts/eval.py \
       run_dir=runs/timbre/pretrained_on_loose \
       ffmpeg4codecs=ffmpeg/ffmpeg-7.0.2-amd64-static/ffmpeg \
       --config-name=../configs/timbre/eval_loose.yaml
   ```

---

### WavMark

1. **Download the WavMark checkpoint**  
   See [WavMark instructions](./wm_models/wavmark.md) for the download link.

2. **Run evaluation (strict attacks):**
   ```bash
   python scripts/eval.py \
       run_dir=runs/wavmark/pretrained_on_strict \
       ffmpeg4codecs=ffmpeg/ffmpeg-7.0.2-amd64-static/ffmpeg \
       --config-name=../configs/wavmark/eval_strict.yaml
   ```

3. **Run evaluation (loose attacks):**
   ```bash
   python scripts/eval.py \
       run_dir=runs/wavmark/pretrained_on_loose \
       ffmpeg4codecs=ffmpeg/ffmpeg-7.0.2-amd64-static/ffmpeg \
       --config-name=../configs/wavmark/eval_loose.yaml
   ```

---

## 5. (Optional) Enable wandb Logging

If you want to track your experiments with [Weights & Biases (wandb)](https://wandb.ai/), see [wandb instructions](./wandb.md).
