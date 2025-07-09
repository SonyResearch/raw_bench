# Timbre Watermarking

**Timbre Watermarking** embeds robust and imperceptible watermarks into speech audio, speficically designed to **defend against voice cloning attacks**. This method is proposed by by Chang Liu, Jie Zhang, Tianwei Zhang, Xi Yang, Weiming Zhang, and Nenghai Yu, and presented at **NDSS 2024**.

- [Audio Samples & Project Website](https://timbrewatermarking.github.io/samples)
- [Paper (NDSS 2024)](https://www.ndss-symposium.org/ndss-paper/detecting-voice-cloning-attacks-via-timbre-watermarking/)
- [Official GitHub Repository](https://github.com/TimbreWatermarking/TimbreWatermarking)

---

## Setup (using `raw_bench`)

In `raw_bench`, we used a [nearly identical version of the official repository](https://github.com/Woosung-sony/TimbreWatermarking) with a minor modification to fix relative path issues and upgrade `librosa` to version 0.10.2.

Before running, initialize the submodule `wm_ckpts/timbre/`:

```bash
cd $raw_bench_home
git submodule update --init wm_ckpts/timbre/
```

---

## Reproducing Experiments

### Prerequisites

- [Download datasets](../datasets.md)
- [Install ffmpeg](../ffmpeg.md)
- [(optional) If you want to use WandB](../wandb.md)

To reproduce the evaluation, download the pretrained **30-bit timbre watermark** checkpoint:

🔗 [Download Checkpoint](https://drive.google.com/drive/folders/131ZtATH5jKPOEqNJQVtVIS-m6dVhlNtC)

Move the file to a dedicated directory:

```bash
mkdir -p wm_ckpts/timbre_ckpt
mv compressed_none-conv2_ep_20_2023-02-14_02_24_57.pth.tar wm_ckpts/timbre_ckpt/
```

Then run the evaluation on strict attacks:

```bash
python scripts/eval.py \
    run_dir=runs/timbre/pretrained_on_strict \
    checkpoint=wm_ckpts/timbre_ckpt/compressed_none-conv2_ep_20_2023-02-14_02_24_57.pth.tar \
    ffmpeg4codecs=ffmpeg-git-20240629-amd64-static/ffmpeg \
    wandb=false \
    --config-name=../configs/timbre/eval_strict.yaml
```

Also, run the evaluation on loose attacks:

```bash
python scripts/eval.py \
    run_dir=runs/timbre/pretrained_on_loose \
    checkpoint=wm_ckpts/timbre_ckpt/compressed_none-conv2_ep_20_2023-02-14_02_24_57.pth.tar \
    ffmpeg4codecs=ffmpeg-git-20240629-amd64-static/ffmpeg \
    wandb=false \
    --config-name=../configs/timbre/eval_loose.yaml
```

---

## Citation

If you use Timbre Watermarking in your work, please cite the following:

```bibtex
@inproceedings{timbrewatermarking-ndss2024,
  title = {Detecting Voice Cloning Attacks via Timbre Watermarking},
  author = {Liu, Chang and Zhang, Jie and Zhang, Tianwei and Yang, Xi and Zhang, Weiming and Yu, Nenghai},
  booktitle = {Network and Distributed System Security Symposium},
  year = {2024},
  doi = {10.14722/ndss.2024.24200},
}
```