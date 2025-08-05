# SilentCipher

**SilentCipher** is the first deep learning-based model to incorporate psychoacoustic model-based thresholding, enabling the embedding of imperceptible watermarks in audio. It achieves state-of-the-art robustness while maintaining audio quality across a wide range of content.

- [Paper (interspeech 2024)](https://arxiv.org/abs/2406.03822)
- [Official GitHub Repository](https://github.com/sony/silentcipher)
- [Colab notebook](https://colab.research.google.com/github/sony/silentcipher/blob/master/examples/colab/demo.ipynb)
- [Huggingface](https://github.com/sony/silentcipher?tab=readme-ov-file)
---
## Note

Due to a technical limitation, `raw_bench` currently does not use the [`silentcipher`](https://pypi.org/project/silentcipher/) package directly. Instead, a custom wrapper interface for SilentCipher is used. This will be updated in a future release. 

## Reproducing Experiments

### Prerequisites

- [Download datasets](../datasets.md)
- [Install ffmpeg](../ffmpeg.md)
- [(optional) If you want to use WandB](../wandb.md)

To reproduce the evaluation, download the pretrained **SilentCipher** checkpoint by initializing the submodule `wm_ckpts/silent_cipher/`:

```bash
cd $raw_bench_home
git submodule update --init wm_ckpts/silent_cipher/
```

Then run the evaluation on strict attacks:

```bash
python scripts/eval.py \
    run_dir=runs/silentcipher/pretrained_on_strict \
    ffmpeg4codecs=ffmpeg/ffmpeg-7.0.2-amd64-static/ffmpeg \
    --config-name=../configs/silentcipher/eval_strict.yaml
```

Also, run the evaluation on loose attacks:

```bash
python scripts/eval.py \
    run_dir=runs/silentcipher/pretrained_on_loose \
    ffmpeg4codecs=ffmpeg/ffmpeg-7.0.2-amd64-static/ffmpeg \
    --config-name=../configs/silentcipher/eval_loose.yaml
```


---

## Citation

If you use SilentCipher in your work, please cite the following:

```bibtex
@inproceedings{singh24_interspeech,
  author={Mayank Kumar Singh and Naoya Takahashi and Weihsiang Liao and Yuki Mitsufuji},
  title={{SilentCipher: Deep Audio Watermarking}},
  year=2024,
  booktitle={Proc. INTERSPEECH 2024},
}
```