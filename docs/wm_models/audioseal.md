# AudioSeal

**AudioSeal** is a method for speech localized watermarking, with state-of-the-art robustness and detector speed. AudioSeal introduces a breakthrough in proactive, localized watermarking for speech. Audioseal was presented at ICML 2024.

- [Paper (ICML 2024)](https://icml.cc/virtual/2024/poster/34713)
- [Official GitHub Repository](https://github.com/facebookresearch/audioseal)

---

## Setup (using `raw_bench`)

In `raw_bench`, we used the [audiocraft repository](https://github.com/facebookresearch/audiocraft). As its readme suggested, you need to install it with `wm` option (`git+https://github.com/facebookresearch/audiocraft@main#egg=audiocraft[wm]`). If you already install `raw_bench` then you don't have to install it by yourself.

---

## Reproducing Experiments

### Prerequisites

- [Download datasets](../datasets.md)
- [Install ffmpeg](../ffmpeg.md)
- [(optional) If you want to use WandB](../wandb.md)

To reproduce the evaluation, download the pretrained **AudioSeal** checkpoint by initializing the submodule `wm_ckpts/audioseal/`:

```bash
cd $raw_bench_home
git submodule update --init wm_ckpts/audioseal/
```

Then run the evaluation on strict attacks:

```bash
python scripts/eval.py \
    run_dir=runs/audioseal/pretrained_on_strict \
    ffmpeg4codecs=ffmpeg-git-20240629-amd64-static/ffmpeg \
    wandb=false \
    --config-name=../configs/audioseal/eval_strict.yaml
```

Also, run the evaluation on loose attacks:

```bash
python scripts/eval.py \
    run_dir=runs/audioseal/pretrained_on_loose \
    ffmpeg4codecs=ffmpeg-git-20240629-amd64-static/ffmpeg \
    wandb=false \
    --config-name=../configs/audioseal/eval_loose.yaml
```

---

## Citation

If you use AudioSeal / Audiocraft in your work, please cite the following:

```bibtex
<!-- audioseal -->
@article{sanroman2024proactive,
  title={Proactive Detection of Voice Cloning with Localized Watermarking},
  author={San Roman, Robin and Fernandez, Pierre and Elsahar, Hady and D´efossez, Alexandre and Furon, Teddy and Tran, Tuan},
  journal={ICML},
  year={2024}
}

<!-- audiocraft -->
@inproceedings{copet2023simple,
    title={Simple and Controllable Music Generation},
    author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}
```