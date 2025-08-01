# WavMark

**WavMark** introduced an innovative audio watermarking framework that encodes up to 32 bits of watermark within a mere 1-second audio snippet. Their work was published in 2023 and is available at: https://arxiv.org/pdf/2308.12770.

- [Paper (arxiv)](https://arxiv.org/pdf/2308.12770)
- [Official GitHub Repository](https://github.com/wavmark/wavmark)
- [Demo](https://wavmark.github.io/)


## Reproducing Experiments

### Prerequisites

- [Download datasets](../datasets.md)
- [Install ffmpeg](../ffmpeg.md)
- [(optional) If you want to use WandB](../wandb.md)

To reproduce the evaluation, download the pretrained **WavMark** checkpoint by initializing the submodule `wm_ckpts/wavmark/`:


Then run the evaluation on strict attacks:

```bash
python scripts/eval.py \
    run_dir=runs/wavmark/pretrained_on_strict \
    ffmpeg4codecs=ffmpeg-git-20240629-amd64-static/ffmpeg \
    wandb=false \
    --config-name=../configs/wavmark/eval_strict.yaml
```

Also, run the evaluation on loose attacks:

```bash
python scripts/eval.py \
    run_dir=runs/wavmark/pretrained_on_loose \
    ffmpeg4codecs=ffmpeg-git-20240629-amd64-static/ffmpeg \
    wandb=false \
    --config-name=../configs/wavmark/eval_loose.yaml
```

---

## Citation

If you use WavMark in your work, please cite the following:

```bibtex
@misc{chen2023wavmark,
      title={WavMark: Watermarking for Audio Generation}, 
      author={Guangyu Chen and Yu Wu and Shujie Liu and Tao Liu and Xiaoyong Du and Furu Wei},
      year={2023},
      eprint={2308.12770},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```