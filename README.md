# RAW-Bench: Robust Audio Watermarking Benchmark

_Accompanying website for the paper:_  
**A Comprehensive Real-World Assessment of Audio Watermarking Algorithms: Will They Survive Neural Codecs?**  
_Yigitcan Özer, Woosung Choi, Joan Serrà, Mayank Kumar Singh, Wei-Hsiang Liao, Yuki Mitsufuji_  
_To appear at Interspeech 2025, Rotterdam, The Netherlands_

---

## Abstract

We introduce the **Robust Audio Watermarking Benchmark (RAW-Bench)**, a benchmark for evaluating deep learning-based audio watermarking methods with standardized and systematic comparisons. To simulate real-world usage, we introduce a comprehensive audio attack pipeline with various distortions such as compression, background noise, and reverberation, along with a diverse test dataset including speech, environmental sounds, and music recordings. Evaluating four existing watermarking methods on RAW-Bench reveals two main insights: (i) neural compression techniques pose the most significant challenge, even when algorithms are trained with such compressions; and (ii) training with audio attacks generally improves robustness, although it is insufficient in some cases. Furthermore, we find that specific distortions, such as polarity inversion, time stretching, or reverb, seriously affect certain methods.


## Links

- [Paper](https://ai.sony/publications/A-Comprehensive-Real-World-Assessment-of-Audio-Watermarking-Algorithms-Will-They-Survive-Neural-Codecs/)
- [Documentation & Tutorials](docs/README.md)
- [PyPI package: `pip install raw-bench`](https://pypi.org/project/raw-bench/)


## TODO

- [x] Refactor core codebase
- [x] Release benchmark evaluation code for reproduction
- [x] Publish pip package
- [x] Add documentation and tutorials
- [ ] Add a guide for perceptual evaluation ([ViSQOL](https://github.com/google/visqol))

---

## Citation

If you find RAW-Bench useful in your research, please cite our paper:

```bibtex
@inproceedings{OezerChoiEtAl25_RAWBench_Interspeech,
 author = {Yigitcan {\"O}zer and Woosung Choi and Joan Serr{\`{a}} and Mayank Kumar Singh and Wei-Hsiang Liao and Yuki Mitsufuji},
 title = {A Comprehensive Real-World Assessment of Audio Watermarking Algorithms: Will They Survive Neural Codecs?},
 booktitle = {Proceedings of the Annual Conference of the International Speech Communication Association (Interspeech)},
 address = {Rotterdam, The Netherlands},
 year = {2025},
 doi = {},
 pages = {}
}
```
