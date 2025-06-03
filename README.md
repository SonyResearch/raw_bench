# Robust Audio Watermarking Benchmark (RAW-Bench)

_Accompanying website for the paper:_  
**A Comprehensive Real-World Assessment of Audio Watermarking Algorithms: Will They Survive Neural Codecs?**  
_Yigitcan Özer, Woosung Choi, Joan Serrà, Mayank Kumar Singh, Wei-Hsiang Liao, Yuki Mitsufuji_  
_To appear at Interspeech 2025, Rotterdam, The Netherlands_

---

## Abstract

We introduce the **Robust Audio Watermarking Benchmark (RAW-Bench)**, a benchmark for evaluating deep learning-based audio watermarking methods with standardized and systematic comparisons. To simulate real-world usage, we introduce a comprehensive audio attack pipeline with various distortions such as compression, background noise, and reverberation, along with a diverse test dataset including speech, environmental sounds, and music recordings. Evaluating four existing watermarking methods on RAW-bench reveals two main insights: (i) neural compression techniques pose the most significant challenge, even when algorithms are trained with such compressions; and (ii) training with audio attacks generally improves robustness, although it is insufficient in some cases. Furthermore, we find that specific distortions, such as polarity inversion, time stretching, or reverb, seriously affect certain methods.

---

# ViSQOL Installation Guide

This guide describes how to install [ViSQOL](https://github.com/google/visqol), a perceptual audio quality model, using Conda and pip.

## ✅ Requirements

- Python 3.10
- Bazel 5.3.0
- TensorFlow 2.13
- A Linux system (tested on Ubuntu-based distributions)

## 🚀 Step-by-Step Installation

### 🔹 Option A: Create a New Conda Environment

```bash
conda create -n visqol-env python=3.10 numpy scipy matplotlib sox bazel=5.3.0 -c conda-forge
conda activate visqol-env
```

### 🔹 Option B: Use an Existing Conda Environment
```bash
conda install numpy scipy matplotlib sox bazel=5.3.0 -c conda-forge
```

### 2. Install required Python packages

```bash
pip install tensorflow==2.13 absl-py protobuf
```

### 3. Clone the ViSQOL repository

```bash
git clone https://github.com/google/visqol.git
cd visqol
```

### 4. Install ViSQOL in editable mode

```bash
pip install -e .
```

## ✅ Verification

You can verify the installation by running a test import:

```python
import visqol
api = visqol.VisqolApi()
```

## ⚠️ Notes

- The SVR model file is required at runtime. It will be loaded automatically from the installed package data if installed correctly.
- You do **not** need to manually build with Bazel.
- If you get `StatusNotOk` or `libsvm_nu_svr_model.txt` errors, make sure the model file is found by `visqol`.

---

For more usage details and test samples, visit the [ViSQOL GitHub repository](https://github.com/google/visqol).
