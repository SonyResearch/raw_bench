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
