# Using Weights & Biases (wandb) with raw-bench

[Weights & Biases (wandb)](https://wandb.ai/) is a web-based tool for experiment tracking and visualization `raw-bench` supports wandb for logging.

---

## Installation

`wandb` is included in `pyproject.toml`, so you don't need to install it separately if you use `pip install raw-bench`.  
If it is missing, you can install it manually:

```bash
pip install wandb
```

---

## How to Disable wandb Logging

By default, `raw-bench` does **not** use `wandb`, so you don't need to do anything to disable it.  
If you want to explicitly disable wandb, add `wandb=false` as an argument for your experiment:

```bash
python scripts/eval.py wandb=false ... --config-name=../configs/wavmark/eval_strict.yaml
```

In this mode, your experiment will automatically switch to TensorBoard logging.

---

## How to Enable wandb Logging

By default, wandb logging is **disabled** in `raw-bench`.  
In `configs/eval.yaml`, you will find `wandb: false`.  
To enable wandb, set the `wandb` argument to `true` when running your experiment:

```bash
python scripts/eval.py wandb=true... --config-name=../configs/wavmark/eval_strict.yaml
```

---

## Logging In

You need to log in to wandb to use it. You can log in interactively or set your API key as an environment variable:

```bash
wandb login
# or
export WANDB_API_KEY=your_api_key_here
```

Before running your experiment with wandb, make sure you have logged in successfully.  
Otherwise, the experiment will automatically switch to TensorBoard logging.

---

## More Information

- [wandb Documentation](https://docs.wandb.ai/)
- [wandb Quickstart](https://docs.wandb.ai/quickstart)
