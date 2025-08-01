# Timbre Watermarking

**Timbre Watermarking** embeds robust and imperceptible watermarks into speech audio, speficically designed to **defend against voice cloning attacks**. This method is proposed by by Chang Liu, Jie Zhang, Tianwei Zhang, Xi Yang, Weiming Zhang, and Nenghai Yu, and presented at **NDSS 2024**.

- [Audio Samples & Project Website](https://timbrewatermarking.github.io/samples)
- [Paper (NDSS 2024)](https://www.ndss-symposium.org/ndss-paper/detecting-voice-cloning-attacks-via-timbre-watermarking/)
- [Official GitHub Repository](https://github.com/TimbreWatermarking/TimbreWatermarking)

---

## Notice

Timbre is distributed under the `GPL-3.0` license, and there is currently no official pip-installable package available. Due to license incompatibility, we cannot provide native support or redistribute Timbre within our `MIT`-licensed codebase. Therefore, Timbre is not officially integrated into this project.

However, we provide a Python wrapper example at `raw_bench/solver/timbre.py.example` to help you integrate Timbre manually.

To use this wrapper, you must implement the `Encoder` and `Decoder` classes in `raw_bench/model/timbre.py`:

```python
from ..model.timbre import Decoder, Encoder  # You need to implement these
```

After implementing these classes, ensure they are registered in `raw_bench/model/__init__.py`:

```python
from .timbre import Encoder, Decoder
```

> **Note:**  
> This manual integration is required to comply with licensing restrictions. Integration and use of Timbre Watermarking is solely the responsibility of the user. We do not provide support, maintenance, or guarantee compatibility for any manual integration. Please ensure you comply with all relevant licenses and usage terms when incorporating Timbre into your workflow. 

---

## Reproducing Experiments

### Prerequisites

- [Download datasets](../datasets.md)
- [Install ffmpeg](../ffmpeg.md)
- [(optional) If you want to use WandB](../wandb.md)

To reproduce the evaluation, download the pretrained **30-bit timbre watermark** checkpoint:

🔗 [Download Checkpoint](https://drive.google.com/drive/folders/131ZtATH5jKPOEqNJQVtVIS-m6dVhlNtC)

### Make a Wrapper Script

To use Timbre Watermarking with this project, you need to implement a wrapper for the Timbre Encoder and Decoder.

1. **Refer to the example:**  
  See the provided example script at `raw_bench/solver/timbre.py.example`.

2. **Implement Encoder and Decoder:**  
  In `raw_bench/model/timbre.py`, define the `Encoder` and `Decoder` classes according to the [official TimbreWatermarking repository](https://github.com/TimbreWatermarking/TimbreWatermarking).  
  For example, your import in the wrapper script should look like:
  ```python
  from ..model.timbre import Decoder, Encoder  # Implement these classes
  ```

3. **Register the classes:**  
  Ensure you register the new classes in `raw_bench/model/__init__.py`:
  ```python
  from .timbre import Encoder, Decoder
  ```


### Reproduction

**Run evaluation (strict attacks):**
   ```bash
   python scripts/eval.py \
       run_dir=runs/timbre/pretrained_on_strict \
       ffmpeg4codecs=ffmpeg/ffmpeg-7.0.2-amd64-static/ffmpeg \
       --config-name=../configs/timbre/eval_strict.yaml
   ```

**Run evaluation (loose attacks):**
   ```bash
   python scripts/eval.py \
       run_dir=runs/timbre/pretrained_on_loose \
       ffmpeg4codecs=ffmpeg/ffmpeg-7.0.2-amd64-static/ffmpeg \
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