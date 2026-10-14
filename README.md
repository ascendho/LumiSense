# LumiSense

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-2ea44f.svg)](LICENSE)

Public implementation of the LumiSense edge diagnosis framework.

> [!NOTE]
> This repository is a streamlined, code-focused reorganization of the
> implementation accompanying the paper. It is intended to make the core
> workflow and logic easy to inspect. Because files were selectively
> consolidated for release, minor omissions or mistakes may remain. Please
> treat it as a reference implementation and report issues when you find them.

LumiSense adapts a small language model for evidence-grounded industrial IoT
root-cause diagnosis. The code is organized by workflow stage:

| Path | Purpose |
|---|---|
| `code/sft/` | Supervised fine-tuning data preparation, training, and evaluation. |
| `code/kd/` | Verified teacher annotation and class-token distillation. |
| `code/wise_ft/` | LoRA adapter interpolation between SFT and KD checkpoints. |
| `code/quantization/` | LoRA merge, GGUF export, quantization gate, and CPU benchmark helpers. |
| `code/cascade/` | Confidence-based edge-cloud routing utilities. |
| `code/baselines/` | Traditional and prompted-LLM baseline scripts. |

Released LumiSense model weights are hosted under the author's [Hugging Face
account](https://huggingface.co/ascendho).

## Install

```bash
python3 -m pip install -e ".[test]"
```

For Apple-Silicon MLX experiments:

```bash
python3 -m pip install -e ".[mac,test]"
```

Traditional baselines additionally use scikit-learn:

```bash
python3 -m pip install -e ".[baseline]"
```

## Smoke Tests

```bash
python3 -m unittest discover -s tests -v
DRY_RUN=1 code/sft/run_sft.sh --model /path/to/Qwen3.5-0.8B-MLX-4bit
```

## Citation

If you use the LumiSense code or released model, please cite the associated
paper. The paper has been accepted at CBASE 2026; page numbers, DOI, and the
final proceedings record will be added once IEEE publishes it:

```bibtex
@inproceedings{lumisense2026,
  title     = {LumiSense: Evidence-Grounded Edge Sensor Diagnosis with Adapted Small Language Models},
  author    = {He, Sheng and Li, Haifeng and Wang, Binxu and Zhang, Chong},
  booktitle = {2026 5th International Conference on Cloud Computing, Big Data Application and Software Engineering (CBASE 2026)},
  address   = {Chengdu, China},
  year      = {2026},
  note      = {Accepted; published by IEEE, ISBN 979-8-3195-0947-5}
}
```

## License

MIT. Model weights, adapters, raw prediction outputs, and paper-run data are
**not** included in this repository.
