![OpenSyntheticEmbeddings](title.png)
**Paper:** [Understanding the Influence of Synthetic Data for Text Embedders](https://aclanthology.org/2025.findings-acl.1160.pdf)

**Data:** [OpenSyntheticEmbeddings on Huggingface](https://huggingface.co/datasets/jspringer/open-synthetic-embeddings/viewer/synthetic_LLaMA-3.1-8B-Instruct_sts?views%5B%5D=synthetic_llama_31_8b_instruct_sts)

This repository provides tools for generating synthetic data for training text embedding models, supporting multiple data types including short-short, short-long, long-long, long-short, bitext, and STS data.

## Installation

```bash
git clone https://github.com/jakespringer/open-synthetic-embeddings.git
cd open-synthetic-data
pip install -e ose
```

## Quick Start

### Generate All Data Types
```bash
bash ose/scripts/run_all.sh
```

### Generate Specific Data Type
```bash
# Example: Generate short-long embeddings data
bash ose/scripts/run_short_long.sh
```

### Individual Steps
```bash
# Generate samples
python -m ose.generate --config ose/configs/short_long/short_long_generation.yaml

# Collect and postprocess
python -m ose.collect --config ose/configs/short_long/short_long_collect.yaml
```

## Configuration

All configurations are in `ose/configs/`. Each data type has its own folder with configs for different pipeline stages:

- `*_brainstorm.yaml` - Initial idea generation (where applicable)
- `*_brainstorm_collect.yaml` - Collect brainstormed ideas (where applicable)  
- `*_generation.yaml` - Main data generation
- `*_collect.yaml` - Data collection and postprocessing
- `*_hard_negatives.yaml` - Hard negative generation (for short_short, short_long, long_long)

Key settings to modify:
- `n_samples` - Number of samples to generate
- `model_path` - LLM model to use
- `overwrite_output` - Set to `true` to overwrite existing files

## Citation

If you use this repository or dataset, please cite:

```bibtex
@inproceedings{springer2025understanding,
  title={Understanding the Influence of Synthetic Data for Text Embedders},
  author={Springer, Jacob Mitchell and Adlakha, Vaibhav and Reddy, Siva and Raghunathan, Aditi and Mosbach, Marius},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  pages={22551--22567},
  year={2025}
}
```
