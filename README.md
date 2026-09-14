# InceptionRAG

> **Accepted by the ACM Conference on Computer and Communications Security (ACM CCS 2026).**

**Authors:** Jiachang Zhang (Zhejiang University), Min Chen (Vrije Universiteit Amsterdam), Xiao Ren (Zhejiang University), Zhenyong Zhang (Guizhou University), Yuanchao Shu (Zhejiang University), Yunjun Gao (Zhejiang University), and Zhikun Zhang* (Zhejiang University).

\* Corresponding author.

A multi-stage adversarial attack framework targeting Retrieval-Augmented Generation (RAG) systems.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets and build corpus index
bash scripts/download_all_datasets.sh

# 3. Configure (add your API key)
# Edit config.yaml with your API credentials

# 4. Run attack with example data
python -m InceptionRAG.attack_example --config config.yaml --dataset nq --limit 20
```

## Overview

InceptionRAG exploits RAG pipelines through a three-stage attack:

1. **Stage 1: Generative Manipulation via Optimized Logic Traps**
   - Generate Logic Trap documents (Attribute + Bridge)
   - ZOSO: Zeroth-Order Suffix Optimization

2. **Stage 2: Retrieval Hijacking via Dual-Ended Injection**
   - Subject Header + Footer Metadata
   - Two-Stage Retrieval

3. **Stage 3: Polymorphic Obfuscation and CTTFT**
   - Polymorphic Noise Injection
   - CTTFT: Cascade Test-Time Fine-Tuning (L1→L2→L3)

## Usage

### Quick Attack (Recommended)

Use pre-generated example data for quick testing:

```bash
python -m InceptionRAG.attack_example --config config.yaml --dataset nq --limit 20
```

Supported datasets: `nq`, `hotpotqa`, `msmarco`

### Full Pipeline

Run complete attack from scratch (generates Logic Traps + ZOSO optimization):

```bash
python -m InceptionRAG.run_attack --config config.yaml --dataset nq --limit 100 --output results.json
```

### Attack Only

Skip generation, use existing adversarial data:

```bash
python -m InceptionRAG.run_attack --config config.yaml --dataset nq --attack-only --output results.json
```

## Configuration

Edit `config.yaml`:

```yaml
llm:
  provider: "custom"           # openai / azure / custom
  api_key: "YOUR_API_KEY"
  api_base: "https://api.openai.com/v1"
  model: "gemini-2.0-flash"

attack:
  retrieval_top_k: 10
  retrieval_first_k: 20
  cttft_neighbor_k: 10
  noise_injection: true
  dual_injection: true
```

## Data Structure

```
data/
├── datasets/       # Raw datasets (BEIR format)
├── corpus/         # Corpus embeddings (.npy) and texts (.json)
├── suffix_pool/    # Suffix candidates + ZOSO results
└── example/        # Pre-generated example data
```

## HODOR Defense

HODOR (Heterogeneous Orthogonal Documents to Obstruct Reasoning) is a defense mechanism against multi-document poisoning attacks. It isolates each retrieved document and performs independent reasoning, then uses majority voting to determine the final answer.

### Enable HODOR Defense

```bash
# Attack with HODOR defense enabled
python -m InceptionRAG.attack_example --config config.yaml --dataset nq --limit 20 --hodor

# Full pipeline with HODOR
python -m InceptionRAG.run_attack --config config.yaml --dataset nq --hodor --output results.json
```

### How HODOR Works

1. Each retrieved document is processed independently by the LLM
2. Each document generates its own answer
3. Answers are normalized and counted via majority voting
4. The most common answer is returned as the final response

This breaks the adversarial logic chain that relies on cross-document dependencies.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | - | Dataset: nq, hotpotqa, msmarco |
| `--limit` | 100 | Number of samples |
| `--attack-only` | False | Skip Stage 1, use existing data |
| `--skip-attack` | False | Only run Stage 1 |
| `--hodor` | False | Enable HODOR defense |
| `--hodor-model` | gemini-2.0-flash | Model for HODOR voting |

## Citation

```bibtex
@inproceedings{zhang2026inceptionrag,
  title={InceptionRAG: Stealthy Poisoning Attack Against Retrieval-Augmented Generation},
  author={Zhang, Jiachang and Chen, Min and Ren, Xiao and Zhang, Zhenyong and Shu, Yuanchao and Gao, Yunjun and Zhang, Zhikun},
  booktitle={ACM Conference on Computer and Communications Security (CCS)},
  year={2026}
}
```

## License

MIT License
