# InceptionRAG

> **Accepted by the ACM Conference on Computer and Communications Security (ACM CCS 2026).**

**Authors:** Jiachang Zhang (Zhejiang University), Min Chen (Vrije Universiteit Amsterdam), Xiao Ren (Zhejiang University), Zhenyong Zhang (Guizhou University), Yuanchao Shu (Zhejiang University), Yunjun Gao (Zhejiang University), and Zhikun Zhang* (Zhejiang University).

\* Corresponding author.

A multi-stage adversarial attack framework targeting Retrieval-Augmented Generation (RAG) systems.

## Prepare Environment on DISA tcsh
If your home directory has a strict quota (or the dataset/corpus cache is too large for the default filesystem), keep the large generated artifacts under `/tmp` and point the attack at those paths explicitly.

```bash
# Create virtual environment in /tmp
mkdir -p /tmp/$USER/venvs
python -m venv /tmp/$USER/venvs/InceptionRAG
source /tmp/$USER/venvs/InceptionRAG/bin/activate

# Copy data files from repo into /tmp
cp -r data/suffix_pool /tmp/$USER/InceptionRAG_data/suffix_pool

# 1. Install dependencies
pip install -r requirements.txt

# Redirect caches to /tmp
# Hugging Face caches
setenv HF_HOME /tmp/$USER/huggingface
setenv HF_HUB_CACHE /tmp/$USER/huggingface/hub
setenv TRANSFORMERS_CACHE /tmp/$USER/huggingface/transformers
setenv XDG_CACHE_HOME /tmp/$USER/.cache

# PyTorch cache
setenv TORCH_HOME /tmp/$USER/torch

# Temporary files
setenv TMPDIR /tmp/$USER/tmp

# Create the directories
mkdir -p \
    $HF_HOME \
    $HF_HUB_CACHE \
    $TRANSFORMERS_CACHE \
    $XDG_CACHE_HOME \
    $TORCH_HOME \
    $TMPDIR
```

## Run InceptionRAG Baselines

```bash
# 2. Download datasets and build corpus index
# Dataset options are "nq", "hotpotqa", "msmarco"
python data/download_datasets.py --dataset nq/hotpotqa/msmarco --output-dir path/to/data_folder


# 3. Configure (add your API key)
# Edit config.yaml with your API credentials
# Set the

# 4. Run attack with data
# only use parameter --stage1 for first time to generate adversarial data
python -m run_attack.py --config config.yaml --dataset nq/hotpotqa/msmarco --limit 100 --data-root path/to/data_folder --output /path/to/results/output.json --add-gt --llm-judge --stage1

```

### Baselines
To get the baseline results for each dataset in InceptionRAG, run the following command:
```bash
python run_baselines.py --data-root path/to/data --output-dir /path/to/results-dir --dataset nq/hotpotqa/msmarco
```

This code reads each attack configuration in the `baselines_config.txt` file. The file is structured as follows, if you wish to change it:
```
#attack-name
attack:	
  retrieval_top_k: 20
  retrieval_first_k: 20
  cttft_neighbor_k: 10
  noise_injection: true
  dual_injection: false
  query_prefix: false
  
#attack-name
attack:	
  retrieval_top_k: 20
  retrieval_first_k: 20
  cttft_neighbor_k: 10
  noise_injection: true
  dual_injection: true
  query_prefix: false

...
```


## Run InceptionRAG CHUNKING
 ```bash
# download & extract the wiki documents related to the requested dataset
python data/download_wiki_corpus.py --dataset nq/hotpotqa/msmarco --data-root path/to/data_folder

# configure config
# USE CONFIG_MODIFIED

# run attack with chunking
python run_attack.py --config config_modified.yaml --llm-judge --add-gt --dataset nq/hotpotqa/msmarco --data-root /path/to/data_folder --output path/to/results/output.json
 ```

### configurations
Set the configuration in `configs/config_modified.yaml`

Each configurations run can be found in `configs/chunker_configs.txt`

To run the chunker configurations consecutively, run the `run_chunkings.py` script with the command below. The script reads the configurations in the `configs/chunker_configs.txt` file and runs the `run_attack.py` code for each configuration.
```bash
python run_chunkings.py --dataset nq/hotpotqa/msmarco --data-root path/to/data_folder --output-dir path/to/output_di
```


For a full run, pass the `/tmp` file paths directly:

```bash
cd /cs/home/$USER/Documents/InceptionRAG
python -m InceptionRAG.run_attack \
  --config /cs/home/$USER/Documents/InceptionRAG/config.yaml \
  --input-queries /tmp/$USER/inception_rag/samples/hotpotqa_queries_100.json \
  --suffix-pool /cs/home/$USER/Documents/InceptionRAG/data/suffix_pool/suffix_candidates.json \
  --corpus-embs /tmp/$USER/inception_rag/corpus/hotpotqa_contriever_embs.npy \
  --corpus-texts /tmp/$USER/inception_rag/corpus/hotpotqa_corpus_texts.json \
  --output /tmp/$USER/inception_rag/results/hotpotqa_attack_100.json \
  --limit 100 \
  --stage1
```

This pattern keeps the heavy outputs away from the quota-limited home directory while still using the project code from the repository checkout under `/cs/home/$USER/Documents/InceptionRAG`.

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
