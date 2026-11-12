# InceptionRAG

> **Accepted by the ACM Conference on Computer and Communications Security (ACM CCS 2026).**

**Authors:** Jiachang Zhang (Zhejiang University), Min Chen (Vrije Universiteit Amsterdam), Xiao Ren (Zhejiang University), Zhenyong Zhang (Guizhou University), Yuanchao Shu (Zhejiang University), Yunjun Gao (Zhejiang University), and Zhikun Zhang* (Zhejiang University).

\* Corresponding author.

A multi-stage adversarial attack framework targeting Retrieval-Augmented Generation (RAG) systems.




## Prepare Environment on DISA tcsh
If your home directory has a strict quota (or the dataset/corpus cache is too large for the default filesystem), keep the large generated artifacts under `/tmp` and point the attack at those paths explicitly.

```bash
# get inside the repo
cd InceptionRAG

# 1. Create virtual environment in /tmp
mkdir -p /tmp/$USER/venvs
python -m venv /tmp/$USER/venvs/InceptionRAG
source /tmp/$USER/venvs/InceptionRAG/bin/activate

# 2. Copy data files from repo into /tmp
# Your root data directory (data-root) is /tmp/$USER/InceptionRAG_data
cp -r data/suffix_pool /tmp/$USER/InceptionRAG_data/suffix_pool

# 3. Install dependencies
pip install -r requirements.txt

# 4. Redirect caches to /tmp
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

## Project InceptionRAG
Our project focuses on how InceptionRAG handles a typical RAG pipeline, which includes a chunker and reranker that InceptionRAG does not use. 

### Baseline
We start be establishing a baseline based on the original InceptionRAG

1. Download a dataset used in InceptionRAG and encode them as embeddings. (NOTE: this step will take multiple hours to embedding millions of passages)
- dataset options are "nq", "hotpotqa", and "msmarco"
```bash
python data/download_datasets.py  nq/hotpotqa/msmarco --output-dir path/to/data_folder
```

2. Configure the `configs/config.yaml` file. This file will be used to obtain the LLM and set up the RAG pipeline configurations
- This project uses the HuggingFace model Qwen/Qwen2.5-3B-Instruct. To use the same model, use the same llm configuration in `configs/config.yaml`. OR to use another HuggingFace model, change the `model` variable. Otherwise, follow the template in `configs/config.yaml.template`
- You can also change the confifugrations under `attack` and `attack_change`

3. Run the attack with the downloaded dataset
- Be sure to use the `--stage1` parameter to run the necessary ZOSO optimization. This only needs to be done once for a model and dataset
```bash
python run_attack.py --config config.yaml --dataset nq/hotpotqa/msmarco --limit 100 --data-root path/to/data_folder --output /path/to/results/output.json --add-gt --llm-judge --stage1
```

4. To run the same configurations as the project, use the `run_configs.py` script along with the `configs/baseline_configs.txt` file to run the attack on each configuration in the file.
- if ZOSO optimization has not been run for the current model, be sure to add the `--stage1` parameter to run ZOSO on the first command
```bash
python run_configs.py --dataset nq/hotpotqa/msmarco --data-root path/to/data_folder --output-dir path/to/results_folder --configs-txt configs/baseline_configs.txt
```

### RQ1 Chunking & Reranker
Next in the project, we implemented and tested InceptionRAG with a chunker, reranker, and chunker & reranker

#### Chunking
Chunking requires the wiki documents each passage in the datasets originate from. 

For our experiment, with a limited time scope, we only obtain the documents related to the queries being asked in the attack (queries can be found at `path/to/data_folder/samples/{dataset}_queries_100.json)

1. Download the Wiki corpus
- This will download a large file in your data-root and search through the documents in the file for those related to the queries.
```bash
python data/download_wiki_corpus.py --dataset nq/hotpotqa/msmarco --data-root path/to/data_folder
```

2. Configure the `configs/config.yaml` file (see Baseline step 2)
- Be sure to set the `chunk` variable to `true` for chunking to run

3. Run the attack
```bash
python run_attack.py --config config.yaml --dataset nq/hotpotqa/msmarco --limit 100 --data-root path/to/data_folder --output /path/to/results/output.json --add-gt --llm-judge
```
4. To run the same configurations as the project, use the `run_configs.py` script along with the `configs/baseline_configs.txt` file to run the attack on each configuration in the file.
```bash
python run_configs.py --dataset nq/hotpotqa/msmarco --data-root path/to/data_folder --output-dir path/to/results_folder --configs-txt configs/chunker_configs.txt
```

#### Reranker
The reranker was a relatively simple modification and does not need any prior work before running the attack.

Follow steps 2-4 from the Chunking section. (Note change the `configs-txt` parameter to `configs/reranker_configs.txt` for step 4)

#### Chunking & Reranker
Follow steps 2-4 from the Chunking section. (Note change the `configs-txt` parameter to `configs/chunker_reranker_configs.txt` for step 4)