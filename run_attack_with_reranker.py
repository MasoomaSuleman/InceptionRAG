#!/usr/bin/env python3
"""Wrapper around the main attack pipeline with reranking enabled by default.

Usage:
    python run_attack_with_reranker.py \
      --config config.yaml \
      --dataset hotpotqa \
      --limit 20 \
      --output /tmp/masoo20/inception_rag/results/hotpotqa_attack_20.json
"""

import os
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

try:
    from InceptionRAG.run_attack import main
except Exception:
    from run_attack import main


if __name__ == "__main__":
    if "--reranker" not in sys.argv and "--no-reranker" not in sys.argv:
        sys.argv.append("--reranker")
    if "--reuse-existing-stage1" not in sys.argv:
        sys.argv.append("--reuse-existing-stage1")

    if "--output" not in sys.argv:
        dataset = None
        for i, arg in enumerate(sys.argv):
            if arg == "--dataset" and i + 1 < len(sys.argv):
                dataset = sys.argv[i + 1]
                break
        if dataset:
            model_name = "default-model"
            for i, arg in enumerate(sys.argv):
                if arg == "--model" and i + 1 < len(sys.argv):
                    model_name = sys.argv[i + 1]
                    break
                elif os.path.exists("config.yaml"):
                    try:
                        import yaml
                        with open("config.yaml", "r") as f:
                            data = yaml.safe_load(f) or {}
                        model_name = (data.get("llm", {}) or {}).get("model", model_name)
                    except Exception:
                        pass
            model_dir = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name or "default-model").strip("-_.") or "default-model"
            limit = 100
            for i, arg in enumerate(sys.argv):
                if arg == "--limit" and i + 1 < len(sys.argv):
                    limit = int(sys.argv[i + 1])
                    break
            out_path = str(Path("results") / model_dir / f"{dataset}_attack_{limit}_with_reranker.json")
            sys.argv.extend(["--output", out_path])
    main()
