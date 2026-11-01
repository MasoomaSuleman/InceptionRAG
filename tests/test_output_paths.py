from pathlib import Path

from InceptionRAG.run_attack import (
    _candidate_stage_artifact_paths,
    _resolve_model_output_path,
    _append_output_suffix,
    build_chunked_adversarial_corpus,
    chunk_text_by_word_count,
    load_attack_config_from_yaml,
)


def test_load_attack_config_from_yaml_respects_rerank_top_k(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
llm:
  provider: "openai"
  model: "gpt-4o-mini"
attack:
  retrieval_top_k: 12
  retrieval_first_k: 25
  cttft_neighbor_k: 8
  noise_injection: false
  dual_injection: true
  rerank_top_k: 6
""".strip()
    )

    result = load_attack_config_from_yaml(str(config_path))

    assert result["retrieval_top_k"] == 12
    assert result["retrieval_first_k"] == 25
    assert result["cttft_neighbor_k"] == 8
    assert result["noise_injection"] is False
    assert result["dual_injection"] is True
    assert result["rerank_top_k"] == 6


def test_output_path_uses_model_folder():
    result = _resolve_model_output_path(
        "/tmp/user/results/hotpotqa_attack_100.json",
        "Qwen/Qwen2.5-7B-Instruct",
    )
    assert result == Path("/tmp/user/results/Qwen-Qwen2-5-7B-Instruct/hotpotqa_attack_100.json")


def test_output_suffix_keeps_model_folder():
    result = _append_output_suffix(
        "/tmp/user/results/hotpotqa_attack_100.json",
        "_with_ad",
        "Qwen/Qwen2.5-7B-Instruct",
    )
    assert result == Path("/tmp/user/results/Qwen-Qwen2-5-7B-Instruct/hotpotqa_attack_100_with_ad.json")


def test_reranker_output_uses_base_name_for_stage_artifacts():
    result = _append_output_suffix(
        "/tmp/masoo20/inception_rag/results/Qwen-Qwen2.5-3B-Instruct/hotpotqa_attack_100_with_reranker.json",
        "_with_ad",
        "Qwen/Qwen2.5-3B-Instruct",
    )
    assert result == Path("/tmp/masoo20/inception_rag/results/Qwen-Qwen2-5-3B-Instruct/hotpotqa_attack_100_with_ad.json")


def test_stage_artifact_candidates_include_model_and_base_paths():
    candidates = _candidate_stage_artifact_paths(
        "/tmp/masoo20/inception_rag/results/Qwen-Qwen2.5-3B-Instruct/hotpotqa_attack_100_with_reranker.json",
        "Qwen/Qwen2.5-3B-Instruct",
        "_with_ad",
    )

    assert candidates[0] == Path(
        "/tmp/masoo20/inception_rag/results/Qwen-Qwen2-5-3B-Instruct/hotpotqa_attack_100_with_ad.json"
    )
    assert Path(
        "/tmp/masoo20/inception_rag/results/hotpotqa_attack_100_with_ad.json"
    ) in candidates


def test_chunk_text_by_word_count_splits_long_text():
    text = " ".join([f"word{i}" for i in range(250)])
    chunks = chunk_text_by_word_count(text, max_words=200)

    assert len(chunks) >= 2
    assert all(len(chunk.split()) <= 200 for chunk in chunks)
    assert " ".join(chunks[0].split()[:5]) == "word0 word1 word2 word3 word4"


def test_build_chunked_adversarial_corpus_mixes_wiki_and_adv_docs(tmp_path):
    items = [
        {
            "id": "item-1",
            "question": "What is the capital of France?",
            "ad_text": {
                "attribute_document": "Paris is the capital of France.",
                "bridge_document": "Paris sits in Europe.",
            },
            "incorrect answer": "Berlin",
            "expanded answer": ["Berlin"],
        }
    ]

    corpus = ["wikipedia paragraph about Europe"]
    merged = build_chunked_adversarial_corpus(
        items,
        corpus,
        anchor_suffix="Anchor suffix",
        bridge_suffix="Bridge suffix",
        chunk_size_words=20,
        verification_path=tmp_path / "adv_docs.txt",
    )

    assert len(merged) > len(corpus)
    assert any("Anchor suffix" in doc for doc in merged)
    assert any("Bridge suffix" in doc for doc in merged)
    assert (tmp_path / "adv_docs.txt").exists()
    assert "Paris is the capital of France" in (tmp_path / "adv_docs.txt").read_text()
