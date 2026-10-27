from pathlib import Path

from InceptionRAG.run_attack import _resolve_model_output_path, _append_output_suffix, load_attack_config_from_yaml


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
