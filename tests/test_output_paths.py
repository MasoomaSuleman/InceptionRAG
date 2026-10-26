from pathlib import Path

from InceptionRAG.run_attack import _resolve_model_output_path, _append_output_suffix


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
