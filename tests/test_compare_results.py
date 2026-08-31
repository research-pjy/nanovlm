import json
import sys

from experiments.encoder_ambiguity.compare_results import main


def _dummy_result(size, strategy, total, encoder_params):
    return {
        "size": size,
        "patch_embed_strategy": strategy,
        "checkpoint": f"checkpoints/nanovlm_{size}_{strategy}/final.pt",
        "num_held_out": 100,
        "num_judge_parse_failures": 2,
        "param_breakdown": {
            "visual_encoder": encoder_params,
            "multimodal_projector": 1000,
            "decoder": 50000,
            "total": encoder_params + 51000,
        },
        "avg_judge_scores": {
            "grammar": 7.5,
            "creativity": 6.2,
            "consistency": 7.8,
            "meaningfulness": 7.0,
            "plot": 6.5,
        },
        "average_total": total,
        "avg_rouge1_f1": 0.31,
        "per_image": [],
    }


def test_compare_two_results_runs_and_prints_summary(tmp_path, monkeypatch, capsys):
    r1 = tmp_path / "nanovlm_base_conv_on_patches_eval.json"
    r2 = tmp_path / "nanovlm_base_conv_on_image_eval.json"
    r1.write_text(json.dumps(_dummy_result("base", "conv_on_patches", 35.0, 40000)))
    r2.write_text(json.dumps(_dummy_result("base", "conv_on_image", 37.5, 60000)))

    monkeypatch.setattr(
        sys, "argv", ["compare_results.py", str(r1), str(r2), "--labels", "conv_on_patches", "conv_on_image"]
    )
    main()

    out = capsys.readouterr().out
    assert "Judge scores" in out
    assert "conv_on_patches" in out
    assert "conv_on_image" in out
    assert "Highest average_total: conv_on_image" in out
    assert "ROUGE-1" in out
