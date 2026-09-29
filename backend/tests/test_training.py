from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from backend.training import engine


def spec(tmp_path, name="run"):
    return dict(
        run_id=name,
        task="pretrain",
        config=dict(
            learning_rate=0.01,
            micro_batch=2,
            gradient_accumulation=2,
            max_steps=6,
            save_every=3,
            eval_every=2,
            mask_ratio=0.4,
            precision="fp32",
            gradient_clip=1.0,
            activation_checkpointing=False,
            device="cpu",
            seed=7,
            sequence_length=16,
        ),
        model_config=dict(
            hidden_size=16,
            num_layers=1,
            num_heads=2,
            intermediate_size=32,
            max_length=16,
            dropout=0.1,
        ),
        records=[{"text": "我养了一只猫叫魔丸", "split": "train"}] * 3,
        vocab=list("我养了一只猫叫魔丸"),
        output_dir=str(tmp_path / name),
        dataset_hash="test-data",
        rule_version="unit-v1",
    )


def test_real_training_and_exact_resume(tmp_path):
    full = spec(tmp_path, "full")
    events = []
    result = engine.run_training(full, events.append, lambda: None)
    assert result["status"] == "completed"
    metrics = [e for e in events if e["type"] == "metrics"]
    assert len(metrics) == 6 and metrics[0]["train_loss"] != metrics[-1]["train_loss"]
    assert all(e["val_loss"] is None for e in metrics)
    assert all(e["gradient_norm"] > 0 and e["chars_per_second"] > 0 for e in metrics)
    paused = spec(tmp_path, "paused")
    ticks = []
    first = engine.run_training(
        paused,
        lambda e: ticks.append(e),
        lambda: "pause"
        if len([x for x in ticks if x["type"] == "metrics"]) >= 3
        else None,
    )
    assert first["status"] == "paused" and first["global_step"] == 3
    paused["resume_path"] = first["last_checkpoint"]
    second = engine.run_training(paused, lambda e: None, lambda: None)
    a = load_file(str(Path(result["last_checkpoint"]) / "model.safetensors"))
    b = load_file(str(Path(second["last_checkpoint"]) / "model.safetensors"))
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    metadata = engine.read_checkpoint(second["last_checkpoint"])
    assert metadata["global_step"] == 6
    paused["dataset_hash"] = "different"
    with pytest.raises(ValueError, match="compatib"):
        engine.run_training(paused, lambda e: None, lambda: None)


def test_inference_coverage_and_checkpoint_integrity(tmp_path):
    config = spec(tmp_path)
    config["config"]["max_steps"] = 1
    result = engine.run_training(config, lambda e: None, lambda: None)
    path = result["last_checkpoint"]
    text = "我🙂e\u0301很很， "
    response = engine.infer_units(path, text)
    assert "".join(response["units"]) == text
    assert response["offset_unit"] == "unicode_codepoint" and response["warning"]
    assert all(
        text[a:b] == u for (a, b), u in zip(response["spans"], response["units"])
    )
    assert len(response["boundary_probabilities"]) == len(text) - 1
    mlm = engine.infer_mlm(path, "我猫", [1], 3)
    assert len(mlm["predictions"][0]["candidates"]) == 3
    assert mlm["device"] == "cpu"
    with pytest.raises(ValueError):
        engine.infer_units(path, "猫" * 17)
    with pytest.raises(ValueError):
        engine.infer_mlm(path, "猫", [1])
    weights = Path(path) / "model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="checksum"):
        engine.read_checkpoint(path)


def test_finetune_metrics_and_model_count(tmp_path):
    p = spec(tmp_path)
    p["config"]["max_steps"] = 1
    base = engine.run_training(p, lambda e: None, lambda: None)
    f = spec(tmp_path, "ft")
    f["task"] = "finetune"
    f["parent_path"] = base["last_checkpoint"]
    f["records"] = [
        dict(
            text="我养了一只猫叫魔丸", units=list("我养了一只猫叫") + ["魔丸"], split=s
        )
        for s in ["train", "val"]
    ]
    events = []
    result = engine.run_training(f, events.append, lambda: None)
    metrics = [e for e in events if e["type"] == "metrics"]
    assert metrics[-1]["val_loss"] is not None
    assert "val_boundary_f1" in metrics[-1] and "train_segment_f1" in metrics[-1]
    assert metrics[-1]["train_reconstruction_rate"] == 1
    assert engine.infer_units(result["last_checkpoint"], "我猫")["warning"] is None
    inspection = engine.inspect_model(f["model_config"], 12)
    assert inspection["parameter_count"] > 1000 and inspection["layers"]
    f["records"] = [dict(text="猫" * 17, units=["猫" * 17], split="train")]
    with pytest.raises(ValueError, match="length"):
        engine.run_training(f, lambda e: None, lambda: None)


def test_initial_checkpoint_warnings_and_layer_inspection(tmp_path):
    p = spec(tmp_path)
    p["config"]["max_steps"] = 1
    events = []
    engine.run_training(p, events.append, lambda: None)
    initial = next(
        e["path"] for e in events if e["type"] == "checkpoint" and e["global_step"] == 0
    )
    assert engine.infer_mlm(initial, "我", [0])["warning"]
    f = spec(tmp_path, "branch")
    f["task"] = "finetune"
    f["config"]["max_steps"] = 1
    f["parent_path"] = initial
    f["records"] = [dict(text="我猫", units=["我", "猫"], split="train")]
    events = []
    engine.run_training(f, events.append, lambda: None)
    initial = next(
        e["path"] for e in events if e["type"] == "checkpoint" and e["global_step"] == 0
    )
    assert engine.read_checkpoint(initial)["boundary_trained"] is False
    assert engine.infer_units(initial, "我🙂")["unknown_char_count"] == 1
    inspection = engine.inspect_model(
        dict(p["model_config"], num_layers=2), len(p["vocab"])
    )
    assert any(layer["name"] == "blocks.1" for layer in inspection["layers"])


def test_best_validation_checkpoint_saved_between_intervals(tmp_path):
    p = spec(tmp_path)
    p["config"].update(max_steps=3, save_every=3, eval_every=2)
    p["records"].append({"text": "猫叫我", "split": "val"})
    events = []
    result = engine.run_training(p, events.append, lambda: None)
    assert any(
        e["type"] == "checkpoint" and e["global_step"] == 2 and e["is_best"]
        for e in events
    )
    assert result["best_checkpoint"]


def test_validation_loss_weights_real_gaps_independent_of_batch_size(tmp_path):
    from backend.training.model import CharacterEncoder

    p = spec(tmp_path)
    p["task"] = "finetune"
    vocab = {c: i + 3 for i, c in enumerate(p["vocab"])}
    torch.manual_seed(9)
    model = CharacterEncoder(p["model_config"], len(vocab) + 3)
    rows = [
        {"text": "我", "gaps": []},
        {"text": "我猫", "gaps": [1.0]},
        {"text": "我养了一只猫", "gaps": [0.0, 0.0, 0.0, 0.0, 0.0]},
    ]
    p["config"]["micro_batch"] = 1
    single = engine._evaluate(model, rows, vocab, p, torch.device("cpu"))["loss"]
    p["config"]["micro_batch"] = 3
    together = engine._evaluate(model, rows, vocab, p, torch.device("cpu"))["loss"]
    assert single == pytest.approx(together, abs=1e-6)


def test_initial_tied_mlm_loss_has_reasonable_scale(tmp_path):
    import math
    from backend.training.model import CharacterEncoder

    config = dict(
        spec(tmp_path)["model_config"],
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
    )
    torch.manual_seed(7)
    model = CharacterEncoder(config, 23).eval()
    ids = torch.full((4, 12), 2, dtype=torch.long)
    targets = torch.arange(48).reshape(4, 12) % 20 + 3
    with torch.no_grad():
        loss = engine._loss(model(ids), targets, "pretrain")
    assert torch.isfinite(loss)
    assert abs(float(loss) - math.log(23)) < 1.0
    assert torch.count_nonzero(model.embedding.weight[0]) == 0
