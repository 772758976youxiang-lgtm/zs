"""Local deterministic training, verified immutable checkpoints, and CPU inference."""

import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .model import CharacterEncoder


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _path(path):
    path = Path(path)
    return path if path.is_dir() else path.parent


def read_checkpoint(path):
    path = _path(path)
    manifest = json.loads((path / "checksums.json").read_text(encoding="utf-8"))
    if set(manifest) != {"model.safetensors", "training.pt", "metadata.json"}:
        raise ValueError("Invalid checkpoint checksum manifest")
    for name, expected in manifest.items():
        if _hash_file(path / name) != expected:
            raise ValueError("Checkpoint checksum mismatch: " + name)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    return metadata


def inspect_model(config, vocab_size):
    with torch.device("meta"):
        model = CharacterEncoder(config, vocab_size + 3)
    layers = []
    for name, module in model.named_children():
        if name == "blocks":
            layers.extend(
                {
                    "name": f"blocks.{i}",
                    "parameters": sum(p.numel() for p in block.parameters()),
                }
                for i, block in enumerate(module)
            )
        else:
            layers.append(
                {
                    "name": name,
                    "parameters": sum(p.numel() for p in module.parameters()),
                }
            )
    layers.append({"name": "mlm_bias", "parameters": model.mlm_bias.numel()})
    return {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "layers": layers,
    }


def _rng():
    return {
        "python": random.getstate(),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _records(spec):
    length = min(spec["config"]["sequence_length"], spec["model_config"]["max_length"])
    groups = {"train": [], "val": []}
    for row in spec["records"]:
        split = row.get("split", "train")
        if split not in groups:
            continue
        text = row["text"]
        if not text:
            continue
        if spec["task"] == "pretrain":
            groups[split].extend(
                {"text": text[i : i + length]} for i in range(0, len(text), length)
            )
        else:
            if len(text) > length:
                raise ValueError(
                    f"Boundary sample length {len(text)} exceeds {length}; split at annotated unit boundaries"
                )
            if "".join(row["units"]) != text or any(not u for u in row["units"]):
                raise ValueError("Invalid unit coverage")
            gaps = [0.0] * (len(text) - 1)
            cursor = 0
            for unit in row["units"][:-1]:
                cursor += len(unit)
                gaps[cursor - 1] = 1.0
            groups[split].append({"text": text, "gaps": gaps})
    if not groups["train"]:
        raise ValueError("No training records")
    if spec["task"] == "finetune" and not any(
        len(r["text"]) > 1 for r in groups["train"]
    ):
        raise ValueError("Boundary training requires at least one character gap")
    return groups


def _batch(rows, vocab, device, task, ratio):
    size = max(len(r["text"]) for r in rows)
    ids = torch.zeros((len(rows), size), dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        ids[i, : len(row["text"])] = torch.tensor(
            [vocab.get(c, 1) for c in row["text"]], device=device
        )
    if task == "pretrain":
        targets = ids.clone()
        chosen = (torch.rand(ids.shape, device=device) < ratio) & ids.ne(0)
        for i, row in enumerate(rows):
            if not chosen[i].any():
                chosen[i, torch.randint(len(row["text"]), (1,), device=device)] = True
        targets[~chosen] = -100
        replacement = torch.rand(ids.shape, device=device)
        ids[chosen & (replacement < 0.8)] = 2
        random_ids = torch.randint(3, len(vocab) + 3, ids.shape, device=device)
        ids[chosen & (replacement >= 0.8) & (replacement < 0.9)] = random_ids[
            chosen & (replacement >= 0.8) & (replacement < 0.9)
        ]
        return ids, targets
    targets = torch.full((len(rows), max(0, size - 1)), -100.0, device=device)
    for i, row in enumerate(rows):
        targets[i, : len(row["gaps"])] = torch.tensor(row["gaps"], device=device)
    return ids, targets


def _loss(logits, targets, task):
    if task == "pretrain":
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
    valid = targets.ne(-100)
    if not valid.any():
        return logits.sum() * 0
    return F.binary_cross_entropy_with_logits(logits[valid], targets[valid])


def _segments(gaps, n):
    starts = [0] + [i + 1 for i, p in enumerate(gaps) if p] + [n]
    return set(zip(starts[:-1], starts[1:]))


def _evaluate(model, rows, vocab, spec, device):
    if not rows:
        return {"loss": None}
    saved = _rng()
    model.eval()
    torch.manual_seed(spec["config"]["seed"] + 983)
    loss_sum = 0.0
    loss_targets = 0
    correct = total = tp = fp = fn = stp = sp = sg = exact = sentences = 0
    try:
        with torch.no_grad():
            for start in range(0, len(rows), spec["config"]["micro_batch"]):
                selected = rows[start : start + spec["config"]["micro_batch"]]
                ids, targets = _batch(
                    selected, vocab, device, spec["task"], spec["config"]["mask_ratio"]
                )
                logits = model(ids, spec["task"])
                valid = targets.ne(-100)
                count = int(valid.sum())
                if count:
                    loss_sum += float(_loss(logits, targets, spec["task"])) * count
                    loss_targets += count
                if spec["task"] == "pretrain":
                    correct += int((logits.argmax(-1)[valid] == targets[valid]).sum())
                    total += int(valid.sum())
                else:
                    prediction = logits.sigmoid().ge(0.5)
                    gold = targets.eq(1)
                    tp += int((prediction & gold & valid).sum())
                    fp += int((prediction & ~gold & valid).sum())
                    fn += int((~prediction & gold & valid).sum())
                    for i, row in enumerate(selected):
                        n = len(row["text"])
                        pred = prediction[i, : n - 1].tolist()
                        truth = gold[i, : n - 1].tolist()
                        ps = _segments(pred, n)
                        gs = _segments(truth, n)
                        stp += len(ps & gs)
                        sp += len(ps)
                        sg += len(gs)
                        exact += int(pred == truth)
                        sentences += 1
        result = {"loss": loss_sum / loss_targets if loss_targets else None}
        if spec["task"] == "pretrain":
            result["masked_accuracy"] = correct / max(total, 1)
        else:
            result.update(
                boundary_precision=tp / max(tp + fp, 1),
                boundary_recall=tp / max(tp + fn, 1),
                boundary_f1=2 * tp / max(2 * tp + fp + fn, 1),
                segment_precision=stp / max(sp, 1),
                segment_recall=stp / max(sg, 1),
                segment_f1=2 * stp / max(sp + sg, 1),
                sentence_accuracy=exact / max(sentences, 1),
                reconstruction_rate=1.0,
            )
        return result
    finally:
        _restore_rng(saved)
        model.train()


def run_training(spec, on_event, should_stop):
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    cfg = spec["config"]
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; explicitly select CPU tiny model")
    if device.type == "cpu" and cfg["precision"] != "fp32":
        raise ValueError("CPU training supports fp32 precision only")
    if (
        cfg["precision"] == "bf16"
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("CUDA device does not support bf16")
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    groups = _records(spec)
    vocab = {c: i + 3 for i, c in enumerate(spec["vocab"])}
    model = CharacterEncoder(spec["model_config"], len(vocab) + 3).to(device)
    model.activation_checkpointing = cfg.get("activation_checkpointing", False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, 1 - step / cfg["max_steps"])
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and cfg["precision"] == "fp16"
    )
    cursor = 0
    step = 0
    order = list(range(len(groups["train"])))
    random.shuffle(order)
    best_loss = math.inf
    best_path = None
    last_path = None
    compatibility = {
        k: spec[k]
        for k in [
            "task",
            "model_config",
            "vocab",
            "dataset_hash",
            "rule_version",
            "config",
        ]
    }
    source = spec.get("resume_path") or spec.get("parent_path")
    if source:
        meta = read_checkpoint(source)
        if spec.get("resume_path"):
            if meta["compatibility"] != compatibility:
                raise ValueError(
                    "Incompatible resume configuration, vocabulary, data or rule version"
                )
        elif (
            meta["model_config"] != spec["model_config"]
            or meta["vocab"] != spec["vocab"]
        ):
            raise ValueError("Incompatible parent model or vocabulary")
        model.load_state_dict(
            load_file(str(_path(source) / "model.safetensors"), device=str(device))
        )
        if spec.get("resume_path"):
            state = torch.load(
                _path(source) / "training.pt", map_location=device, weights_only=True
            )
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            cursor = state["cursor"]
            order = state["order"]
            step = meta["global_step"]
            best_loss = state["best_loss"]
            best_path = state["best_path"]
            last_path = str(_path(source))
            state["rng"]["cpu"] = state["rng"]["cpu"].cpu()
            state["rng"]["cuda"] = [s.cpu() for s in state["rng"]["cuda"]]
            _restore_rng(state["rng"])
    output = Path(spec["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    last_metrics = {}
    last_saved_step = -1

    def save():
        nonlocal last_path, last_saved_step, best_path, best_loss
        if step == last_saved_step:
            return last_path
        checkpoint_id = f"{spec['run_id']}-step-{step:08d}-{uuid.uuid4().hex[:8]}"
        destination = output / checkpoint_id
        temp = output / (".tmp-" + uuid.uuid4().hex)
        temp.mkdir()
        score = last_metrics.get("val_loss")
        is_best = score is not None and score < best_loss
        if is_best:
            best_loss = score
            best_path = str(destination)
        metadata = dict(
            format_version=1,
            checkpoint_id=checkpoint_id,
            run_id=spec["run_id"],
            global_step=step,
            task=spec["task"],
            model_config=spec["model_config"],
            vocab=spec["vocab"],
            vocab_hash=hashlib.sha256(_json(spec["vocab"]).encode()).hexdigest(),
            dataset_hash=spec["dataset_hash"],
            rule_version=spec["rule_version"],
            parent_path=spec.get("parent_path"),
            compatibility=compatibility,
            metrics=last_metrics,
            boundary_trained=spec["task"] == "finetune" and step > 0,
            is_best=is_best,
            device=str(device),
        )
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
            str(temp / "model.safetensors"),
        )
        torch.save(
            dict(
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),
                rng=_rng(),
                cursor=cursor,
                order=order,
                best_loss=best_loss,
                best_path=best_path,
            ),
            temp / "training.pt",
        )
        (temp / "metadata.json").write_text(_json(metadata), encoding="utf-8")
        (temp / "checksums.json").write_text(
            _json(
                {
                    name: _hash_file(temp / name)
                    for name in ["model.safetensors", "training.pt", "metadata.json"]
                }
            ),
            encoding="utf-8",
        )
        read_checkpoint(temp)
        os.replace(temp, destination)
        last_path = str(destination)
        last_saved_step = step
        on_event(dict(type="checkpoint", path=last_path, **metadata))
        return last_path

    status = "completed"
    model.train()
    # Initial durable state is a recovery point if the very first update fails.
    if last_path is None:
        save()
    while step < cfg["max_steps"]:
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        chars = 0
        hit = target_count = 0
        for _ in range(cfg["gradient_accumulation"]):
            rows = []
            for __ in range(cfg["micro_batch"]):
                if cursor >= len(order):
                    random.shuffle(order)
                    cursor = 0
                rows.append(groups["train"][order[cursor]])
                cursor += 1
            ids, targets = _batch(rows, vocab, device, spec["task"], cfg["mask_ratio"])
            chars += sum(len(r["text"]) for r in rows)
            dtype = torch.float16 if cfg["precision"] == "fp16" else torch.bfloat16
            with torch.autocast(
                device_type=device.type, dtype=dtype, enabled=cfg["precision"] != "fp32"
            ):
                logits = model(ids, spec["task"])
                loss = _loss(logits, targets, spec["task"])
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite loss; recover the last complete checkpoint"
                )
            loss_sum += float(loss.detach())
            if spec["task"] == "pretrain":
                valid = targets.ne(-100)
                hit += int((logits.argmax(-1)[valid] == targets[valid]).sum())
                target_count += int(valid.sum())
            scaler.scale(loss / cfg["gradient_accumulation"]).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        elapsed = time.perf_counter() - started
        free, total = (
            torch.cuda.mem_get_info(device) if device.type == "cuda" else (None, None)
        )
        last_metrics = dict(
            type="metrics",
            global_step=step,
            train_loss=loss_sum / cfg["gradient_accumulation"],
            val_loss=None,
            learning_rate=optimizer.param_groups[0]["lr"],
            gradient_norm=float(norm),
            chars_per_second=chars / max(time.perf_counter() - tick, 1e-9),
            elapsed_seconds=elapsed,
            masked_accuracy=hit / max(target_count, 1)
            if spec["task"] == "pretrain"
            else None,
            epoch=(step * cfg["micro_batch"] * cfg["gradient_accumulation"])
            / len(groups["train"]),
            device=str(device),
            gpu_allocated=torch.cuda.memory_allocated(device)
            if device.type == "cuda"
            else None,
            gpu_reserved=torch.cuda.memory_reserved(device)
            if device.type == "cuda"
            else None,
            gpu_device_used=total - free if total is not None else None,
            gpu_device_total=total,
            gpu_utilization=None,
        )
        if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
            for split in ("train", "val"):
                measured = _evaluate(model, groups[split], vocab, spec, device)
                for key, value in measured.items():
                    if key == "loss":
                        if split == "val":
                            last_metrics["val_loss"] = value
                        else:
                            last_metrics["train_eval_loss"] = value
                    else:
                        last_metrics[f"{split}_{key}"] = value
        on_event(last_metrics)
        command = should_stop()
        improved = (
            last_metrics["val_loss"] is not None
            and last_metrics["val_loss"] < best_loss
        )
        if (
            step % cfg["save_every"] == 0
            or step == cfg["max_steps"]
            or command
            or improved
        ):
            save()
        if command:
            status = "paused" if command == "pause" else "cancelled"
            break
    save()
    return dict(
        status=status,
        global_step=step,
        last_checkpoint=last_path,
        best_checkpoint=best_path,
        metrics=last_metrics,
    )


def _load_inference(path, text):
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    meta = read_checkpoint(path)
    if len(text) > meta["model_config"]["max_length"]:
        raise ValueError("Input length exceeds model position capacity")
    model = CharacterEncoder(meta["model_config"], len(meta["vocab"]) + 3)
    model.load_state_dict(load_file(str(_path(path) / "model.safetensors")))
    model.eval()
    vocab = {c: i + 3 for i, c in enumerate(meta["vocab"])}
    ids = torch.tensor([[vocab.get(c, 1) for c in text]], dtype=torch.long)
    return meta, model, ids


def infer_mlm(path, text, mask_positions, top_k=5):
    started = time.perf_counter()
    if (
        not text
        or not mask_positions
        or any(type(p) != int or p < 0 or p >= len(text) for p in mask_positions)
    ):
        raise ValueError("Invalid mask positions")
    if len(set(mask_positions)) != len(mask_positions):
        raise ValueError("Duplicate mask positions")
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    meta, model, ids = _load_inference(path, text)
    ids[0, mask_positions] = 2
    with torch.no_grad():
        probabilities = model(ids)[0, :, 3:].softmax(-1)
    predictions = []
    for pos in mask_positions:
        values, indices = probabilities[pos].topk(min(top_k, len(meta["vocab"])))
        predictions.append(
            dict(
                position=pos,
                candidates=[
                    dict(character=meta["vocab"][i], probability=p)
                    for i, p in zip(indices.tolist(), values.tolist())
                ],
            )
        )
    unknown = sum(c not in set(meta["vocab"]) for c in text)
    warning = (
        "随机初始化检查点，尚未进行预训练。"
        if meta["global_step"] == 0 and not meta.get("parent_path")
        else None
    )
    return dict(
        text=text,
        mask_positions=mask_positions,
        predictions=predictions,
        checkpoint_id=meta["checkpoint_id"],
        global_step=meta["global_step"],
        device="cpu",
        latency_ms=(time.perf_counter() - started) * 1000,
        offset_unit="unicode_codepoint",
        warning=warning,
        unknown_char_count=unknown,
        vocabulary_warning=f"{unknown} 个字符不在词表中，按 UNK 编码。"
        if unknown
        else None,
    )


def infer_units(path, text, threshold=0.5):
    started = time.perf_counter()
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    meta, model, ids = _load_inference(path, text)
    with torch.no_grad():
        probs = model(ids, "finetune").sigmoid()[0].tolist() if len(text) > 1 else []
    cuts = (
        [0]
        + [i + 1 for i, p in enumerate(probs) if p >= threshold]
        + ([len(text)] if text else [])
    )
    spans = list(map(list, zip(cuts[:-1], cuts[1:])))
    unknown = sum(c not in set(meta["vocab"]) for c in text)
    return dict(
        text=text,
        units=[text[a:b] for a, b in spans],
        spans=spans,
        boundary_probabilities=probs,
        offset_unit="unicode_codepoint",
        rule_version=meta["rule_version"],
        checkpoint_id=meta["checkpoint_id"],
        global_step=meta["global_step"],
        threshold=threshold,
        device="cpu",
        latency_ms=(time.perf_counter() - started) * 1000,
        warning=None
        if meta["boundary_trained"]
        else "边界头未训练；当前分界来自随机初始化的边界头。",
        unknown_char_count=unknown,
        vocabulary_warning=f"{unknown} 个字符不在词表中，按 UNK 编码；输出仍保留原文。"
        if unknown
        else None,
    )
