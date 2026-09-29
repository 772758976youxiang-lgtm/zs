"""Lossless codepoint annotation and immutable dataset preparation."""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def validate_units(text, units):
    if (
        not isinstance(text, str)
        or not isinstance(units, list)
        or any(not isinstance(u, str) or not u for u in units)
    ):
        raise ValueError("原文必须为字符串，单位必须为非空字符串数组")
    if "".join(units) != text:
        raise ValueError("单位必须按顺序完整还原原文，不得删改、去重或裁剪空白")
    spans, gaps, pos = [], [0] * max(0, len(text) - 1), 0
    for unit in units:
        spans.append([pos, pos + len(unit)])
        pos += len(unit)
        if pos < len(text):
            gaps[pos - 1] = 1
    return {
        "text": text,
        "units": units,
        "spans": spans,
        "gap_labels": gaps,
        "offset_unit": "unicode_codepoint",
    }


def prepare_records(records, kind, rule_version):
    result, seen, groups = [], {}, {}
    # Signature buckets flag obvious near-duplicates without quadratic corpus scans.
    buckets = {}
    for raw in records:
        for field in ("id", "source", "group_id", "status"):
            if (
                field in raw
                and raw[field] is not None
                and (not isinstance(raw[field], str) or len(raw[field]) > 512)
            ):
                raise ValueError(f"{field} 必须为不超过 512 字符的字符串")
        text = raw.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("训练记录 text 必须为非空字符串；正文空白不会被裁剪")
        split = raw.get("split", "train")
        if split not in ("train", "val", "test"):
            raise ValueError("split 必须为 train / val / test")
        record = dict(
            raw,
            id=raw.get("id") or uuid.uuid4().hex,
            text=text,
            split=split,
            rule_version=rule_version,
            created_at=raw.get("created_at") or now(),
            updated_at=now(),
        )
        record["source"] = raw.get("source", "manual")
        record["group_id"] = raw.get("group_id") or (
            record["source"]
            if record["source"] not in ("manual", "user_confirmed", "model_correction")
            else digest(text)
        )
        record["status"] = raw.get(
            "status", "draft" if kind == "finetune" else "confirmed"
        )
        if record["status"] not in ("draft", "confirmed"):
            raise ValueError("status 必须为 draft / confirmed")
        if kind == "finetune":
            checked = validate_units(text, raw.get("units"))
            for field in ("spans", "gap_labels"):
                if field in raw and raw[field] != checked[field]:
                    raise ValueError(f"{field} 与原文/单位不一致")
            record.update(checked)
        record.pop("conflict", None)
        duplicate = False
        for other in seen.get(text, []):
            if other["split"] != split:
                raise ValueError("精确重复样本跨集合泄漏")
            if other.get("units") != record.get("units"):
                if other["status"] == record["status"] == "confirmed":
                    raise ValueError(
                        "相同原文、同一规范版本的已确认标注冲突，请先解决边界分歧"
                    )
                other["conflict"] = record["conflict"] = True
            elif other["status"] == record["status"]:
                duplicate = True
        if duplicate:
            continue
        group = record["group_id"]
        if group in groups and groups[group] != split:
            raise ValueError("同一文档/模板分组跨集合泄漏")
        groups[group] = split
        grams = (
            {text[i : i + 3] for i in range(len(text) - 2)} if len(text) >= 8 else set()
        )
        candidates = set()
        for gram in grams:
            candidates.update(buckets.get(gram, set()))
        for index in candidates:
            other = result[index]
            if (
                other["split"] != split
                and 2
                * min(len(text), len(other["text"]))
                / (len(text) + len(other["text"]))
                >= 0.9
                and SequenceMatcher(None, text, other["text"], autojunk=False).ratio()
                >= 0.90
            ):
                raise ValueError(
                    "发现明显近重复跨集合泄漏，请将同句改写放入相同 group_id / split"
                )
        for gram in grams:
            buckets.setdefault(gram, set()).add(len(result))
        seen.setdefault(text, []).append(record)
        result.append(record)
    if not result:
        raise ValueError("数据集不能为空")
    return result


def build_vocab(records):
    return sorted(
        {
            char
            for record in records
            if record["split"] == "train" and record["status"] == "confirmed"
            for char in record["text"]
        }
    )
