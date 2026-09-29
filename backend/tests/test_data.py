import pytest
from backend.app.data import validate_units, prepare_records, build_vocab


def test_gold_and_unicode():
    text = "我养了十二只猫，它们都很乖很可爱"
    units = [
        "我",
        "养",
        "了",
        "十二",
        "只",
        "猫",
        "，",
        "它",
        "们",
        "都",
        "很",
        "乖",
        "很",
        "可爱",
    ]
    result = validate_units(text, units)
    assert result["spans"][-1] == [14, 16]
    assert len(result["gap_labels"]) == len(text) - 1
    assert validate_units("😀 e\u0301\n", ["😀", " ", "e\u0301", "\n"])["spans"][
        -1
    ] == [4, 5]
    with pytest.raises(ValueError):
        validate_units("了猫", ["猫"])


def test_conflict_and_leakage():
    with pytest.raises(ValueError, match="冲突"):
        prepare_records(
            [
                {"text": "十二", "units": ["十二"], "status": "confirmed"},
                {"text": "十二", "units": ["十", "二"], "status": "confirmed"},
            ],
            "finetune",
            "unit-v1",
        )
    with pytest.raises(ValueError, match="泄漏"):
        prepare_records(
            [
                {"text": "猫", "split": "train", "group_id": "a"},
                {"text": "狗", "split": "val", "group_id": "a"},
            ],
            "pretrain",
            "unit-v1",
        )


def test_train_only_vocab_and_preserve_whitespace():
    records = prepare_records(
        [{"text": " 我\n", "split": "train"}, {"text": "陌", "split": "val"}],
        "pretrain",
        "unit-v1",
    )
    assert records[0]["text"] == " 我\n"
    assert "陌" not in build_vocab(records)
    assert " " in build_vocab(records)


@pytest.mark.parametrize(
    "first,second",
    [("abcdefghijklmnop", "xbcdefghijklmnop"), ("abcdefghijklmno", "abcdefghijklmnop")],
)
def test_near_duplicates_across_prefix_or_length_boundary(first, second):
    with pytest.raises(ValueError, match="泄漏"):
        prepare_records(
            [{"text": first, "split": "train"}, {"text": second, "split": "val"}],
            "pretrain",
            "unit-v1",
        )


def test_source_group_leakage():
    with pytest.raises(ValueError, match="泄漏"):
        prepare_records(
            [
                {"text": "甲", "source": "document1", "split": "train"},
                {"text": "乙", "source": "document1", "split": "val"},
            ],
            "pretrain",
            "unit-v1",
        )


def test_model_correction_can_be_saved_as_conflicting_draft():
    rows = prepare_records(
        [
            {"text": "十二", "units": ["十二"], "status": "confirmed"},
            {"text": "十二", "units": ["十", "二"], "status": "draft"},
        ],
        "finetune",
        "unit-v1",
    )
    assert len(rows) == 2 and all(r["conflict"] for r in rows)
    assert len([r for r in rows if r["status"] == "confirmed"]) == 1
    for row in rows:
        row["status"] = "confirmed"
    with pytest.raises(ValueError, match="冲突"):
        prepare_records(rows, "finetune", "unit-v1")
