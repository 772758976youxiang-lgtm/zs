import pytest
from fastapi.testclient import TestClient
from backend.app.main import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path)) as c:
        token = c.get("/api/session").json()["token"]
        c.headers["X-Local-Token"] = token
        yield c


def test_local_security(client):
    assert client.get("/api/health").status_code == 200
    assert (
        client.get(
            "/api/session", headers={"Origin": "https://evil.example"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/datasets", json={}, headers={"X-Local-Token": "bad"}
        ).status_code
        == 403
    )


def test_dataset_and_config_only_model(client):
    ds = client.post(
        "/api/datasets",
        json={"name": "语料", "records": [{"text": "我养了一只猫叫魔丸"}]},
    ).json()
    assert ds["counts"]["train"] == 1
    model = client.post(
        "/api/models", json={"name": "调试", "dataset_id": ds["id"], "preset": "debug"}
    ).json()
    assert model["parameter_count"] > 10000
    assert client.get("/api/runs").json() == []
    assert (
        client.get(f"/api/datasets/{ds['id']}/records").json()["records"][0]["text"]
        == "我养了一只猫叫魔丸"
    )


def test_annotations_and_invalid_run(client):
    bad = client.post(
        "/api/datasets/validate",
        json={
            "name": "bad",
            "kind": "finetune",
            "records": [{"text": "我了", "units": ["我"]}],
        },
    )
    assert bad.status_code == 422
    assert bad.json()["error"]["code"] == "validation_error"
    assert (
        client.post(
            "/api/runs",
            json={"model_id": "missing", "dataset_id": "missing", "task": "pretrain"},
        ).status_code
        == 404
    )


def test_edit_page_preserves_unloaded_records(client):
    rows = [
        {"text": str(i), "units": [str(i)], "status": "confirmed"} for i in range(51)
    ]
    ds = client.post(
        "/api/datasets", json={"name": "完整标注", "kind": "finetune", "records": rows}
    ).json()
    first = client.get(f"/api/datasets/{ds['id']}/records").json()["records"]
    first[12]["units"] = ["1", "2"]
    first[12].pop("spans")
    first[12].pop("gap_labels")
    changed = client.post(
        "/api/datasets",
        json={
            "name": "编辑第一页",
            "kind": "finetune",
            "base_dataset_id": ds["id"],
            "records": first,
        },
    ).json()
    assert changed["count"] == 51
    assert (
        client.get(f"/api/datasets/{changed['id']}/records?offset=50").json()[
            "records"
        ][0]["text"]
        == "50"
    )
