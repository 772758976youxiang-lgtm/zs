import pytest
from backend.app.store import Store, Conflict


def test_single_worker_claim_and_event_replay(tmp_path):
    store = Store(tmp_path)
    store.create_run({"id": "a", "status": "queued"})
    with pytest.raises(Conflict):
        store.create_run({"id": "b", "status": "queued"})
    store.event("a", {"type": "metrics", "global_step": 1})
    store.event("a", {"type": "metrics", "global_step": 2})
    assert len(store.events("a", 1)) == 1
    store.update_run("a", status="paused")
    store.create_run({"id": "b", "status": "queued"})
    with pytest.raises(Conflict):
        store.claim_resume("a")


def test_dead_worker_reconciled(tmp_path):
    store = Store(tmp_path)
    store.create_run(
        {"id": "a", "status": "running", "pid": 999999999, "pid_created": 0}
    )
    store.reconcile()
    assert store.get("runs", "a")["status"] == "interrupted"


def test_stale_worker_cannot_change_new_generation(tmp_path):
    store = Store(tmp_path)
    store.create_run({"id": "a", "status": "queued", "generation": "new"})
    with pytest.raises(Conflict):
        store.update_run("a", expected={"generation": "old"}, status="failed")
    assert store.get("runs", "a")["status"] == "queued"


def test_stop_during_pause_save_wins(tmp_path):
    store = Store(tmp_path)
    store.create_run(
        {"id": "a", "status": "stopping", "generation": "worker", "control": "stop"}
    )
    result = store.finish_worker("a", "worker", "paused", {})
    assert result["status"] == "cancelled" and result["control"] is None
