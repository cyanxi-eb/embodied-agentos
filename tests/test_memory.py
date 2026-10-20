"""Qdrant local memory, hashing embeddings, memory reuse rules, SQLite session store."""

from __future__ import annotations

import pytest

from agentos.memory.session_store import SessionStore
from agentos.memory.vector_store import (
    InMemoryMemoryStore,
    QdrantMemoryStore,
    build_memory_store,
    cosine,
    embed,
)


def test_embedding_is_deterministic_and_normalised():
    a = embed("object red box")
    b = embed("object red box")
    assert a == b
    assert len(a) == 64
    assert pytest.approx(sum(value * value for value in a), abs=1e-6) == 1.0
    assert cosine(a, b) == pytest.approx(1.0, abs=1e-6)


def test_embedding_similarity_prefers_same_object():
    red_box = embed("object red box")
    red_cup = embed("object red cup")
    green_ball = embed("object green ball")
    assert cosine(red_box, red_cup) > cosine(red_box, green_ball)


def test_in_memory_store_search_and_filter():
    store = InMemoryMemoryStore()
    store.remember("scene", "warehouse scene with shelves", scene="warehouse")
    store.remember("task", "fetch red box", scene="warehouse")
    store.remember("task", "dock the robot", scene="lab")
    assert store.count() == 3
    hits = store.search("fetch red box", scene="warehouse", kind="task")
    assert len(hits) == 1
    assert hits[0].data is not None
    lab_hits = store.search("dock the robot", scene="lab")
    assert lab_hits and lab_hits[0].scene == "lab"


def test_last_object_location_requires_exact_object_match():
    store = InMemoryMemoryStore()
    store.remember_object("warehouse", {"color": "blue", "category": "cup"}, "obj_blue_cup_1", 5.0, 8.0, "zone_b")
    assert store.last_object_location("warehouse", {"color": "blue", "category": "cup"}) is not None
    assert store.last_object_location("warehouse", {"color": "red", "category": "box"}) is None
    assert store.last_object_location("warehouse", {"color": "blue", "category": "box"}) is None


@pytest.mark.parametrize("factory", ["qdrant", "memory"])
def test_memory_store_backends(tmp_path, factory):
    path = tmp_path / "qdrant" if factory == "qdrant" else None
    store = build_memory_store(path, backend=factory)
    try:
        store.remember_object("warehouse", {"color": "red", "category": "box"}, "obj_red_box_0", 4.2, 4.8, "zone_a")
        hit = store.last_object_location("warehouse", {"color": "red", "category": "box"})
        assert hit is not None
        assert hit.data["x"] == pytest.approx(4.2)
        assert hit.data["zone"] == "zone_a"
        assert store.count() >= 1
    finally:
        store.close()


def test_qdrant_local_mode_persists_between_clients(tmp_path):
    path = tmp_path / "qdrant-persist"
    store = QdrantMemoryStore(path)
    store.remember_object("warehouse", {"color": "red", "category": "box"}, "obj_red_box_0", 1.0, 2.0, "zone_a")
    store.close()
    reopened = QdrantMemoryStore(path)
    try:
        hit = reopened.last_object_location("warehouse", {"color": "red", "category": "box"})
        assert hit is not None
        assert hit.data["y"] == pytest.approx(2.0)
    finally:
        reopened.close()


def test_session_store_roundtrip(tmp_path):
    store = SessionStore(tmp_path / "sessions.sqlite")
    store.create_session("s1", "把红色盒子拿来", "warehouse")
    store.update_session("s1", "finished", {"goal_verified": True})
    session = store.get_session("s1")
    assert session["status"] == "finished"
    assert session["result"]["goal_verified"] is True
    store.append_event("s1", 1, "plan", {"nodes": 3})
    store.append_event("s1", 2, "finish", {"status": "finished"})
    events = store.get_events("s1")
    assert [event["type"] for event in events] == ["plan", "finish"]
    assert store.list_sessions()[0]["session_id"] == "s1"
    store.close()
