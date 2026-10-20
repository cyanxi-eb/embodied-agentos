"""Deterministic hashing embedding + Qdrant local-mode memory.

Why a hashing embedding instead of a downloaded model?  The development host has
no GPU and the evaluation must be reproducible offline.  A hashing embedding is a
real (if simple) bag-of-features vectoriser: deterministic, dependency-free, and
good enough to retrieve "where was the red box last time" by colour/category.

Qdrant runs in **local mode** (``QdrantClient(path=...)``), which persists to a
git-ignored ``.cache/qdrant`` directory.  If Qdrant is unavailable the store
degrades to an in-memory cosine index instead of faking a successful write.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

EMBEDDING_DIM = 64
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def embed(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Hashing embedding: token/bigram counts hashed into a fixed-size vector."""
    tokens = _TOKEN_RE.findall((text or "").lower())
    vector = [0.0] * dim
    if not tokens:
        return vector
    features = list(tokens)
    features += [tokens[i] + tokens[i + 1] for i in range(len(tokens) - 1)]
    for feature in features:
        digest = hashlib.md5(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(a: Iterable[float], b: Iterable[float]) -> float:
    av, bv = list(a), list(b)
    dot = sum(x * y for x, y in zip(av, bv))
    na = math.sqrt(sum(x * x for x in av)) or 1.0
    nb = math.sqrt(sum(y * y for y in bv)) or 1.0
    return dot / (na * nb)


@dataclass
class MemoryHit:
    memory_id: str
    score: float
    kind: str
    scene: str
    text: str
    data: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "score": round(self.score, 4),
            "kind": self.kind,
            "scene": self.scene,
            "text": self.text,
            "data": self.data,
            "created_at": self.created_at,
        }


class MemoryStore:
    """Interface used by the planner/graph; both backends implement it."""

    backend_name = "base"

    def remember(self, kind: str, text: str, scene: str = "", data: dict[str, Any] | None = None) -> str:
        raise NotImplementedError

    def search(
        self, query: str, scene: str | None = None, kind: str | None = None, limit: int = 3
    ) -> list[MemoryHit]:
        raise NotImplementedError

    def remember_object(
        self, scene: str, query: dict[str, Any], object_id: str, x: float, y: float, zone: str, session_id: str = ""
    ) -> str:
        query_text = " ".join(str(query.get(key, "")) for key in ("color", "category") if query.get(key))
        return self.remember(
            "object",
            f"object {query_text}",
            scene=scene,
            data={
                "query": query,
                "object_id": object_id,
                "x": round(x, 3),
                "y": round(y, 3),
                "zone": zone,
                "session_id": session_id,
            },
        )

    def last_object_location(self, scene: str, query: dict[str, Any], limit: int = 3) -> MemoryHit | None:
        query_text = " ".join(str(query.get(key, "")) for key in ("color", "category") if query.get(key))
        if not query_text:
            return None
        hits = self.search(f"object {query_text}", scene=scene, kind="object", limit=limit)
        for hit in hits:
            stored_query = hit.data.get("query") or {}
            # exact colour/category match: a similar-looking memory must not be
            # reused for a different object (that would be a silent wrong plan)
            if stored_query.get("color") != query.get("color"):
                continue
            if stored_query.get("category") != query.get("category"):
                continue
            if hit.score < 0.5:
                continue
            if hit.data.get("x") is not None and hit.data.get("y") is not None:
                return hit
        return None

    def count(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        return None


class InMemoryMemoryStore(MemoryStore):
    """Fallback vector store (cosine over the same hashing embeddings)."""

    backend_name = "in-memory"

    def __init__(self) -> None:
        self._items: list[MemoryHit] = []
        self._vectors: list[list[float]] = []
        self._seq = 0

    def remember(self, kind: str, text: str, scene: str = "", data: dict[str, Any] | None = None) -> str:
        self._seq += 1
        memory_id = f"mem-{self._seq}"
        self._items.append(
            MemoryHit(memory_id, 1.0, kind, scene, text, dict(data or {}), time.time())
        )
        self._vectors.append(embed(text))
        return memory_id

    def search(
        self, query: str, scene: str | None = None, kind: str | None = None, limit: int = 3
    ) -> list[MemoryHit]:
        vector = embed(query)
        scored: list[MemoryHit] = []
        for item, item_vector in zip(self._items, self._vectors):
            if scene and item.scene and item.scene != scene:
                continue
            if kind and item.kind != kind:
                continue
            score = cosine(vector, item_vector)
            scored.append(
                MemoryHit(item.memory_id, score, item.kind, item.scene, item.text, item.data, item.created_at)
            )
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:limit]

    def count(self) -> int:
        return len(self._items)


class QdrantMemoryStore(MemoryStore):
    """Qdrant local-mode store; falls back to in-memory on import/lock errors."""

    backend_name = "qdrant"

    def __init__(self, path: str | Path, collection: str = "agentos_memory", dim: int = EMBEDDING_DIM) -> None:
        self.path = str(path)
        self.collection = collection
        self.dim = dim
        self._fallback: InMemoryMemoryStore | None = None
        self._client = None
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            Path(self.path).mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=self.path)
            if not self._client.collection_exists(collection):
                self._client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
        except Exception as error:  # pragma: no cover - depends on local qdrant state
            self._client = None
            self._fallback = InMemoryMemoryStore()
            self.backend_name = "in-memory(fallback)"
            self.fallback_reason = repr(error)
            return
        self.fallback_reason = ""

    # ------------------------------------------------------------------ write
    def remember(self, kind: str, text: str, scene: str = "", data: dict[str, Any] | None = None) -> str:
        if self._client is None and self._fallback is not None:
            return self._fallback.remember(kind, text, scene, data)
        from qdrant_client.models import PointStruct

        point_id = (zlib.crc32(f"{time.time_ns()}|{kind}|{text}".encode("utf-8")) & 0x7FFFFFFF) or 1
        payload = {
            "kind": kind,
            "text": text,
            "scene": scene,
            "data": dict(data or {}),
            "created_at": time.time(),
        }
        self._client.upsert(  # type: ignore[union-attr]
            self.collection,
            points=[PointStruct(id=point_id, vector=embed(text, self.dim), payload=payload)],
        )
        return str(point_id)

    # ------------------------------------------------------------------- read
    def search(
        self, query: str, scene: str | None = None, kind: str | None = None, limit: int = 3
    ) -> list[MemoryHit]:
        if self._client is None and self._fallback is not None:
            return self._fallback.search(query, scene, kind, limit)
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        conditions = []
        if scene:
            conditions.append(FieldCondition(key="scene", match=MatchValue(value=scene)))
        if kind:
            conditions.append(FieldCondition(key="kind", match=MatchValue(value=kind)))
        query_filter = Filter(must=conditions) if conditions else None
        response = self._client.query_points(  # type: ignore[union-attr]
            self.collection,
            query=embed(query, self.dim),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        hits: list[MemoryHit] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                MemoryHit(
                    memory_id=str(point.id),
                    score=float(point.score),
                    kind=str(payload.get("kind", "")),
                    scene=str(payload.get("scene", "")),
                    text=str(payload.get("text", "")),
                    data=dict(payload.get("data") or {}),
                    created_at=float(payload.get("created_at", 0.0)),
                )
            )
        return hits

    def count(self) -> int:
        if self._client is None and self._fallback is not None:
            return self._fallback.count()
        try:
            return int(self._client.count(self.collection).count)  # type: ignore[union-attr]
        except Exception:  # pragma: no cover
            return -1

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover
                pass
            self._client = None


def build_memory_store(path: str | Path | None, backend: str = "qdrant") -> MemoryStore:
    """Factory used by the runtime / API / evaluation."""
    if path is None or backend == "memory":
        return InMemoryMemoryStore()
    return QdrantMemoryStore(path)
