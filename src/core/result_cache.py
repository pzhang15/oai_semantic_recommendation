from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple


class _ResultCache:
    def __init__(self, max_size: int, ttl_secs: int) -> None:
        self.max_size = max_size
        self.ttl_secs = ttl_secs
        self._store: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

    def put(self, items: List[Dict[str, Any]]) -> str:
        # Evict if over capacity (simple FIFO over insertion order)
        if len(self._store) >= self.max_size:
            self._store.pop(next(iter(self._store)))
        rid = uuid.uuid4().hex
        self._store[rid] = (time.time(), items)
        return rid

    def put_with_id(self, rid: str, items: List[Dict[str, Any]]) -> str:
        if len(self._store) >= self.max_size:
            self._store.pop(next(iter(self._store)))
        self._store[rid] = (time.time(), items)
        return rid
    def get(self, rid: str) -> Optional[List[Dict[str, Any]]]:
        if not rid:
            return None
        val = self._store.get(rid)
        if not val:
            return None
        ts, items = val
        if (time.time() - ts) > self.ttl_secs:
            # Expired
            self._store.pop(rid, None)
            return None
        return items


_CACHE: _ResultCache | None = None


def configure(max_size: int, ttl_secs: int) -> None:
    global _CACHE
    _CACHE = _ResultCache(max_size=max_size, ttl_secs=ttl_secs)


def put(items: List[Dict[str, Any]]) -> str:
    if _CACHE is None:
        return ""
    return _CACHE.put(items)


def get(rid: str) -> Optional[List[Dict[str, Any]]]:
    if _CACHE is None:
        return None
    return _CACHE.get(rid)


def put_with_id(rid: str, items: List[Dict[str, Any]]) -> str:
    if _CACHE is None:
        return rid
    return _CACHE.put_with_id(rid, items)


