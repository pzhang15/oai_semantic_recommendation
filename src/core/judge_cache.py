from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


class _LRUTTLCache:
    def __init__(self, max_size: int, ttl_secs: int) -> None:
        self._max_size = int(max_size)
        self._ttl = int(ttl_secs)
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, Tuple[Any, float]]" = OrderedDict()

    def get(self, key: str) -> Optional[Tuple[Any, int]]:
        now = time.time()
        with self._lock:
            val = self._data.get(key)
            if not val:
                return None
            value, ts = val
            age = now - ts
            if age > self._ttl:
                # expired
                try:
                    del self._data[key]
                except KeyError:
                    pass
                return None
            # move to end (most-recent)
            self._data.move_to_end(key)
            return value, int(age)

    def put(self, key: str, value: Any) -> None:
        now = time.time()
        with self._lock:
            self._data[key] = (value, now)
            self._data.move_to_end(key)
            # evict if too large
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)


_instance: Optional[_LRUTTLCache] = None


def init(size: int, ttl_secs: int) -> None:
    global _instance
    _instance = _LRUTTLCache(size, ttl_secs)


def get(key: str) -> Optional[Tuple[Any, int]]:
    if _instance is None:
        return None
    return _instance.get(key)


def put(key: str, value: Any) -> None:
    if _instance is None:
        return
    _instance.put(key, value)


