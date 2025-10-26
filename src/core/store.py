from typing import Any


class VectorStore:
    def __init__(self) -> None:
        self._initialized = False

    def initialize(self) -> None:
        self._initialized = True

    def is_ready(self) -> bool:
        return self._initialized

    def search(self, query_vector: list[float], top_k: int = 10) -> list[dict[str, Any]]:
        return []


