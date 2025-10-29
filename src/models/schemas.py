from typing import List, Optional, Literal

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class Item(BaseModel):
    id: str
    title: Optional[str] = None
    score: Optional[float] = None


class SearchResponse(BaseModel):
    items: List[Item]


class Budget(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    currency: str = "USD"


class QueryFacets(BaseModel):
    intent: Literal["search", "outfit", "compare", "unknown"] = "unknown"
    occasion: Optional[str] = None
    season: Optional[Literal["spring", "summer", "fall", "winter", "all-season"]] = None
    style_adjectives: List[str] = Field(default_factory=list)
    colors: List[str] = Field(default_factory=list)
    materials: List[str] = Field(default_factory=list)
    categories_include: List[str] = Field(default_factory=list)
    categories_exclude: List[str] = Field(default_factory=list)
    gender_or_fit: Optional[Literal["men", "women", "unisex", "girls", "boys", "unknown"]] = None
    size_notes: Optional[str] = None
    budget_usd: Budget = Field(default_factory=Budget)
    must_have: List[str] = Field(default_factory=list)
    hard_constraints: List[str] = Field(default_factory=list)
    num_items: int = 12

    @field_validator("num_items")
    @classmethod
    def _clamp_num_items(cls, v: int) -> int:
        try:
            n = int(v)
        except Exception:
            n = 12
        return max(1, min(50, n))


def build_structured_output_schema(model: type[BaseModel], name: str) -> dict:
    schema = model.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema,
            "strict": True,
        },
    }


class RetrieveRequest(BaseModel):
    query: str
    k: int = 12
    use_hybrid: bool | None = None
