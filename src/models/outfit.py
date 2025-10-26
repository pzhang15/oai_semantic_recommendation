from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


SlotLiteral = Literal["top", "bottom", "shoes", "outerwear", "accessories"]


class OutfitItem(BaseModel):
    id: str
    slot: SlotLiteral
    title: str
    brand: Optional[str] = None
    price: Optional[float] = None
    product_url: Optional[str] = None
    image_url: Optional[str] = None
    why: Optional[str] = None


class OutfitPlan(BaseModel):
    slots: Dict[str, List[OutfitItem]] = Field(default_factory=dict)
    palette: List[str] = Field(default_factory=list)
    materials: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    total_price: Optional[float] = None
    under_budget: Optional[bool] = None


def build_outfit_schema(name: str = "outfit_plan") -> dict:
    schema = OutfitPlan.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema,
            "strict": True,
        },
    }


