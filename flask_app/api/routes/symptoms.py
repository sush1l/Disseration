"""
routes/symptoms.py — Symptom registry endpoints.

GET /symptoms                    → all symptoms (filterable by category)
GET /symptoms/categories         → list of category ids + labels + counts
GET /symptoms/{symptom_id}       → single symptom lookup
GET /symptoms/search/{query}     → fuzzy text search
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api.models.ml import registry

router = APIRouter(prefix="/symptoms", tags=["Symptoms"])


# ─── Response schemas ─────────────────────────────────────────────────────────

class SymptomItem(BaseModel):
    id:       str
    label:    str
    category: str
    index:    int


class CategoryItem(BaseModel):
    id:     str
    label:  str
    count:  int


class SymptomListResponse(BaseModel):
    total:    int
    symptoms: list[SymptomItem]


class CategoryListResponse(BaseModel):
    total:      int
    categories: list[CategoryItem]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=SymptomListResponse,
    summary="List all symptoms",
    description="Returns all 377 symptoms. Optionally filter by category ID.",
)
async def list_symptoms(
    category: str | None = Query(
        default=None,
        description="Filter by category ID (e.g. 'respiratory', 'skin'). "
                    "Use GET /symptoms/categories to see all category IDs.",
    )
) -> SymptomListResponse:
    data = registry.get_symptoms()
    flat: list[dict] = data["flat"]

    if category:
        flat = [s for s in flat if s["category"] == category]
        if not flat:
            raise HTTPException(
                status_code=404,
                detail=f"No symptoms found for category '{category}'. "
                       f"Use GET /symptoms/categories to see valid IDs.",
            )

    return SymptomListResponse(
        total=len(flat),
        symptoms=[SymptomItem(**s) for s in flat],
    )


@router.get(
    "/categories",
    response_model=CategoryListResponse,
    summary="List symptom categories",
    description="Returns all symptom categories with their IDs, labels, and symptom counts.",
)
async def list_categories() -> CategoryListResponse:
    data       = registry.get_symptoms()
    categories = [
        CategoryItem(id=c["id"], label=c["label"], count=c["count"])
        for c in data["categories"]
    ]
    return CategoryListResponse(total=len(categories), categories=categories)


@router.get(
    "/search/{query}",
    response_model=SymptomListResponse,
    summary="Search symptoms by text",
    description="Case-insensitive substring search across symptom IDs and labels.",
)
async def search_symptoms(query: str) -> SymptomListResponse:
    q    = query.strip().lower().replace(" ", "_")
    data = registry.get_symptoms()
    flat = data["flat"]

    matches = [
        s for s in flat
        if q in s["id"] or q in s["label"].lower()
    ]

    return SymptomListResponse(
        total=len(matches),
        symptoms=[SymptomItem(**s) for s in matches],
    )


@router.get(
    "/{symptom_id}",
    response_model=SymptomItem,
    summary="Get a single symptom by ID",
)
async def get_symptom(symptom_id: str) -> SymptomItem:
    data        = registry.get_symptoms()
    id_to_index = data["id_to_index"]
    flat        = data["flat"]
    sid         = symptom_id.strip().lower().replace("-", "_")

    if sid not in id_to_index:
        raise HTTPException(
            status_code=404,
            detail=f"Symptom '{symptom_id}' not found. "
                   f"Use GET /symptoms/search/{{query}} to find it.",
        )

    idx  = id_to_index[sid]
    item = next(s for s in flat if s["index"] == idx)
    return SymptomItem(**item)
