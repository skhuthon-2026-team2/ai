from fastapi import APIRouter, Body

from services.recommender import recommend

router = APIRouter()


@router.post("/recommend")
def recommend_activity(data: dict = Body(...)):

    return recommend(data)