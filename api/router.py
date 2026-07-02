from fastapi import APIRouter, Body

from services.recommender import recommend

router = APIRouter()


@router.get("/")
def home():
    return {
        "message": "AI Server Running"
    }


@router.post("/recommend")
def recommend_activity(data: dict = Body(...)):

    return recommend(data)