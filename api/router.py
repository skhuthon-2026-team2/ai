from fastapi import APIRouter, Body

from services.recommender import recommend
from services.feedback_service import add_feedback

router = APIRouter()


@router.get("/")
def home():
    return {
        "message": "AI Server Running"
    }


@router.post("/recommend")
def recommend_activity(data: dict = Body(...)):

    return recommend(data)


@router.post("/feedback")
def feedback(data: dict = Body(...)):

    add_feedback(
        data["activity"],
        data["score"]
    )

    return {
        "message": "feedback saved"
    }