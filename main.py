from services.recommender import recommend
from utils.json_manager import load_json
from fastapi import FastAPI
from api.router import router

app = FastAPI(
    title="AI Activity Recommendation Server",
    version="1.0"
)

app.include_router(router)

def main():

    print("=" * 60)
    print("AI 활동 분석 및 추천")
    print("=" * 60)

    activity_data = load_json("data/activities.json")

    result = recommend(activity_data)

    print(result)


if __name__ == "__main__":
    main()