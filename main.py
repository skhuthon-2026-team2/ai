import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# 캡션 생성 라우터
from routers import ai_router
# 활동 추천 라우터
from api.router import router as recommend_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI 캡션 생성 서버",
    description="Gemini 2.5 Flash를 활동 제목, 자세한 설명 자동 생성",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ai_router.router, prefix="/api/v1", tags=["AI_Caption"])
app.include_router(recommend_router, prefix="/api/v1", tags=["AI_Recommendation"])

@app.get("/")
def health_check(request: Request):
    """서버가 죽지 않고 잘 살아있는지 확인하는 창구입니다."""
    logger.info(f"Health check called from {request.client.host}")
    return {
        "status": "ok",
        "message": "AI 캡션 생성 서버 정상 작동 중!",
        "version": "2.0.0"
    }