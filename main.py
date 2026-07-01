import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# 새로 만든 ai_router를 불러옵니다
from routers import ai_router
# from routers import recommend_router  # (2차 AI - 활동 추천 라우터 완성되면 주석 해제)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI 카피라이터 서버",
    description="Gemini Vision AI를 이용한 인스타 해시태그 및 캡션 자동 생성 API",
    version="1.0.0"
)

# CORS 설정: 프론트엔드(다른 도메인)에서 이 서버 API를 호출할 수 있도록 허용합니다.
# 해커톤 개발 단계라 우선 전체 허용("*")으로 두고,
# 프론트 배포 주소가 확정되면 allow_origins를 그 주소로 좁히는 걸 권장합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록: 외부에서 /api/v1/generate-keywords 주소로 접속할 수 있게 연결합니다.
app.include_router(ai_router.router, prefix="/api/v1", tags=["AI_Generation"])
# app.include_router(recommend_router.router, prefix="/api/v1", tags=["AI_Recommendation"])  # (2차 AI 완성되면 주석 해제)

@app.get("/")
def health_check(request: Request):
    """서버가 죽지 않고 잘 살아있는지 확인하는 창구입니다."""
    logger.info(f"Health check called from {request.client.host}")
    return {"status": "ok", "message": "AI Server is running perfectly!"}