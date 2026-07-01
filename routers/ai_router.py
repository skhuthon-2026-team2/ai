import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from typing import List, Optional

# 옆 폴더(services)에 있는 AI 함수를 불러옵니다.
from services.AI_keyword import generate_caption_directly

router = APIRouter()
logger = logging.getLogger(__name__)


def _validate_and_extract(files: List[UploadFile]):
    """모든 파일이 이미지인지 검증만 하는 헬퍼 (읽지는 않음)."""
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")


@router.post("/auto-caption")
async def auto_caption(
    files: List[UploadFile] = File(...),
    date: Optional[str] = Form(None),
):
    """
    이미지 업로드하면 AI가 바로 제목과 상세 설명을 작성합니다.
    
    입력:
    - files (필수): 사진 파일 (1장 이상)
    - date (선택): 게시 예정 날짜 (예: "2026-07-01")
    
    출력:
    {
      "success": true,
      "title": "고급스러운 백화점 가방",
      "script": "거의 안 써서 상태 좋습니다. 색상도 예쁘고...",
      "message": "제목과 설명이 작성되었습니다. 다시 작성하고 싶다면 [다시 생성] 버튼을 눌러주세요."
    }
    """
    _validate_and_extract(files)

    try:
        # 비동기로 파일을 읽어서 바이트(bytes) 리스트로 변환
        image_bytes_list = []
        mime_types = []
        for file in files:
            contents = await file.read()
            image_bytes_list.append(contents)
            mime_types.append(file.content_type)

        logger.info(f"{len(files)}장의 이미지로 캡션을 생성합니다... (date={date})")
        
        # AI 호출을 백그라운드 스레드로 넘겨서 서버 멈춤 방지
        caption = await run_in_threadpool(
            generate_caption_directly,
            image_bytes_list,
            mime_types=mime_types,
            date=date,
        )

        logger.info("캡션 생성 성공!")
        return {
            "success": True,
            "title": caption.title,
            "script": caption.script,
            "message": "제목과 설명이 작성되었습니다. 마음에 안 들면 다시 생성할 수 있습니다.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI 처리 중 오류가 발생했습니다: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"AI가 현재 너무 바쁩니다! 잠시 후 다시 시도해주세요. (사유: {str(e)})"
        )