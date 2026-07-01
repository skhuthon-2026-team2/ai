import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.concurrency import run_in_threadpool  # ✨ 추가됨: 비동기 최적화를 위한 스레드풀
from typing import List, Optional

# 옆 폴더(services)에 있는 AI 함수를 불러옵니다.
from services.AI_keyword import (
    extract_keywords_from_multiple_images,
    generate_scripts_from_description,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _validate_and_extract(files: List[UploadFile]):
    """모든 파일이 이미지인지 검증만 하는 헬퍼 (읽지는 않음)."""
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")


@router.post("/generate-keywords")
async def generate_keywords(
    files: List[UploadFile] = File(...),
    date: Optional[str] = Form(None),
    time: Optional[str] = Form(None),
    place: Optional[str] = Form(None),
):
    """
    유저가 업로드한 여러 장의 사진(+선택적으로 날짜/시간/장소)을 분석하여
    인스타 감성 해시태그를 즉시 반환합니다.
    """
    _validate_and_extract(files)

    try:
        # 비동기로 파일을 읽어서 바이트(bytes) 리스트로 변환하면서, 실제 mime_type도 같이 기록
        image_bytes_list = []
        mime_types = []
        for file in files:
            contents = await file.read()
            image_bytes_list.append(contents)
            mime_types.append(file.content_type)

        logger.info(f"{len(files)}장의 이미지 분석을 시작합니다... (date={date}, time={time}, place={place})")
        
        # ✨ 수정됨: AI 호출을 백그라운드 스레드로 넘겨서 서버 멈춤 방지
        result = await run_in_threadpool(
            extract_keywords_from_multiple_images,
            image_bytes_list,
            mime_types=mime_types,
            date=date,
            time=time,
            place=place,
        )

        logger.info("키워드 추출 성공!")
        return {
            "success": True,
            "tags": result["keywords"],
            # 2단계(/generate-script) 호출 시 프론트가 이 값을 그대로 다시 보내주면 됨 (사진 재첨부 불필요)
            "image_description": result["image_description"],
        }

    except HTTPException:
        raise
    except Exception as e:
        # 구글 서버 과부하(503, 429) 등의 에러가 발생하면 여기서 부드럽게 처리됩니다.
        logger.error(f"AI 처리 중 오류가 발생했습니다: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI가 현재 너무 바쁩니다! 잠시 후 다시 시도해주세요. (사유: {str(e)})")


@router.post("/generate-script")
async def generate_script(
    image_description: str = Form(...),
    selected_tags: List[str] = Form(...),
    date: Optional[str] = Form(None),
    time: Optional[str] = Form(None),
    place: Optional[str] = Form(None),
):
    """
    유저가 앞 단계에서 선택한 키워드 + 1단계 응답에서 받은 image_description을 받아
    최종 캡션(제목 + 스크립트 + 태그) 3개를 생성합니다.

    사진을 다시 첨부할 필요 없음. 프론트엔드는 /generate-keywords 응답에서 받은
    image_description 값을 그대로 이 요청에 담아 보내면 됩니다.
    """
    if not selected_tags:
        raise HTTPException(status_code=400, detail="선택된 키워드가 없습니다.")

    if not image_description or not image_description.strip():
        raise HTTPException(status_code=400, detail="image_description이 필요합니다. (1단계 응답값을 그대로 전달해주세요)")

    try:
        logger.info(f"선택된 키워드 {selected_tags}로 캡션 생성을 시작합니다...")
        
        # ✨ 수정됨: 스크립트 생성 역시 무거운 작업이므로 스레드풀 적용
        captions = await run_in_threadpool(
            generate_scripts_from_description,
            image_description,
            selected_tags,
            date=date,
            time=time,
            place=place,
        )

        logger.info("캡션 생성 성공!")
        return {
            "success": True,
            "captions": [c.model_dump() for c in captions]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI 처리 중 오류가 발생했습니다: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI가 현재 너무 바쁩니다! 잠시 후 다시 시도해주세요. (사유: {str(e)})")