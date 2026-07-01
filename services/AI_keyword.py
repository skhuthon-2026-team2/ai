import os
import json
from google import genai
from google.genai import types 
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# 1. API 키 셋업
load_dotenv()
client = genai.Client(api_key=os.environ.get("API_KEY"))

# 2. 데이터 규격 정의
class AutoCaption(BaseModel):
    """자동 생성된 캡션 응답"""
    title: str
    script: str
    success: bool = True


def _build_image_parts(image_bytes_list: List[bytes], mime_types: Optional[List[str]] = None) -> List[types.Part]:
    """
    이미지 바이트 리스트를 Gemini에 넣을 Part 객체로 변환.
    mime_types가 주어지면 각 이미지의 실제 타입을 사용하고,
    없으면 안전하게 jpeg로 기본 처리.
    """
    parts = []
    for i, image_bytes in enumerate(image_bytes_list):
        mime_type = mime_types[i] if mime_types and i < len(mime_types) else "image/jpeg"
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
    return parts


def _build_context_string(date: Optional[str] = None) -> str:
    """
    날짜 정보를 프롬프트에 넣을 문자열로 변환.
    날짜는 참고용일 뿐 AI 분석에 크게 영향을 주지는 않음.
    """
    if date:
        return f"[게시 예정 날짜]\n{date}\n"
    return ""


# ========================================================
# [메인 함수] 이미지를 받아서 바로 제목 + 캡션 1개 생성, 사진만 올리면 AI가 바로 제목+설명 작성
# ========================================================
def generate_caption_directly(
    image_bytes_list: List[bytes],
    mime_types: Optional[List[str]] = None,
    date: Optional[str] = None,
) -> AutoCaption:
    """
    이미지를 분석해서 제목과 상세 설명을 한 번에 생성합니다.
    
    입력: 이미지(들) + 날짜(선택)
    출력: {"title": "...", "script": "..."}
    
    사진을 한 번만 업로드하면 되고, "다시 생성" 버튼으로 다른 스타일을 얻을 수 있습니다.
    """
    contents_list = _build_image_parts(image_bytes_list, mime_types)
    context_string = _build_context_string(date)

    sys_instruct = f"""
너는 동아리 활동 사진을 보고 그 순간의 감정과 분위기를 담아 제목과 설명을 써주는 AI야.
사진 속 순간의 매력을 간결하고 감각적으로 표현해줘.

{context_string}

[제목 작성 규칙]
- 8~12자, 사진의 핵심을 한 단어나 짧은 구로
- 그 순간의 감정이나 분위기를 담기
- 예시: "바다 맑음", "봉우리 도착", "햇빛 반사", "숲 속 한숨", "따뜻한 오후"
- 이모지나 특수기호는 쓰지 말 것

[설명 작성 규칙]
- 2~3문장, 마치 일기를 쓰듯 차분하고 따뜻한 톤
- 사진에 보이는 것만 담기 (풍경, 날씨, 분위기, 그때의 기분)
- 과장이나 거짓 정보는 절대 금지
- 사진에 없는 정보 추가하지 않기
- 예시 톤:
  "햇빛이 물에 반사되고 있다. 자전거를 타며 불어오는 바람이 좋다."
  "숲 사이로 햇빛이 들어온다. 조용함 속에서 평온을 느낀다."
  "카페 창밖의 거리가 보인다. 시간이 천천히 흐르는 오후다."

[절대 금지]
- "최고예요", "정말 좋아요!" 같은 과도한 표현
- "이렇게 아름다운 곳은 처음" 같은 과장
- 추측이나 상상으로 정보 만들기
- 광고나 홍보하는 톤
"""

    contents_list.append("이 사진을 보고 제목과 상세 설명을 작성해줘. 자연스럽고 사실적으로.")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents_list,
        config=types.GenerateContentConfig(
            system_instruction=sys_instruct,
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "10자 내외의 제목"
                    },
                    "script": {
                        "type": "string",
                        "description": "2~3문단의 상세 설명"
                    }
                },
                "required": ["title", "script"]
            },
            temperature=1.1  # 매번 다른 스타일로 생성하기 위해 약간의 창의성 확보
        )
    )
    
    result = json.loads(response.text)
    return AutoCaption(
        title=result.get("title", ""),
        script=result.get("script", ""),
        success=True
    )


# ========================================================
# [로컬 테스트]
# ========================================================
if __name__ == "__main__":
    target_image = "lt2.jpg"

    print("\n" + "="*50)
    print("AI 캡션 생성 테스트")
    print("="*50 + "\n")

    if not os.path.exists(target_image):
        print(f"에러: '{target_image}' 파일이 없습니다!")
        exit()
    
    with open(target_image, "rb") as f:
        image_bytes = f.read()

    print("AI가 제목과 설명을 작성 중입니다...\n")
    caption = generate_caption_directly([image_bytes], date="2026-07-01")

    print("[생성된 캡션]")
    print("=" * 50)
    print(f"제목: {caption.title}")
    print(f"\n상세 설명:\n{caption.script}")
    print("=" * 50)
    
    # 다시 생성 테스트
    print("\n\n[다시 생성 - 다른 스타일]")
    caption2 = generate_caption_directly([image_bytes], date="2026-07-01")
    print("=" * 50)
    print(f"제목: {caption2.title}")
    print(f"\n상세 설명:\n{caption2.script}")
    print("=" * 50)