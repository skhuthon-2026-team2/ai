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
class KeywordResponse(BaseModel):
    keywords: List[str]
    image_description: str  # 사진 속 상황/분위기를 요약한 텍스트 (2단계에서 사진 재전송 없이 재사용)

class Caption(BaseModel):
    title: str
    script: str
    tags: List[str]

class CaptionResponse(BaseModel):
    captions: List[Caption]


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


def _build_context_string(date: Optional[str] = None, time: Optional[str] = None, place: Optional[str] = None) -> str:
    """
    날짜/시간/장소 메타데이터를 프롬프트에 넣을 문자열로 변환.
    값이 없으면 그냥 빈 문자열을 반환 (기존 이미지 전용 흐름과 동일하게 동작).
    """
    lines = []
    if place:
        lines.append(f"장소: {place}")
    if date:
        lines.append(f"날짜: {date}")
    if time:
        lines.append(f"시간: {time}")

    if not lines:
        return ""

    return "\n[사용자가 입력한 상황 정보]\n" + "\n".join(lines) + "\n위 정보도 참고해서 분위기와 상황을 더 정확하게 파악해줘.\n"


# ========================================================
#[백엔드 전용] 이미지 바이트 배열 (+선택적 메타데이터)를 받아 키워드 리스트 반환
# ========================================================
def extract_keywords_from_multiple_images(
    image_bytes_list: List[bytes],
    mime_types: Optional[List[str]] = None,
    date: Optional[str] = None,
    time: Optional[str] = None,
    place: Optional[str] = None,
) -> dict:
    """
    반환값: {"keywords": [...], "image_description": "..."}
    image_description은 2단계(generate_scripts_from_description)에서
    사진을 다시 보내지 않고도 캡션을 쓸 수 있도록, 사진 속 상황/분위기를 요약한 텍스트입니다.
    """
    contents_list = _build_image_parts(image_bytes_list, mime_types)

    context_string = _build_context_string(date, time, place)

    sys_instruct = f"""
너는 인스타 피드 감각 좋은 25살이야.
유저가 올린 사진들을 여러 번 훑어보면서 상황이랑 분위기를 파악해줘.
{context_string}
아래 두 가지를 뽑아줘.

1. keywords: 이 사진들에 딱 맞는 인스타그램 해시태그 6개
[이런 태그로]
- 실제로 인스타에서 쓰이는 태그.
- 사진 속 장소, 상황, 감정을 자연스럽게 담을 것.
- 예시: #도쿄한달살기 #라멘투어 #여행마지막날 #발아파죽는줄 #또가고싶음 #여행같이가자
- 예시: #원데이클래스 #취미생활 #또하고싶다 #주말취미 #손으로만드는것들
[절대 쓰지 마]
- #일상 #소통 #맞팔 같은 범용 태그
- #빛나는하루 #소중한인연 같은 올드한 표현
- 광고나 홍보 느낌 나는 태그

2. image_description: 사진 속 상황, 장소, 분위기, 눈에 띄는 디테일(색감, 날씨, 사물, 표정 등)을
   나중에 다른 사람이 사진을 안 보고도 캡션을 쓸 수 있을 정도로 3~5문장으로 구체적으로 묘사.
   감성적 미사여구 없이 담백하게, 관찰한 사실 위주로 작성.
"""
    contents_list.append("제시된 모든 사진들을 분석해서 keywords와 image_description을 만들어줘.")

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=contents_list,
        config=types.GenerateContentConfig(
            system_instruction=sys_instruct,
            response_mime_type="application/json",
            response_schema=KeywordResponse,
            temperature=1.2
        )
    )
    return json.loads(response.text)

# ========================================================
# [백엔드 전용] 이미지 설명 텍스트 + 태그 (+선택적 메타데이터)를 받아 캡션 리스트 반환
# 1단계(extract_keywords_from_multiple_images)에서 받은 image_description을 그대로 넘기면 됨.
# 사진 파일을 다시 첨부할 필요 없음.
# ========================================================
def generate_scripts_from_description(
    image_description: str,
    selected_tags: List[str],
    date: Optional[str] = None,
    time: Optional[str] = None,
    place: Optional[str] = None,
) -> List[Caption]:
    tags_string = ", ".join(selected_tags)
    context_string = _build_context_string(date, time, place)

    sys_instruct = f"""
너는 인스타 캡션 잘 쓰는 25살이야.

[사진 속 상황 묘사]
{image_description}

유저가 선택한 태그: {tags_string}
{context_string}
위 사진 묘사를 참고해서 제목 + 캡션 세트를 3개 써줘.
---
[이런 말투로 써줘]
"갔다왔는데 생각보다 별로였음.
근데 이상하게 기억은 남아.
사진도 많이 찍었고 ㅋㅋ 왜 찍었는지 모르겠지만."

"딱히 기대한 건 없었는데
막상 가니까 나쁘지 않았어.
다음에 또 올지는 모르겠고."

"되게 오래 걸어서 다리 좀 아팠음.
근데 그것도 나름 재밌었던 것 같기도 하고.
배고파서 중간에 뭔가 먹었는데 그게 더 맛있었음."

"사진은 열심히 찍었는데 정작 올릴 건 별로 없네.
뭔가 그냥 그랬달까.
또 가자고는 했는데 진짜 갈지는 모름."

이런 사용자들의 말투를 참고하여 스크립트를 작성해줘. 이 문구 그대로 작성해달라는 것 아님.
---
[제목 쓰는 법]
- 10자 내외, 사진 분위기랑 딱 맞는 한 줄로
- 아래 3가지 스타일 중 사진에 맞는 걸로 써줘

  (감정 툭 던지기) "그냥 좋았어", "생각보다였음", "이상하게 기억에 남음"
  (상황 단편적으로) "비 오기 직전", "마지막 날 아침", "밥 먹으러 갔다가"
  (혼잣말) "또 가야 하나", "다음엔 더 오래", "근데 또 가고 싶음"
- 나쁜 예시: "빛나는 서울의 하루", "소중한 우리들의 이야기", "롯데타워 너무 높음"
---
[스크립트 분량]
- 3~4줄, 한 줄당 20~35자 내외
- 너무 짧으면 안 됨. 생각이 자연스럽게 이어지는 느낌으로
---
[문장 끝내는 법]
- 깔끔하게 마무리 짓거나 결론 내리지 마.
- 잘된 예시: "근데 다리는 좀 아팠음", "뭔가 또 오고 싶긴 한데 귀찮아질 것 같고", "같이 간 친구가 더 신나했음 ㅋㅋ"
- 나쁜 예시: "기분 전환 제대로 한 듯", "다음에 또 와도 괜찮을 것 같아", "좋은 하루였어"
---
[절대 쓰지 마]
- "황홀하다", "예술이다", "압도된다", "제대로 난다"
- "빛나는", "소중한", "따뜻한", "몽글몽글"
- "함께여서 행복해", "오늘도 좋은 하루"
- 이모지, 이모티콘 (ㅎㅎ, ㅠㅠ 정도는 허용)
- 3개의 캡션 톤·구조가 서로 비슷하게 반복되는 것
"""
    # ✨ 수정됨: 위 프롬프트 맨 아래에 있던 불필요한 [출력 형식] 텍스트 지시문을 깔끔하게 삭제했습니다.

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="위 사진 묘사와 태그를 조합해서 제목과 캡션 3개 만들어줘.",
        config=types.GenerateContentConfig(
            system_instruction=sys_instruct,
            response_mime_type="application/json",
            response_schema=CaptionResponse,
            temperature=1.0  # ✨ 보너스 수정: JSON 포맷이 깨지지 않도록 창의성 온도를 1.2에서 1.0으로 살짝 낮춰 안정성을 챙겼습니다.
        )
    )
    return [Caption(**c) for c in json.loads(response.text)['captions']]


# ========================================================
# [로컬 테스트]
# ========================================================
if __name__ == "__main__":
    target_images = ["lt1.jpg", "lt2.jpg"]

    print("\n" + "="*50)
    print("AI 초안 로컬 테스트")
    print("="*50 + "\n")

    image_bytes_list = []
    for path in target_images:
        if not os.path.exists(path):
            print(f"에러: '{path}' 파일이 없습니다!")
            exit()
        with open(path, "rb") as f:
            image_bytes_list.append(f.read())

    print("[1/2] 이미지 분석 중...\n")
    result = extract_keywords_from_multiple_images(image_bytes_list)
    ai_keywords = result["keywords"]
    image_description = result["image_description"]

    print(f"[사진 설명]\n{image_description}\n")

    if ai_keywords:
        print("원하는 키워드를 선택해주세요")
        for i, kw in enumerate(ai_keywords, 1):
            print(f"  [{i}] {kw}")

        custom_btn_num = len(ai_keywords) + 1
        print(f"  [{custom_btn_num}] + 직접 입력할래요")
        print("\n" + "-"*50)

        user_choices = input(f"번호를 띄어쓰기로 입력 (예: 1 3 {custom_btn_num}) : ")
        choice_numbers = user_choices.split()

        selected_tags = []
        for num in choice_numbers:
            if num.isdigit() and 1 <= int(num) <= len(ai_keywords):
                selected_tags.append(ai_keywords[int(num)-1])

        if str(custom_btn_num) in choice_numbers:
            print("\n나만의 키워드를 입력하세요.")
            custom_tag = input("키워드 입력 : ")
            if custom_tag.strip():
                if not custom_tag.startswith("#"):
                    custom_tag = "#" + custom_tag
                selected_tags.append(custom_tag)

        if not selected_tags:
            selected_tags = [ai_keywords[0]]

        print(f"\n선택된 키워드: {selected_tags}")
        print("-" * 50 + "\n")

        print("[2/2] 스크립트 작성 중...\n")
        final_scripts = generate_scripts_from_description(image_description, selected_tags)

        print("[최종 결과] 생성된 캡션 3가지")
        print("=" * 50)
        for i, caption in enumerate(final_scripts, 1):
            tags_str = " ".join(caption.tags) if caption.tags else ""
            print(f"{i}번 캡션")
            print(f"제목: {caption.title}")
            print(f"스크립트:\n{caption.script}")
            print(f"태그: {tags_str}\n")
        print("=" * 50)