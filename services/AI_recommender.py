import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List, Optional, Union
from dotenv import load_dotenv

# 1. API 키 셋업
load_dotenv()
client = genai.Client(api_key=os.environ.get("API_KEY"))

# 2. 백엔드와 약속한 데이터 출력 규격 (JSON)
class RecommendationResponse(BaseModel):
    similar_activity: str
    new_activity: str


# ==========================================
# [내부 헬퍼] 구조화된 활동 기록(List[dict])을 프롬프트용 텍스트로 변환
# ==========================================
def _format_user_history(user_history: Union[str, List[dict]]) -> str:
    """
    user_history를 두 가지 형태로 받을 수 있게 처리:
    1. 이미 텍스트로 가공된 문자열 (기존 방식, 하위 호환)
    2. DB에서 바로 꺼낸 구조화된 기록 리스트
       예: [{"days_ago": 5, "tags": ["#한강공원", "#자전거"]}, ...]

    구조화된 리스트를 넘기면 이 함수가 알아서 텍스트로 변환해주기 때문에,
    백엔드는 DB 레코드를 그대로 넘기기만 하면 됨.
    """
    if isinstance(user_history, str):
        return user_history.strip()

    if not user_history:
        return ""

    lines = []
    for i, record in enumerate(user_history, 1):
        days_ago = record.get("days_ago")
        tags = record.get("tags", [])
        tags_str = ", ".join(tags) if tags else "태그 없음"
        time_label = f"[{days_ago}일 전]" if days_ago is not None else ""
        lines.append(f"{i}. {time_label} 주 사용 태그: {tags_str}")

    return "\n".join(lines)


# ==========================================
# [메인 엔진] 백엔드가 호출해서 사용할 순수 함수
# ==========================================
def get_activity_recommendation(
    room_context: str,
    user_history: Union[str, List[dict]] = "",
) -> dict:
    """
    백엔드로부터 방 정보와 유저 과거 기록을 받아,
    투트랙(유사 취향, 분위기 환기) 추천 결과를 딕셔너리로 반환합니다.

    user_history는 문자열(기존 방식) 또는 구조화된 리스트 둘 다 지원합니다.
    """
    formatted_history = _format_user_history(user_history)
    has_history = bool(formatted_history)

    if has_history:
        history_section = f"- 이 유저의 과거 활동 기록(주로 쓴 태그들): {formatted_history}"
        history_guideline = "위 데이터를 바탕으로, 이 방의 주제에 맞는 다음 활동을 2가지 방향으로 제안해 줘."
    else:
        # 기록이 없는 신규 유저 → 과거 기록 언급 없이 방 성격만으로 추천
        history_section = "- 이 유저는 아직 활동 기록이 없는 신규 유저야."
        history_guideline = "과거 기록이 없으니, 이 방의 주제와 성격만 보고 처음 시작하기 좋은 활동을 2가지 방향으로 제안해 줘."

    sys_instruct = f"""
    너는 인스타그램 릴스와 트렌드에 극도로 민감한 20대 대학생 맞춤형 큐레이터야.

    [분석할 데이터]
    - 현재 소속된 방의 성격: {room_context}
    {history_section}

    {history_guideline}
    1. similar_activity (취향 연장선): 과거 기록의 감성과 비슷한 분위기나 장소 (기록 없으면 방 성격에 잘 맞는 무난한 활동)
    2. new_activity (분위기 환기): 기존 활동과 반대되거나 완전 색다른 자극을 주는 추천

    [절대 금지 규칙]
    - '청춘', '낭만', '추억', '힐링', '우정' 등 뻔하고 올드한 싸이월드 감성 단어 절대 금지.
    - 너무 길고 딱딱한 문장 금지.

    [추구하는 요즘 감성]
    - 친구에게 DM이나 카톡으로 무심하게 툭 추천하듯, 담백하고 센스 있는 말투.
    - 각 옵션당 핵심만 2~3줄로 명확하게 작성.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="위의 방 성격과 유저 기록을 분석해서 맞춤형 활동 2가지를 추천해 줘.",
        config=types.GenerateContentConfig(
            system_instruction=sys_instruct,
            response_mime_type="application/json",
            response_schema=RecommendationResponse,
            temperature=1.1,  # 1차 AI(캡션 생성)와 톤 일관성 유지 + 매번 비슷한 문구 반복 방지
        )
    )

    try:
        return json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"AI 응답을 파싱하는 데 실패했습니다: {str(e)}")


# ==========================================
# 로컬 자동화 테스트 실행부 (백엔드 연동 시 무시됨)
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("AI 맞춤형 추천 기능 로컬 테스트")
    print("="*50 + "\n")

    # [테스트 1] 기존 방식 - 텍스트로 미리 가공된 기록
    test_room = "방 이름: 기분 전환 / 카테고리: 맛집, 공원, 액티비티, 야외"
    test_history_str = """
    1. [5일 전] 주 사용 태그: #한강공원, #자전거, #치맥, #야외활동
    2. [2주 전] 주 사용 태그: #보드게임, #실내놀거리, #승부욕, #루미큐브
    3. [3일 전] 주 사용 태그: #방탈출, #머리쓰기, #힌트요정, #실내액티비티"""

    print("[테스트 1] 텍스트 기록 방식")
    print(f" - 방 성격: {test_room}")
    print(f" - 유저 기록: {test_history_str}\n")
    result = get_activity_recommendation(test_room, test_history_str)
    print("[옵션 A: 취향 연장선]", result['similar_activity'])
    print("[옵션 B: 분위기 환기]", result['new_activity'])
    print("-" * 50 + "\n")

    # [테스트 2] 구조화된 리스트 방식 (실제 DB 조회 결과와 유사한 형태)
    test_history_list = [
        {"days_ago": 5, "tags": ["#한강공원", "#자전거", "#치맥"]},
        {"days_ago": 14, "tags": ["#보드게임", "#실내놀거리"]},
    ]
    print("[테스트 2] 구조화된 리스트 방식")
    result2 = get_activity_recommendation(test_room, test_history_list)
    print("[옵션 A: 취향 연장선]", result2['similar_activity'])
    print("[옵션 B: 분위기 환기]", result2['new_activity'])
    print("-" * 50 + "\n")

    # [테스트 3] 신규 유저 (기록 없음)
    print("[테스트 3] 신규 유저 (기록 없음)")
    result3 = get_activity_recommendation(test_room, "")
    print("[옵션 A: 취향 연장선]", result3['similar_activity'])
    print("[옵션 B: 분위기 환기]", result3['new_activity'])
    print("-" * 50)