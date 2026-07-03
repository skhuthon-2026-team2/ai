import json

from services.gemini import client
from services.image_analyzer import analyze_image
from services.image_search import search_image_for_recommendation

from services.recommendation_memory import (
    get_recent_recommendations,
    save_recommendations,
    allow_duplicate
)

from prompts.recommend_prompt import RECOMMEND_PROMPT

from utils.parser import parse_json


def recommend(activity_data):

    print("=" * 50)
    print("사용자 활동 분석 시작")
    print("=" * 50)

    # 활동이 하나도 없는 경우
    if not activity_data.get("activities"):
        return {
            "message": "분석할 활동 데이터가 없습니다.",
            "recommendations": []
        }

    history = ""

    print("\n이미지 분석 중...\n")

    # -------------------------
    # 활동 + 이미지 분석
    # -------------------------
    for activity in activity_data["activities"]:

        image_result = analyze_image(activity["image"])

        history += f"""
제목 : {activity["title"]}


설명 : {activity["description"]}

날짜 : {activity["date"]}

사진 분석 결과

{json.dumps(image_result, ensure_ascii=False)}

"""

    # -------------------------
    # 최근 추천 기록
    # -------------------------
    recent = get_recent_recommendations()
    recent_text = "\n".join(recent)

    # -------------------------
    # Gemini Prompt 생성
    # -------------------------
    prompt = f"""
{RECOMMEND_PROMPT}

=========================
사용자 활동
=========================

{history}

=========================
최근 추천했던 활동
=========================

{recent_text}

=========================

최근 추천과 최대한 겹치지 않도록 추천하세요.

단,

약 20% 정도는 기존 추천을 다시 추천할 수 있습니다.

반드시 JSON만 출력하세요.
"""

    print("작성자에게 적합한 활동 추천 중...\n")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    result = parse_json(response.text)

    # -------------------------
    # Python 중복 검사
    # -------------------------
    if "recommendations" in result:

        recent = get_recent_recommendations()

        filtered = []

        duplicate_allowed = allow_duplicate()

        for rec in result["recommendations"]:

            title = rec["title"]

            # 최근 추천과 중복인 경우
            if title in recent:

                # 20% 확률만 허용
                if duplicate_allowed:
                    filtered.append(rec)

            else:
                filtered.append(rec)

        # -------------------------
        # 추천 활동별 이미지 검색
        # image_keyword -> location -> title 순으로 시도
        # -------------------------
        print("\n추천 활동 이미지 검색 중...\n")

        for rec in filtered:
            rec["image_url"] = search_image_for_recommendation(rec)

        result["recommendations"] = filtered

        # 추천 기록 저장
        save_recommendations(filtered)

    return result