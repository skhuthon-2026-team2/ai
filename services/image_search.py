import os
import requests

UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")


def _try_search(query: str) -> str | None:
    """Unsplash에 검색어 하나를 보내 이미지 URL을 시도합니다. 실패 시 None."""
    if not UNSPLASH_ACCESS_KEY or not query:
        return None

    url = "https://api.unsplash.com/search/photos"
    params = {
        "query": query,
        "per_page": 1,
        "client_id": UNSPLASH_ACCESS_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        if data.get("results"):
            return data["results"][0]["urls"]["regular"]

    except requests.RequestException:
        pass

    return None


def search_image(query: str) -> str | None:
    """
    검색어 하나로 Unsplash에서 이미지를 찾습니다.

    여러 단어가 합쳐진 검색어("바다 여행" 등)는 결과가 없을 때가 있어,
    먼저 전체 문구로 검색하고, 결과가 없으면 단어 단위로 쪼개서
    하나씩 재시도합니다. 그래도 없으면 None을 반환합니다.
    """
    result = _try_search(query)
    if result:
        return result

    words = query.split()
    for word in words:
        if word == query:
            continue
        result = _try_search(word)
        if result:
            return result

    return None


def search_image_for_recommendation(rec: dict) -> str | None:
    """
    추천 항목(rec) 하나를 받아, 이미지 검색에 가장 적합한 필드부터
    순서대로 시도해 가장 먼저 성공하는 이미지 URL을 반환합니다.

    시도 순서:
    1. image_keyword — AI가 이미지 검색용으로 만든 영어 키워드 (가장 결과가 풍부함)
    2. title — 활동 제목 (단어 단위 재시도 포함, search_image가 처리)

    모두 실패하면 None을 반환하며, 이 경우 프론트는 이미지 없이
    텍스트만 표시하면 됩니다.
    """
    candidates = []

    if rec.get("image_keyword"):
        candidates.append(rec["image_keyword"])

    if rec.get("title"):
        candidates.append(rec["title"])

    for query in candidates:
        result = search_image(query)
        if result:
            return result

    return None