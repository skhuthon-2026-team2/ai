import io

import requests
from PIL import Image

from services.gemini import client
from prompts.image_prompt import IMAGE_ANALYSIS_PROMPT

from utils.parser import parse_json


def _load_image(image_url: str) -> Image.Image:
    """
    activity["image"]에 담긴 S3 퍼블릭 URL로부터 이미지를 다운로드하여
    PIL Image로 변환합니다.
    """
    response = requests.get(image_url, timeout=10)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))


def analyze_image(image_url):

    try:

        image = _load_image(image_url)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                image,
                IMAGE_ANALYSIS_PROMPT
            ]
        )

        return parse_json(response.text)

    except Exception as e:

        return {
            "activity": "",
            "place_type": "",
            "season": "",
            "mood": "",
            "features": [],
            "error": str(e)
        }