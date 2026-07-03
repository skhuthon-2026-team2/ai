from PIL import Image

from services.gemini import client
from prompts.image_prompt import IMAGE_ANALYSIS_PROMPT

from utils.parser import parse_json


def analyze_image(image_path):

    try:

        image = Image.open(image_path)

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