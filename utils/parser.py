import json


def parse_json(text):

    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    try:
        return json.loads(text)

    except Exception:
        return {
            "raw": text
        }