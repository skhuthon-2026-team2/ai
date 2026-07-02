from utils.json_manager import load_json
from utils.json_manager import save_json


FILE = "data/feedback.json"


def add_feedback(activity, score):

    data = load_json(FILE)

    if "feedback" not in data:
        data["feedback"] = []

    data["feedback"].append(
        {
            "activity": activity,
            "score": score
        }
    )

    save_json(FILE, data)


def get_feedback():

    data = load_json(FILE)

    return data.get("feedback", [])


def get_best_activities():

    feedback = get_feedback()

    feedback.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return feedback[:5]