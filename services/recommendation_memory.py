from datetime import datetime
import random

from utils.json_manager import load_json
from utils.json_manager import save_json


HISTORY_FILE = "data/recommendation_history.json"


def get_history():

    data = load_json(HISTORY_FILE)

    if "history" not in data:
        data["history"] = []

    return data


def get_recent_recommendations(limit=10):

    data = get_history()

    recent = []

    for item in reversed(data["history"]):

        recent.extend(item["recommendations"])

        if len(recent) >= limit:
            break

    return recent[:limit]


def save_recommendations(recommendations):

    data = get_history()

    titles = []

    for item in recommendations:

        if isinstance(item, dict):
            titles.append(item["title"])
        else:
            titles.append(item)

    data["history"].append(
        {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "recommendations": titles
        }
    )

    save_json(HISTORY_FILE, data)


def allow_duplicate():

    return random.random() < 0.2