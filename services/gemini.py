import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY is None:
    raise ValueError("GEMINI_API_KEY가 .env에 없습니다.")

client = genai.Client(
    api_key=API_KEY
)