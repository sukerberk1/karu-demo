import os

from dotenv import load_dotenv

load_dotenv()

# === OPEN AI CONNECTION DETAILS ===
OPEN_AI_KEY = os.environ.get("OPENAI_API_KEY")