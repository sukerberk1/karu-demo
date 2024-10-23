import os

from dotenv import load_dotenv

load_dotenv()

# === OPEN AI CONNECTION DETAILS ===
OPEN_AI_KEY = os.environ.get("OPEN_AI_KEY")
OPEN_AI_PROJ = os.environ.get("OPEN_AI_PROJ")
OPEN_AI_ORG = os.environ.get("OPEN_AI_ORG")