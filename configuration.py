import os
from dotenv import load_dotenv

load_dotenv()

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_API_KEY")
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY")
