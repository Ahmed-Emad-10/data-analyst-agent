import os
import dotenv

dotenv.load_dotenv(".env")

# --- Dataset ---
CSV_PATH = os.environ.get("CSV_PATH", "D:\\Edu\\Projects\\draft\\CSV Agent\\customers.csv")
SCHEMA_SAMPLE_ROWS = int(os.environ.get("SCHEMA_SAMPLE_ROWS", "5"))

# --- LLM ---
LLM_MODEL = os.environ.get("LLM_MODEL", "ollama/gemma4:e4b")

LLM_API_KEY = os.environ.get("LLM_API_KEY")  # required, no default
#LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434")  # only used for Ollama LLMs

if not LLM_API_KEY:
    raise RuntimeError(
        "LLM_API_KEY environment variable is not set. "
        "Export it or add it to a .env file before starting the app."
    )

# --- Database ---
DB_PATH = os.environ.get("DB_PATH", "sessions.db")

# --- Chat history ---
# Number of most recent FULL TURNS (1 turn = 1 user msg + 1 assistant msg)
# to inject into the prompt as conversation context.
TOP_K_TURNS = int(os.environ.get("TOP_K_TURNS", "5"))
