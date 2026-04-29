import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    MODEL_NAME = os.getenv("MODEL_NAME")

    TOP_K = int(os.getenv("TOP_K", "4"))

    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))

    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "data/chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "portfolio")

    @classmethod
    def get_llm_params(cls) -> Dict[str, Any]:
        return {
            "temperature": cls.LLM_TEMPERATURE,
            "max_output_tokens": cls.MAX_TOKENS,
        }

    @classmethod
    def is_valid(cls) -> bool:
        return bool(cls.GOOGLE_API_KEY)
