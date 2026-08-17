"""Configuration module for Agentic RAG system"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # optional, raises GitHub API rate limit

    # Langfuse (observability/tracing) - optional, tracing is skipped if unset
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # Postgres (structured "scientists" dataset lookup tool) - optional, tool is
    # skipped if the database isn't reachable
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/ragdemo")

    # Model Configuration
    LLM_MODEL = "openai:gpt-4o"

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # PDFs to load by default (directory or single file)
    DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

    
    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        return init_chat_model(cls.LLM_MODEL)