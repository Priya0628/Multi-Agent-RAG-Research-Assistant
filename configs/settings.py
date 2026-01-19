"""
Configuration Management Using Pydantic Settings
=================================================
Loads environment variables from .env file for OpenAI API access.
Supports both OpenAI (paid) and Ollama (free) LLM backends.

Features:
- Type-safe configuration with validation
- Automatic .env file loading
- Clear error messages for missing variables
- Support for multiple LLM backends
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with validation"""
    
    # OpenAI settings (default backend)
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    
    # Local embedding model (FREE - runs on your machine)
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    
    # Vector database storage path
    chroma_dir: str = Field(default="vectorstore", alias="CHROMA_DIR")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton instance - import this in other modules
settings = Settings()
