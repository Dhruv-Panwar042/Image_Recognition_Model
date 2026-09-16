import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Intelligent Image Analysis System"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Google GenAI Key (loads from .env or system environment)
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    # YOLO Settings (checks local folder or standard weights)
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
    DEFAULT_CONFIDENCE: float = 0.25
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    
    # Server host & port
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()
