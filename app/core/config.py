from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Heydi_AI"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str
    CONVERSATION_SERVICE_MODE: str = "basic"
    MEMORY_RECENT_DIARY_LIMIT: int = 5
    MEMORY_RECENT_SPECIAL_LIMIT: int = 3
    MEMORY_RECENT_PREFERENCE_LIMIT: int = 3
    MEMORY_RECENT_FACT_LIMIT: int = 5
    MEMORY_DEFAULT_SEARCH_LIMIT: int = 5
    TOOL_CALL_LOG_PATH: str | None = None
    CONVERSATION_LOG_PATH: str | None = None
    FUTURE_REMINDER_SEND_START_PROMPT: bool = False

    class Config:
        env_file = ".env"

settings = Settings()
