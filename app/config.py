from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    app_name: str = "MLPilot"
    app_version : str = "0.1.0"
    llm_provider: str = "groq"
    groq_api_key : str|None = None
    database_url : str = "sqlite:///./mlpilot.db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()