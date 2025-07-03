from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class FrozenBaseSettings(BaseSettings):
    model_config = ConfigDict(frozen=True)


class Config(FrozenBaseSettings):
    # Available models
    GOOGLE_GENAI_MODEL: str = "google_genai:gemini-2.0-flash"
    GOOGLE_GENAI_EMBEDDING_MODEL: str = "models/text-embedding-004"

    # Model settings
    USE_OPENAI: bool = True
    TEMPERATURE: float = 0.0

    # PDF settings
    FILE_PATH: str = ""

    # Mocks
    USE_MOCK_TEXT: bool = True
