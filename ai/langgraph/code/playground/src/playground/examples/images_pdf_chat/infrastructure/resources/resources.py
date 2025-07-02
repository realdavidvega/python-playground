import logging
from pathlib import Path

from playground.examples.images_pdf_chat.infrastructure.image_processor import (
    ImageProcessor,
)
from playground.examples.images_pdf_chat.infrastructure.vector_store import VectorStore
from playground.examples.images_pdf_chat.infrastructure.resources.config import Config

logger = logging.getLogger(__name__)


class Resources:
    def __init__(self, file: str, config: Config) -> None:
        pdf_path: Path = Path(file)
        texts: str = ImageProcessor.process(pdf_path)
        self.vector_store = VectorStore(
            text=texts, model=config.GOOGLE_GENAI_EMBEDDING_MODEL
        )
