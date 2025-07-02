import logging
from pathlib import Path
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from playground.examples.images_pdf_chat.infrastructure.image_processor import (
    ImageProcessor,
)
from playground.examples.images_pdf_chat.infrastructure.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(config=ConfigDict(frozen=True))
class Resources:
    def __init__(self, file_name: str) -> None:
        self.vector_store = self.__init_vector_store(file_name)

    @staticmethod
    def __init_vector_store(file_name: str) -> VectorStore:
        pdf_path: Path = Path(__file__).parent / f"{file_name}.pdf"
        texts: str = ImageProcessor.process(pdf_path)
        return VectorStore(text=texts, model="google_genai:gemini-2.0-flash")
