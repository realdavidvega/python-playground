import logging
from dataclasses import dataclass
from logging import Logger
from typing import Self

from playground.examples.images_pdf_chat.infrastructure.image_processor import (
    ImageProcessor,
)
from playground.examples.images_pdf_chat.infrastructure.vector_store import VectorStore
from playground.examples.images_pdf_chat.infrastructure.resources.config import (
    Config,
)

logger: Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Resources:
    vector_store: VectorStore

    @classmethod
    def load(cls, config: Config) -> Self:
        logger.info("Loading resources...")
        texts: str = ImageProcessor.process(config)
        vector_store: VectorStore = VectorStore.create(
            text=texts, model=config.GOOGLE_GENAI_EMBEDDING_MODEL
        )
        return cls(vector_store=vector_store)
