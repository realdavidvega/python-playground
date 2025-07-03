import logging
from pathlib import Path
from typing import List

import pytesseract
from PIL.Image import Image
from pdf2image import convert_from_path

from playground.examples.images_pdf_chat.infrastructure.resources.config import Config

logger = logging.getLogger(__name__)


class ImageProcessor:
    @staticmethod
    def process(config: Config) -> str:
        if not config.USE_MOCK_TEXT:
            logger.info("Converting PDF to images...")
            path: Path = Path(config.FILE_PATH)
            images: List[Image] = convert_from_path(path, dpi=300, thread_count=4)
            logger.info(f"PDF converted to images. Number of images: {len(images)}")

            extracted_text = []
            for i, image in enumerate(images):
                logger.info(f"Processing image: {i}")
                text: str = str(pytesseract.image_to_string(image))

                if text.strip():
                    extracted_text.append(f"--- Page {i + 1} ---\n{text.strip()}")

            return "\n\n".join(extracted_text)

        else:
            logger.info("Using mock text...")
            return "The quick brown fox jumps over the lazy dog."
