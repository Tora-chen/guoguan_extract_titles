from pathlib import Path

from core._1_to_image import pdf_to_image
from core._4_extract_titles import VLMTextExtractor
from core.timer import Timer

BATCH_SIZE = 50 # 控制并发请求数量

def extract_texts_from_pdf(pdf: str | Path, max_pages: int = 0):
    with Timer():
        images = pdf_to_image(pdf)
        if max_pages > 0:
            images = images[:max_pages]

    with Timer():
        vlm_extractor = VLMTextExtractor(n_jobs=BATCH_SIZE)
        texts = vlm_extractor.extract_from_batch(images)

    return texts

