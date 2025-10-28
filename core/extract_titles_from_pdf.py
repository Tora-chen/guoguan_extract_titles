from pathlib import Path

import fitz
from core._1_to_image import pdf_to_image
from core._2_get_title_bbox import get_titles_bbox_batch
from core._3_crop_titles import crop_titles
from core._4_extract_titles import VLMTextExtractor
from core._5_insert_to_toc import insert_titles_to_toc_by_page_number
from core.timer import Timer


def extract_titles_from_pdf(pdf: str | Path):
    with Timer():
        # 转图片
        images = pdf_to_image(pdf)

    with Timer():
        # 获取标题区域框
        boxes_list: list[list] = get_titles_bbox_batch(images)

    with Timer():
        # 裁剪标题区域
        cropped_pairs: list[tuple[int, bytes]] = crop_titles(images, boxes_list)
        cropped_images = [pair[1] for pair in cropped_pairs]
        cropped_indexes = [pair[0] for pair in cropped_pairs]

    with Timer():
        # 提取标题文本
        vlm_extractor = VLMTextExtractor()
        titles = vlm_extractor.extract_from_batch(cropped_images)
        titles = list(zip(cropped_indexes, titles))

    doc = fitz.open(pdf)
    toc = doc.get_toc()
    print("Original TOC:")
    print(toc)

    new_toc = insert_titles_to_toc_by_page_number(doc, titles)

    return new_toc

