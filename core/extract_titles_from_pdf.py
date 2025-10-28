from openai import OpenAI
from pathlib import Path

from core._1_to_image import pdf_to_image
from core._2_get_title_bbox import get_titles_bbox_batch
from core._3_crop_titles import crop_titles
from core._4_extract_titles import VLMTextExtractor
from core.timer import Timer

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "sk-0cb46f17675443d583cce0470d4d93df"
MODEL = "qwen-vl-max"


def extract_titles_from_pdf(pdf: str | Path):
    # 初始化 OpenAI 客户端
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )

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

    with Timer():
        # 提取标题文本
        vlm_extractor = VLMTextExtractor(client=client, model=MODEL)
        titles = vlm_extractor.extract_from_batch(cropped_images)

    for i, title in enumerate(titles):
        print(f"Page {i + 1} Titles: {title}")

