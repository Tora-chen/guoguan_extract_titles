import base64
from abc import ABC, abstractmethod
from typing import List

import openai
from joblib import Parallel, delayed
from openai import OpenAI


class TextExtractor(ABC):
    """
    文字提取器接口定义。
    所有具体的提取器（无论是OCR还是VLM）都应继承此类并实现其方法。
    """

    @abstractmethod
    def extract_from_single_image(self, image: bytes) -> str:
        """
        从单张图片中提取文字。
        这是所有子类必须实现的核心方法。
        """
        raise NotImplementedError

    def extract_from_batch(self, images: List[bytes]) -> List[str]:
        """
        从一个批次的图片中提取文字。

        提供一个默认的循环实现，如果后端支持真正的批量处理，
        子类应该重写此方法以获得更好的性能。
        """
        print("使用默认的循环方式进行批量处理...")
        results = []
        for image in images:
            results.append(self.extract_from_single_image(image))
        return results


class VLMTextExtractor(TextExtractor):
    """
    基于视觉语言模型（VLM）的标题提取器。
    使用 OpenAI API 和 joblib 进行并行处理。
    """

    def __init__(self, n_jobs: int = -1):
        self.client = OpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="sk-xxxxxxxxxxxx",
        )
        self.model = "qwen-vl-max"
        self.n_jobs = n_jobs
        self.prompt = "你是一个 OCR 模型。提取图中文字。不要输出任何额外字符。"

    def _image_bytes_to_jpg_base64(self, image_bytes: bytes) -> str:
        base64_string = base64.b64encode(image_bytes).decode("utf-8")
        return base64_string

    def extract_from_single_image(self, image: bytes) -> str:
        """
        从单张图片中提取文字。
        """
        base64_image = self._image_bytes_to_jpg_base64(image)

        messages: list = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages
            )
            return response.choices[0].message.content or ""
        except openai.OpenAIError as e:
            print(f"OpenAI API 调用失败: {e}")
            return ""
        except Exception as e:
            print(f"未知错误: {e}")
            return ""

    def extract_from_batch(self, images: List[bytes]) -> List[str]:
        return [
            result or ""
            for result in Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(self.extract_from_single_image)(image) for image in images
            )
        ]


if __name__ == "__main__":
    from openai import OpenAI

    from core._1_to_image import pdf_to_image
    from core._2_get_title_bbox import get_titles_bbox_batch
    from core._3_crop_titles import crop_titles
    from core.timer import Timer

    with Timer():
        images = pdf_to_image("test_book_pdf/7301098332霸权之翼：美国国际制度战略.pdf")
        images = images[:50]

    with Timer():
        boxes_list: list[list] = get_titles_bbox_batch(images)

    with Timer():
        cropped_pairs: list[tuple[int, bytes]] = crop_titles(images, boxes_list)
        cropped_images = [pair[1] for pair in cropped_pairs]
        cropped_indexes = [pair[0] for pair in cropped_pairs]

    with Timer():
        vlm_extractor = VLMTextExtractor()
        titles = vlm_extractor.extract_from_batch(cropped_images)

    for i, title in enumerate(titles):
        print(f"Page {i + 1} Titles: {title}")
