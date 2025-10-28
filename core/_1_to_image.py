from dataclasses import dataclass
from pathlib import Path

import fitz
from joblib import Parallel, delayed


@dataclass
class PageImage:
    image_bytes: bytes
    width: int
    height: int


def process_page(pdf_path: str | Path, page_num: int):
    """
    处理单个页面的 worker 函数。
    它接收路径和页码（都是可序列化的），然后在子进程中打开文档并提取图像。
    """
    with fitz.open(pdf_path) as doc:
        page = doc[page_num]
        image_xref = page.get_images()[0][0]
        pix = fitz.Pixmap(doc, image_xref)
        img_bytes: bytes = pix.tobytes(output="jpg")
        return img_bytes


def pdf_to_image(pdf_path: str | Path):
    with fitz.open(pdf_path) as doc:
        page_count = len(doc)

    images: list[bytes] = [
        page_image or b""
        for page_image in Parallel(n_jobs=-1)(
            # 并行处理每一页
            delayed(process_page)(pdf_path, i)
            for i in range(page_count)
        )
    ]

    return images


if __name__ == "__main__":
    from core.timer import Timer

    with Timer():
        images = pdf_to_image("test_book_pdf/7301098332霸权之翼：美国国际制度战略.pdf")

    for i, image_bytes in enumerate(images):
        if image_bytes is None:
            continue
        with open(f"temp/page_{i + 1}.jpg", "wb") as f:
            f.write(image_bytes)
