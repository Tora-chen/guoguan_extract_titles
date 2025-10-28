import io
from typing import List

from PIL import Image

from core._2_get_title_bbox import FloatBBox


def crop_titles(
    page_images: List[bytes],
    norm_bboxes: List[List[FloatBBox]],
) -> List[tuple[int, bytes]]:
    if len(page_images) != len(norm_bboxes):
        raise ValueError("页图片数量与边界框列表数量不匹配")

    norm_bboxes_flat: list[tuple[int, FloatBBox]] = [
        (i, bbox) for i, bboxes in enumerate(norm_bboxes) for bbox in bboxes
    ]

    cropped_images: list[tuple[int, bytes]] = []
    for index, bbox in norm_bboxes_flat:
        image_bytes = page_images[index]
        img = Image.open(io.BytesIO(image_bytes))

        page_width, page_height = img.size
        # 归一化坐标转像素

        x0 = int(bbox[0] * page_width - 5)
        y0 = int(bbox[1] * page_height - 5)
        x1 = int(bbox[2] * page_width + 5)
        y1 = int(bbox[3] * page_height + 5)
        bbox = (x0, y0, x1, y1)

        cropped_img = img.crop(bbox)

        try:
            with io.BytesIO() as buffer:
                cropped_img.save(buffer, format="jpeg")
                png_bytes = buffer.getvalue()
            cropped_images.append((index, png_bytes))
        except Exception as e:
            print(f"裁剪并保存图片失败: {e}")
            continue

    return cropped_images


if __name__ == "__main__":
    from core._1_to_image import pdf_to_image
    from core._2_get_title_bbox import get_titles_bbox_batch

    pdf_file = "test_book_pdf/7301098332霸权之翼：美国国际制度战略.pdf"

    imgs: list[bytes] = pdf_to_image(pdf_file)

    boxes_list: list = get_titles_bbox_batch(imgs)

    cropped_images: list[tuple[int, bytes]] = crop_titles(imgs, boxes_list)

    for i, (page_index, img_bytes) in enumerate(cropped_images):
        with open(f"temp/cropped_page_{page_index + 1}_title_{i + 1}.jpg", "wb") as f:
            f.write(img_bytes)
