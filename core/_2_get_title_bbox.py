import io

import torch  # noqa: F401
from doclayout_yolo import YOLOv10
from doclayout_yolo.engine.results import Results
from PIL import Image

from core.timer import Timer

DEVICE = "cuda:0"  # "cpu" or "cuda:0" or "npu:0"
BATCH_SIZE = 10  # 随显存增大可适当增大，一张910B可调至40以上

if DEVICE.startswith("npu"):
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401


model = YOLOv10("core/doclayout_yolo_docstructbench_imgsz1024.pt")
FloatBBox = tuple[float, float, float, float]  # x0, y0, x1, y1


def get_titles_bbox_batch(imgs: list[bytes], imgsz=1024, conf=0.8):
    # 按 BATCH_SIZE 分批处理
    pil_imgs = [Image.open(io.BytesIO(img_bytes)) for img_bytes in imgs]
    batch_imgs = [
        pil_imgs[i : i + BATCH_SIZE] for i in range(0, len(pil_imgs), BATCH_SIZE)
    ]

    results: list[list[FloatBBox]] = []
    for i, batch in enumerate(batch_imgs):
        det_res: list[Results] = model.predict(
            source=batch,  # Images to predict
            imgsz=imgsz,  # Prediction image size
            conf=conf,  # Confidence threshold
            device=DEVICE,  # Device to use (e.g., 'cuda:0' or 'cpu')
        )

        for res in det_res:
            img_boxes: list[FloatBBox] = []
            assert res.boxes is not None
            xyxyns = res.boxes.xyxyn.tolist()  # type: ignore
            labels = res.boxes.cls.tolist()

            for i in range(len(labels)):
                if labels[i] != 0:  # Only keep title boxes
                    continue
                box: FloatBBox = (
                    xyxyns[i][0],
                    xyxyns[i][1],
                    xyxyns[i][2],
                    xyxyns[i][3],
                )
                img_boxes.append(box)
            results.append(img_boxes)
    return results


if __name__ == "__main__":
    from core._1_to_image import pdf_to_image
    from core.timer import Timer

    pdf_path = "./test_book_pdf/7301098332霸权之翼：美国国际制度战略.pdf"
    images = pdf_to_image(pdf_path)

    with Timer():
        boxes_list: list[list[FloatBBox]] = get_titles_bbox_batch(images[0:100])

    print(boxes_list[0:20])
