"""
把提取的标题匹配、插入PDF目录。
"""

import re

import fitz


def calculate_ngram_jaccard(s1: str, s2: str, n=2):
    """
    计算基于N-gram的Jaccard相似度。
    相似度 = |N-gram交集| / |N-gram并集|

    Args:
        s1 (str): 第一个字符串。
        s2 (str): 第二个字符串。
        n (int): N-gram中N的值，对于中文，2或3通常效果较好。
    """
    # 预处理：去除所有空格和换行，让比较更关注核心字符
    translation_table = str.maketrans("", "", " \n\t")
    s1_processed = s1.translate(translation_table)
    s2_processed = s2.translate(translation_table)

    # 如果处理后字符串长度小于n，则无法生成n-gram，特殊处理
    if len(s1_processed) < n or len(s2_processed) < n:
        return 1.0 if s1_processed == s2_processed else 0.0

    # 生成N-gram集合
    ngrams1 = set(s1_processed[i : i + n] for i in range(len(s1_processed) - n + 1))
    ngrams2 = set(s2_processed[i : i + n] for i in range(len(s2_processed) - n + 1))

    # 计算交集和并集
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)

    # 计算Jaccard相似度
    similarity = len(intersection) / len(union) if len(union) > 0 else 1.0
    return similarity


def is_similar(s1: str, s2: str, threshold=0.6):
    similarity = calculate_ngram_jaccard(s1, s2)
    return similarity >= threshold


def insert_titles_to_toc_by_page_number(
    doc: fitz.Document | str, ocr_titles: list[tuple[int, str]]
):
    """
    假设目录标题的最高级别为MAX_LEVEL，根据页码，从识别出的标题中，找到目录中缺失的MAX_LEVEL+1级标题，并插入到目录中。
    返回更新后的目录列表。
    遍历目录，找到所有MAX_LEVEL级标题的“范围”，即该级目录所在页码到下一标题所在页码。
    再查看pdf_images中对于哪些标题在该标题的范围之中，插入到目录中。但需要注意排除该标题本身和下一标题。
    """
    if isinstance(doc, str):
        doc = fitz.open(doc)
    toc: list[tuple[int, str, int]] = doc.get_toc()  # 获取当前目录
    new_toc = []
    MAX_LEVEL = max(item[0] for item in toc)  # 假设目录中最高级别的标题为MAX_LEVEL

    for i, (level, toc_title, page_num) in enumerate(toc):
        if (i + 1) < len(toc):
            next_page_num = toc[i + 1][2]
            next_title = toc[i + 1][1]
        else:
            next_page_num = float("inf")
            next_title = ""

        new_toc.append([level, toc_title, page_num])
        if level != MAX_LEVEL:
            continue

        # 在ocr_titles中寻找该范围内的标题
        for index, ocr_title in ocr_titles:
            if not (page_num <= index + 1 < next_page_num):
                continue

            # 排除与当前二级标题和下一个标题相同的标题
            if is_similar(ocr_title, toc_title) or is_similar(ocr_title, next_title):
                continue

            # 排除类似“第二节  国际制度的战略价值”与“第二节”重复的问题
            if (
                re.match(r"^(第[一二三四五六七八九十]+[章节])\s*$", ocr_title)
                and ocr_title in toc_title
            ):
                continue

            # 插入新标题，层级设为当前目录项层级+1
            new_level = level + 1
            new_toc.append([new_level, ocr_title, index + 1])
    return new_toc


if __name__ == "__main__":
    from openai import OpenAI

    from core._1_to_image import pdf_to_image
    from core._2_get_title_bbox import get_titles_bbox_batch
    from core._3_crop_titles import crop_titles
    from core._4_extract_titles import VLMTextExtractor
    from core.timer import Timer

    pdf_path = "test_book_pdf/7301098332霸权之翼：美国国际制度战略.pdf"

    # 初始化 OpenAI 客户端
    client = OpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="xxx",
    )

    with Timer():
        images = pdf_to_image(pdf_path)

    with Timer():
        boxes_list: list[list] = get_titles_bbox_batch(images)

    with Timer():
        cropped_pairs: list[tuple[int, bytes]] = crop_titles(images, boxes_list)
        cropped_images = [pair[1] for pair in cropped_pairs]
        cropped_indexes = [pair[0] for pair in cropped_pairs]

    with Timer():
        vlm_extractor = VLMTextExtractor(client=client, model="qwen-vl-plus")
        titles = vlm_extractor.extract_from_batch(cropped_images)
        titles = list(zip(cropped_indexes, titles))
        for i, title in titles:
            print(f"Page {i + 1} Titles: {title}")

    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    print("Original TOC:")
    print(toc)

    new_toc = insert_titles_to_toc_by_page_number(doc, titles)

    print("Updated TOC:")
    print(new_toc)
