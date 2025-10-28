from core.extract_titles_from_pdf import extract_titles_from_pdf
from core.extract_text_from_pdf import extract_texts_from_pdf

if __name__ == "__main__":
    pdf_path = "test_book_pdf/7301098332霸权之翼：美国国际制度战略.pdf"

    # print("抽取目录标题...")
    # new_toc = extract_titles_from_pdf(pdf_path)
    # print("New TOC:")
    # print(new_toc)

    print("抽取文本内容...")
    texts = extract_texts_from_pdf(pdf_path, max_pages=10)
    save_path = f"{pdf_path}.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")
    print(texts)
