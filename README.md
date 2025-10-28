1   拉取仓库
```
git clone ....
cd guoguan_extract_titles
```

2   创建`test_book_pdf`目录，将 pdf 文件放入该目录下。


3   安装依赖
```
uv sync
```

4   安装 cuda 版本 torch
```
uv pip install torch -U --torch-backend=auto
```

```
uv run python

import torch
torch.cuda.is_available()
```
如果返回 True 则表示安装成功。

5   更改 `core\_4_extract_titles.py` 中的 base_url, api_key 和 model；

6   修改 `main.py` 中的 pdf 文件名和想要运行的函数

7   运行
```
uv run main.py
```

常见问题：
1 cuda out of memory
解决方法：降低 `core\extract_titles_from_pdf.py` 中的 BATCH_SIZE 。

