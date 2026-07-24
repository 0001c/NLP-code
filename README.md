# NLP 学习

本仓库是学习自然语言处理（NLP）的代码实践项目，基于 B 站课程《【2025版】动手学自然语言处理NLP系列！》。

## 📂 项目结构

```
NLP/
├── NLP-3/          # 第三章 - 中文分词
│   ├── 规则分词.ipynb      # 正向最大匹配、逆向最大匹配等
│   ├── jieba分词.ipynb     # jieba 分词实现
│   ├── pkuseg分词.ipynb    # pkuseg 分词实现
│   └── data/               # 词典与停用词数据
├── NLP-4/          # 第四章 - 文本表示与相似度
│   ├── tf-idf.ipynb        # TF-IDF 实现
│   ├── BM25.ipynb          # BM25 实现
│   ├── tfidf_similarity_task.ipynb  # TF-IDF 相似度任务
│   ├── cos.py              # 余弦相似度计算
│   └── data/               # 实验数据
├── NLP-5/          # 第五章 - 深度学习基础
│   ├── 基础代码.ipynb      # TensorFlow 1.x 基础语法
│   ├── 线性回归.ipynb      # 单变量线性回归
│   ├── 多点线性回归.ipynb  # 多变量线性回归
│   ├── 逻辑回归.ipynb      # 逻辑回归（二分类）
│   └── 数字图片分类.ipynb  # MNIST 手写数字分类
└── README.md       # 本文件
```

## 🛠️ 环境要求

- Python 3.12.10
- 各章节有独立的虚拟环境，以下为各章节所需依赖

### 各章节依赖

| 章节 | 依赖包 |
|------|--------|
| NLP-3（中文分词） | `jieba`、`pkuseg`、`numpy` |
| NLP-4（文本表示与相似度） | `jieba`、`gensim`、`rank-bm25` |
| NLP-5（深度学习基础） | `tensorflow-cpu==2.16.1`、`numpy==1.26.4`、`matplotlib==3.11.1` |

### 安装命令

```bash
# NLP-3
pip install jieba pkuseg numpy

# NLP-4
pip install jieba gensim rank-bm25

# NLP-5（推荐使用虚拟环境）
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Linux/Mac
pip install tensorflow-cpu==2.16.1 numpy==1.26.4 matplotlib==3.11.1
```

## 📖 课程链接

[B站课程地址](https://www.bilibili.com/video/BV1KSh1zvEpY/)
