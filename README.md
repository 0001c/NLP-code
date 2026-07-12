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
└── README.md       # 本文件
```

## 🛠️ 环境要求

- Python 3.8+
- 依赖包：`jieba`、`pkuseg`、`numpy`、`scikit-learn`、`gensim`

```bash
pip install jieba pkuseg numpy scikit-learn
```

## 📖 课程链接

[B站课程地址](https://www.bilibili.com/video/BV1KSh1zvEpY/)
