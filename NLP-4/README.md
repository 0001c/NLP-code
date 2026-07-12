# NLP学习 - 第四章代码包

本项目是学习B站课程《【2025版】动手学自然语言处理NLP系列！》第四章「文本表示与相似度」的代码实现。

## 📚 课程来源

- **课程链接**: [【2025版】动手学自然语言处理NLP系列！](https://www.bilibili.com/video/BV1KSh1zvEpY/?p=18&share_source=copy_web&vd_source=697642d055ec82d49734008a241110ad)

- **章节**: 第四章 - 文本表示与相似度

## 📁 项目结构

```
NLP-4/
├── tf-idf.ipynb                  # TF-IDF 实现（分词→词袋模型→TF-IDF计算）
├── BM25.ipynb                    # BM25 算法实现（文本检索与相似度排序）
├── tfidf_similarity_task.ipynb   # TF-IDF 相似度任务（文档检索实践）
├── cos.py                        # 余弦相似度计算工具函数
├── data/
│   ├── stopwords.txt             # 停用词表
│   ├── userdict_mb.txt           # 用户自定义词典（手机领域）
│   ├── phone_dict.txt            # 手机词典
│   ├── mb.txt                    # 文档语料库
│   ├── demo.txt                  # 示例文本
│   ├── analysis_result.txt       # 分析结果
│   ├── tfidf.model               # TF-IDF 模型文件
│   ├── tfidf_model.bin           # TF-IDF 模型（序列化）
│   └── tfidf_model_phone.bin     # 手机领域 TF-IDF 模型
└── README.md                     # 本文件
```

## 📖 内容说明

| 文件 | 说明 |
|------|------|
| `tf-idf.ipynb` | 从分词开始，构建词袋模型，使用 Gensim 计算 TF-IDF 值 |
| `BM25.ipynb` | 基于 `rank_bm25` 库实现 BM25 文本检索与相似度评分 |
| `tfidf_similarity_task.ipynb` | 完整流程：读取文档→构建 TF-IDF 语料→计算输入文本与文档库的相似度 |
| `cos.py` | 手写余弦相似度计算函数 |

## 🛠️ 环境配置

### 安装依赖

```bash
# 创建虚拟环境（可选）
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装依赖包
pip install jieba gensim rank-bm25
```
