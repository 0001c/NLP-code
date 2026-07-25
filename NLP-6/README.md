# NLP学习 - 第六章代码包

本项目是学习B站课程《【2025版】动手学自然语言处理NLP系列！》第六章「词向量与词嵌入」的代码实现。

## 课程来源

- **课程链接**:  [6.1 word2vec\_哔哩哔哩\_bilibili](https://www.bilibili.com/video/BV1KSh1zvEpY?share_source=copy_web\&vd_source=697642d055ec82d49734008a241110ad\&p=35\&spm_id_from=333.788.videopod.episodes)
- **章节**: 第六章 - 词向量与词嵌入

## 项目结构

```
NLP-6/
├── data/
│   ├── mb.txt                    # 原始手机商品语料库（500条）
│   ├── mb_train.txt              # 分词后语料（供 LineSentence 使用）
│   ├── phone_dict.txt            # 自定义词典（手机领域专有词）
│   └── stopwords.txt             # 停用词表
├── models/
│   ├── corpus_model.pkl          # GloVe 共现矩阵模型
│   ├── glove_model.pkl           # GloVe 词向量模型
│   ├── mb_word2vec.model         # Word2Vec 词向量模型
│   └── mb_fasttext.model         # FastText 词向量模型
├── glove_simple.py               # 纯 Python 实现的 GloVe 模块（兼容 Python 3.12）
├── work2vce.ipynb                # Word2Vec 词向量训练与使用
├── glove.ipynb                   # GloVe 词向量训练与使用
└── fasttext.ipynb                # FastText 词向量训练与使用
```

## 内容说明

| 文件                | 说明                                                                              |
| ----------------- | ------------------------------------------------------------------------------- |
| `work2vce.ipynb`  | 使用 Gensim 训练 Word2Vec 词向量（Skip-gram + Hierarchical Softmax），包含内存方式和文件方式两种语料读取方式 |
| `glove.ipynb`     | 使用 GloVe 算法训练词向量，包含共现矩阵构建与词向量训练                                                 |
| `fasttext.ipynb`  | 使用 FastText 训练词向量，支持 subword 信息，可获取 OOV 词向量                                     |
| `glove_simple.py` | 纯 Python 实现的 GloVe 模块，提供 `Corpus` 和 `Glove` 类，等价于 `glove_python` 库的 API         |

## 环境配置

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
pip install jieba gensim numpy matplotlib fasttext-wheel
```

### 注意事项

- **Gensim**: 4.0+ 版本中词向量维度参数由 `size` 改为 `vector_size`，获取词向量改用 `model.wv['word']`
- **GloVe**: `glove_python` 库已停止维护，Python 3.12 下无法安装。本项目中提供了纯 Python 实现的替代模块 `glove_simple.py`，API 完全兼容
- **FastText**: 安装 `fasttext-wheel` 而非 `fasttext`，后者在 Windows 上兼容性较差
- **Python**: 推荐 Python 3.9+

## 使用说明

### 1. Word2Vec

```python
from gensim.models import Word2Vec

# 方式一：内存方式（使用已分词的列表）
sentences = [["手机", "5G", "快充"], ["4G", "双卡双待"]]
model = Word2Vec(sentences, sg=1, hs=1, window=5, min_count=1, vector_size=200)

# 方式二：文件方式（已分词的文件，每行一篇文档）
from gensim.models import word2vec
sentences = word2vec.LineSentence("data/mb_train.txt")
model = Word2Vec(sentences, sg=1, hs=1, window=5, min_count=1, vector_size=200)

# 保存与加载
model.save("models/mb_word2vec.model")
model = Word2Vec.load("models/mb_word2vec.model")

# 获取词向量
print(model.wv["5G"])

# 计算相似度
print(model.wv.similarity("5G", "4G"))
```

### 2. GloVe

```python
from glove_simple import Corpus, Glove

# 构建共现矩阵
corpus_model = Corpus(dictionary=None)
corpus_model.fit(sentences, window=5)

# 训练词向量
glove = Glove(no_components=200, learning_rate=0.05)
glove.fit(corpus_model.matrix, epochs=10, no_threads=1, verbose=True)
glove.add_dictionary(corpus_model.dictionary)

# 保存与加载
glove.save("models/glove_model.pkl")
corpus_model.save("models/corpus_model.pkl")
```

### 3. FastText

```python
import fasttext

# 训练（直接使用已分词文件）
model = fasttext.train_unsupervised("data/mb_train.txt", model="skipgram", dim=200, ws=5)

# 保存与加载
model.save_model("models/mb_fasttext.model")
model = fasttext.load_model("models/mb_fasttext.model")

# 获取词向量（支持 OOV 词）
print(model.get_word_vector("安卓"))
print(model.get_word_vector("nihao"))  # 训练语料中不存在的词也能获得向量
```

## 练习内容

代码包包含以下练习：

1. Word2Vec 的 Skip-gram 与 CBOW 两种训练模式对比
2. Gensim 的 LineSentence 文件式读取 vs 内存式读取
3. GloVe 共现矩阵的构建原理
4. FastText 对 OOV（未登录词）的处理能力
5. 三种词向量模型的保存与加载方式
6. 词向量相似度计算（语义相似度分析）

## 词向量对比

| 模型       | 特点            | 优势                | 劣势         |
| -------- | ------------- | ----------------- | ---------- |
| Word2Vec | 基于预测的词嵌入      | 训练速度快，语义相似度好      | 无法处理 OOV 词 |
| GloVe    | 基于全局共现统计      | 融合了全局统计信息         | 需要先构建共现矩阵  |
| FastText | 基于 subword 信息 | 可处理 OOV 词，形态学信息丰富 | 模型文件较大     |

