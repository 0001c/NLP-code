# NLP 学习

本仓库是学习自然语言处理（NLP）的代码实践项目，基于 B 站课程《【2025版】动手学自然语言处理NLP系列！》。包含从中文分词、文本表示、深度学习基础、词向量到文本分类的完整学习路径，每个章节有独立的代码包（notebook）与配套数据。

## 📖 课程来源

[【2025版】动手学自然语言处理NLP系列！](https://www.bilibili.com/video/BV1KSh1zvEpY/)

## 📂 项目结构

```
NLP/
├── NLP-3/          # 第三章 - 中文分词
│   ├── 规则分词.ipynb      # 正向最大匹配、逆向最大匹配等
│   ├── jieba分词.ipynb     # jieba 分词实现
│   ├── pkuseg分词.ipynb    # pkuseg 分词实现
│   ├── data/               # 词典、停用词数据
│   ├── pkuseg_data/        # pkuseg 实验输入输出
│   └── README.md
├── NLP-4/          # 第四章 - 文本表示与相似度
│   ├── tf-idf.ipynb        # TF-IDF 实现
│   ├── BM25.ipynb          # BM25 实现
│   ├── tfidf_similarity_task.ipynb  # TF-IDF 相似度任务
│   ├── cos.py              # 余弦相似度计算
│   ├── data/               # 实验数据与模型
│   └── README.md
├── NLP-5/          # 第五章 - 深度学习基础
│   ├── 基础代码.ipynb      # TensorFlow 1.x 基础语法
│   ├── 线性回归.ipynb      # 单变量线性回归
│   ├── 多点线性回归.ipynb  # 多变量线性回归
│   ├── 逻辑回归.ipynb      # 逻辑回归（二分类）
│   ├── 数字图片分类.ipynb  # MNIST 手写数字分类
│   ├── requirements.txt    # 依赖库清单
│   └── README.md
├── NLP-6/          # 第六章 - 词向量与词嵌入
│   ├── work2vce.ipynb      # Word2Vec 词向量训练与使用
│   ├── glove.ipynb         # GloVe 词向量训练与使用
│   ├── fasttext.ipynb      # FastText 词向量训练与使用
│   ├── glove_simple.py     # 纯 Python 实现的 GloVe 模块
│   ├── data/               # 手机商品语料库（分词/未分词）
│   └── README.md
├── NLP-7/          # 第七章 - 文本分类（CLUE TNEWS）
│   ├── EDA.ipynb           # 探索性数据分析
│   ├── data_process.py     # 数据处理模块（Date_Process 类）
│   ├── data_process.ipynb  # 数据预处理与分词器对比
│   ├── data_process2.ipynb # 数据预处理（pickle 保存/加载版）
│   ├── machine_learning_classification.ipynb  # 机器学习文本分类
│   ├── data/               # TNEWS 数据集（train/dev/test）
│   ├── stopwords/          # 停用词表
│   ├── userdict/           # 自定义词典
│   └── README.md
└── README.md       # 本文件
```

## 🛠️ 环境要求

- **Python 3.12.10**
- 各章节有独立的虚拟环境，以下为各章节所需依赖

### 各章节依赖

| 章节 | 依赖包 |
|------|--------|
| NLP-3（中文分词） | `jieba`、`pkuseg`、`numpy` |
| NLP-4（文本表示与相似度） | `jieba`、`gensim`、`rank-bm25` |
| NLP-5（深度学习基础） | `tensorflow-cpu==2.16.1`、`numpy==1.26.4`、`matplotlib==3.11.1` |
| NLP-6（词向量与词嵌入） | `jieba`、`gensim`、`numpy`、`matplotlib`、`fasttext-wheel` |
| NLP-7（文本分类） | `jieba`、`pkuseg`、`zhconv`、`scikit-learn`、`numpy`、`pandas`、`matplotlib` |

### 安装命令

```bash
# 创建并激活虚拟环境（推荐，以 NLP-5 为例）
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

# NLP-3
pip install jieba pkuseg numpy

# NLP-4
pip install jieba gensim rank-bm25

# NLP-5
pip install tensorflow-cpu==2.16.1 numpy==1.26.4 matplotlib==3.11.1

# NLP-6
pip install jieba gensim numpy matplotlib fasttext-wheel

# NLP-7
pip install jieba pkuseg zhconv scikit-learn numpy pandas matplotlib
```

## 🚀 使用说明

1. 进入对应章节目录（如 `cd NLP-7`）
2. 启动 Jupyter Notebook：

```bash
jupyter notebook
```

3. 依次运行 notebook 中的代码单元格

### 注意事项

- **NLP-5**：代码使用 TensorFlow 1.x 兼容模式（`tf.compat.v1.disable_v2_behavior()`），并在导入处手动映射 `tf.Session`、`tf.placeholder` 等 1.x API 到顶层
- **NLP-6**：Gensim 4.0+ 中词向量维度参数由 `size` 改为 `vector_size`；`glove_python` 库已停止维护，使用配套的纯 Python 模块 `glove_simple.py`；FastText 请安装 `fasttext-wheel`
- **NLP-7**：数据为 JSONL 格式（每行一个 JSON 对象）；`pkuseg` 首次使用需联网下载模型；Windows 中文环境下 sklearn 显示模型对象可能报 `UnicodeDecodeError`，可设置 `PYTHONUTF8=1` 或 `set_config(display='text')`

## 🤝 贡献指南

本项目为个人学习代码实践，欢迎以以下方式参与：

1. **提交 Issue**：发现代码错误或运行问题，请提交 Issue 并附上：
   - 运行环境（Python 版本、依赖版本、操作系统）
   - 完整的错误堆栈信息
   - 复现步骤
2. **提交 PR**：
   - 从 `main` 分支创建新分支，命名建议：`fix/xxx`、`feat/xxx`
   - 保持代码风格与现有 notebook 一致
   - 提交信息使用中文，简要描述改动内容
   - 提交前确认代码可正常运行

## 📄 许可证

- 本项目代码部分仅供**个人学习交流**使用，未附加正式开源许可证
- 课程内容版权归原课程作者所有
- 数据集 [CLUE TNEWS](https://www.CLUEbenchmarks.com) 版权归 CLUE 团队所有，使用请遵循其相关许可协议
