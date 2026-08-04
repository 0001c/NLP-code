# NLP学习 - 第七章代码包

本项目是学习B站课程《【2025版】动手学自然语言处理NLP系列！》第七章「文本分类」的代码实现，基于 CLUE 中文任务基准的 TNEWS（今日头条新闻短文本分类）数据集，完成从数据预处理、探索性数据分析（EDA）到机器学习分类模型训练与评估的完整流程。

## 📚 课程来源

- **课程链接**: [B站课程地址](https://www.bilibili.com/video/BV1KSh1zvEpY/)
- **章节**: 第七章 - 文本分类（机器学习）
- **数据集**: [CLUE TNEWS](https://www.CLUEbenchmarks.com) 今日头条中文新闻短文本分类数据集

## 📁 项目结构

```
NLP-7/
├── data/
│   ├── train.jsonl            # 训练集（每行一个 JSON 对象：sentence / label_desc）
│   ├── dev.jsonl              # 验证集
│   ├── test.jsonl             # 测试集（1.1 版本，用于预测提交）
│   ├── test1.0.jsonl          # 旧版测试集（1.0，仅供测试或备份）
│   ├── labels.json            # 标签映射（label 编号 → label_desc 类别名）
│   ├── news_headlines.txt     # 自制头条语料（53 条，格式：标题!--!--标签）
│   └── README.txt             # CLUE 数据集官方说明
├── stopwords/
│   └── stopwords.txt          # 停用词表
├── userdict/
│   └── userdict.txt           # 自定义词典（领域专有词）
├── data_process.py            # 数据处理模块（Date_Process 类）
├── EDA.ipynb                  # 探索性数据分析
├── data_process.ipynb         # 数据预处理与分词器对比
├── data_process2.ipynb        # 数据预处理（pickle 保存/加载版）
├── machine_learning_classification.ipynb  # 机器学习文本分类
└── README.md                  # 本文件
```

## 📖 内容说明

| 文件 | 说明 |
| ---- | ---- |
| `data_process.py` | 数据处理核心模块，提供 `Date_Process` 类：数据读取、分词、清洗、停用词过滤、词表/标签映射构建、训练测试集划分（8:2）、pickle 保存与加载 |
| `EDA.ipynb` | 探索性数据分析：句子长度分布统计（`describe()`）、标签类别分布可视化、中文乱码处理等 |
| `data_process.ipynb` | 数据预处理流程演示，对比 jieba / pkuseg / 单字切分三种分词器效果 |
| `data_process2.ipynb` | 数据预处理升级版，使用 `save_model()` / `load_model()` 将预处理结果保存为 pickle 文件，避免重复分词 |
| `machine_learning_classification.ipynb` | 机器学习文本分类：TF-IDF 特征提取 + 多种分类器（SVC、SGDClassifier、GradientBoostingClassifier）训练与评估 |

### 标签类别（15 类）

| 编号 | 类别 | 编号 | 类别 |
| ---- | ---- | ---- | ---- |
| 100 | news_story（故事） | 108 | news_edu（教育） |
| 101 | news_culture（文化） | 109 | news_tech（科技） |
| 102 | news_entertainment（娱乐） | 110 | news_military（军事） |
| 103 | news_sports（体育） | 112 | news_travel（旅游） |
| 104 | news_finance（财经） | 113 | news_world（国际） |
| 106 | news_house（房产） | 114 | news_stock（股市） |
| 107 | news_car（汽车） | 115 | news_agriculture（农业） |
| | | 116 | news_game（游戏） |

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
pip install jieba pkuseg zhconv scikit-learn numpy pandas matplotlib
```

### 依赖库

- `jieba` - 中文分词（支持自定义词典）
- `pkuseg` - 北大中文分词工具（首次运行需下载模型）
- `zhconv` - 繁体中文转简体中文
- `scikit-learn` - 机器学习分类（TF-IDF、SVC、SGD、GBDT 等）
- `numpy` / `pandas` - 数值计算与数据处理
- `matplotlib` - 数据可视化

### 注意事项

- 数据文件为 **JSONL 格式**（每行一个 JSON 对象），Python 需用 `json.loads()` 逐行读取；VS Code 中若将其当作普通 JSON 校验会报"预期为文件结尾"，属正常现象
- `pkuseg` 首次使用时需联网下载模型文件，如遇网络问题可手动下载放入 `~/.pkuseg/` 目录
- Windows 中文环境下 sklearn 显示模型对象时可能出现 `UnicodeDecodeError`（HTML 图表读取 JS 文件编码问题），可在代码中添加 `from sklearn import set_config; set_config(display='text')`，或设置环境变量 `PYTHONUTF8=1`
- 测试集提交文件命名统一为 `tnews_predict.json`（详见 `data/README.txt`）

## 🚀 使用说明

### 1. 数据预处理

```python
from data_process import Date_Process

file_path = "data/train.jsonl"
tokenizer_name = "jieba"                    # jieba / pkuseg / single
userdict_path = "userdict/userdict.txt"
stopwords_path = "stopwords/stopwords.txt"
# [繁体转简体, 小写化, 去除空格, 全角转半角]
cleaning_parameters = [True, True, True, True]

data_p = Date_Process(file_path, tokenizer_name, userdict_path, stopwords_path, cleaning_parameters)
data_p.init()

# 保存预处理结果，避免重复分词
data_p.save_model("data_process_result.pkl")

# 重新加载
data_p.load_model("data_process_result.pkl")
print(len(data_p.train_data), len(data_p.test_data))
```

### 2. 机器学习分类

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 将分词结果还原为空格分隔文本，提取 TF-IDF 特征
train_texts = [" ".join(words) for words, _ in data_p.train_data]
test_texts  = [" ".join(words) for words, _ in data_p.test_data]

tfidf_vec = TfidfVectorizer(max_df=0.5, min_df=2, ngram_range=(1, 2))
train_tfidf = tfidf_vec.fit_transform(train_texts)
test_tfidf  = tfidf_vec.transform(test_texts)

# 训练与评估
clf = SVC()
clf.fit(train_tfidf, train_labels)
pred = clf.predict(test_tfidf)
print(accuracy_score(test_labels, pred))
print(classification_report(test_labels, pred))
```

## 📝 练习内容

代码包包含以下练习：

1. TNEWS 数据集读取与 JSONL 格式解析
2. 三种分词器（jieba / pkuseg / 单字切分）对比实验
3. 数据清洗：繁简转换、全角转半角、小写化、去空格
4. EDA 探索性数据分析：句子长度分布、标签分布可视化
5. TF-IDF 特征提取参数调优（max_df、min_df、ngram_range）
6. 多种分类器对比：SVC、SGDClassifier、GradientBoostingClassifier
7. 模型评估指标：accuracy、precision、recall、f1-score
