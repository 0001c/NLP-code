# NLP学习 - 第三章代码包

本项目是学习B站课程《【2025版】动手学自然语言处理NLP系列！》第三章「中文分词」的代码实现。

## 📚 课程来源

- **课程链接**: [【2025版】动手学自然语言处理NLP系列！](https://www.bilibili.com/video/BV1KSh1zvEpY/?p=18&share_source=copy_web&vd_source=697642d055ec82d49734008a241110ad)

- **章节**: 第三章 - 中文分词

## 📁 项目结构

```
NLP学习/
├── jieba分词.ipynb          # jieba分词实现与练习
├── pkuseg分词.ipynb         # pkuseg分词实现与练习
├── 规则分词.ipynb           # 规则分词（正向最大匹配、逆向最大匹配等）
├── data/
│   ├── stopwords/           # 停用词文件
│   │   ├── stopwords.txt    # 通用中文停用词表
│   │   ├── stopwords_p.txt  # 手机领域停用词
│   │   └── stopwords_phone.txt
│   ├── userdict/            # 用户自定义词典
│   │   ├── userdict.txt     # 通用用户词典
│   │   ├── userdict_p.txt   # 手机领域用户词典
│   │   └── userdict_phone.txt
│   └── word_dict.txt        # 词字典
└── pkuseg_data/
    ├── input.txt            # pkuseg输入文本
    └── output.txt           # pkuseg分词输出
```

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
pip install jieba pkuseg numpy
```

### 注意事项

- pkuseg 安装可能需要先安装 numpy
- 如果 pkuseg 安装失败，可尝试：
  ```bash
  pip install pkuseg --no-build-isolation
  ```

## 🚀 使用说明

### 1. jieba分词

```python
import jieba

# 加载用户词典
jieba.load_userdict("data/userdict/userdict.txt")

# 分词
seg_list = jieba.cut("我来到北京清华大学")
print(" ".join(seg_list))
```

### 2. 去除停用词

```python
import jieba

# 加载停用词
stopwords = [line.strip() for line in open("data/stopwords/stopwords.txt", "r", encoding="utf-8").readlines()]

# 分词并去除停用词
seg_list = jieba.cut("李小福是创新办主任也是云计算方面的专家")
seg_list = [word for word in seg_list if word not in stopwords]
print(" ".join(seg_list))
```

### 3. pkuseg分词

```python
import pkuseg

# 使用默认模型分词
seg = pkuseg.pkuseg()
text = seg.cut("我爱北京天安门")
print(" ".join(text))
```

### 4. 规则分词

包含正向最大匹配、逆向最大匹配、双向最大匹配等算法实现。

## 📝 练习内容

代码包包含以下练习：

1. jieba分词的三种模式（全模式、精确模式、搜索引擎模式）
2. 自定义词典加载与使用
3. 停用词去除
4. 词性标注
5. 手机商品标题分词优化
6. pkuseg分词对比
7. 规则分词算法实现

## 📧 联系方式

如有问题，欢迎交流学习！
