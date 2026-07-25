"""
纯 Python 实现的 GloVe 模块，兼容 glove_python 的 API。
适用于 Python 3.12（glove_python 因 C 扩展不兼容无法安装）。

用法：
    from glove_simple import Glove, Corpus
"""

import numpy as np
from collections import defaultdict, OrderedDict
import itertools
import math


class Corpus:
    """构建共现矩阵 (Co-occurrence Matrix)"""

    def __init__(self, dictionary=None):
        self.dictionary = dictionary  # {word: id}
        self.matrix = None           # 共现矩阵

    def fit(self, corpus, window=10, ignore_missing=False):
        """
        corpus: [[word1, word2, ...], ...]
        window: 窗口大小
        ignore_missing: 如果 dictionary 已指定，遇到 OOV 是否忽略
        """
        if self.dictionary is None:
            # 自动构建词典
            vocab = OrderedDict()
            for sentence in corpus:
                for word in sentence:
                    if word not in vocab:
                        vocab[word] = len(vocab)
            self.dictionary = vocab

        word2id = self.dictionary
        vocab_size = len(word2id)
        cooc = defaultdict(float)

        for sentence in corpus:
            sent_ids = []
            for word in sentence:
                if word in word2id:
                    sent_ids.append(word2id[word])
                elif not ignore_missing:
                    raise KeyError(f"Word '{word}' not in dictionary")
                # 如果 ignore_missing=True 且词不在词典中，就跳过
            for i, center in enumerate(sent_ids):
                # 窗口内的上下文词
                left = max(0, i - window)
                right = min(len(sent_ids), i + window + 1)
                for j in range(left, right):
                    if i == j:
                        continue
                    context = sent_ids[j]
                    # 距离权重: 1 / distance
                    distance = abs(i - j)
                    cooc[(center, context)] += 1.0 / distance

        # 转换为稠密矩阵 (glove_python 使用稠密矩阵)
        self.matrix = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        for (i, j), val in cooc.items():
            self.matrix[i, j] = val

    def save(self, path):
        """保存 Corpus 模型"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """加载 Corpus 模型"""
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)


class Glove:
    """GloVe 词向量训练模型"""

    def __init__(self, no_components=30, learning_rate=0.05,
                 alpha=0.75, max_count=100, max_loss=10.0,
                 random_state=None):
        self.no_components = no_components
        self.learning_rate = learning_rate
        self.alpha = alpha          # 权重函数的指数
        self.max_count = max_count  # 权重函数的截断值
        self.max_loss = max_loss    # 梯度裁剪
        self.random_state = random_state

        self.word_vectors = None        # 主词向量
        self.context_vectors = None     # 上下文词偏置
        self.biases = None              # 主词偏置
        self.context_biases = None      # 上下文词偏置
        self.cost = None                # 训练损失
        self.dictionary = None          # {word: id}，用于词查询

    def add_dictionary(self, dictionary):
        """添加词到 id 的映射，用于后续通过词名查询向量"""
        self.dictionary = dictionary

    def _weight(self, count):
        """GloVe 权重函数 f(x) = (x/x_max)^alpha if x < x_max else 1"""
        if count < self.max_count:
            return (count / self.max_count) ** self.alpha
        return 1.0

    def fit(self, matrix, epochs=5, no_threads=2, verbose=False):
        """
        训练 GloVe 模型

        matrix: 共现矩阵 (vocab_size x vocab_size)
        epochs: 训练轮数
        no_threads: 线程数（纯 Python 实现，此参数无效，仅用于 API 兼容）
        verbose: 是否打印损失
        """
        vocab_size = matrix.shape[0]
        dim = self.no_components

        rng = np.random.RandomState(self.random_state)

        # 初始化参数
        self.word_vectors = (rng.rand(vocab_size, dim) - 0.5) / dim
        self.context_vectors = (rng.rand(vocab_size, dim) - 0.5) / dim
        self.biases = np.zeros(vocab_size)
        self.context_biases = np.zeros(vocab_size)

        # 提取非零共现对
        rows, cols = np.nonzero(matrix)
        cooc_values = np.array([matrix[i, j] for i, j in zip(rows, cols)])
        weights = np.array([self._weight(v) for v in cooc_values])

        total_pairs = len(rows)
        lr = self.learning_rate

        for epoch in range(epochs):
            # 打乱训练顺序
            indices = np.arange(total_pairs)
            rng.shuffle(indices)

            total_loss = 0.0
            for idx in indices:
                i, j = rows[idx], cols[idx]
                val = cooc_values[idx]
                w = weights[idx]

                # 预测值: w_i · w_j + b_i + b_j
                dot = np.dot(self.word_vectors[i], self.context_vectors[j])
                pred = dot + self.biases[i] + self.context_biases[j]

                # 损失梯度: f(x_ij) * (pred - log(x_ij))
                diff = pred - math.log(val)
                grad = w * diff

                # 梯度裁剪
                if abs(grad) > self.max_loss:
                    grad = self.max_loss if grad > 0 else -self.max_loss

                # 更新
                self.word_vectors[i] -= lr * grad * self.context_vectors[j]
                self.context_vectors[j] -= lr * grad * self.word_vectors[i]
                self.biases[i] -= lr * grad
                self.context_biases[j] -= lr * grad

                total_loss += 0.5 * w * diff * diff

            avg_loss = total_loss / total_pairs
            self.cost = avg_loss
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, loss = {avg_loss:.6f}")

    def save(self, path):
        """保存模型"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """加载模型"""
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
