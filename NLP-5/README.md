# NLP学习 - 第五章代码包

本项目是学习B站课程《【2025版】动手学自然语言处理NLP系列！》第五章「深度学习基础」的代码实现。

## 📚 课程来源

- **课程链接**: [5.1 机器学习基本问题\_哔哩哔哩\_bilibili](https://www.bilibili.com/video/BV1KSh1zvEpY?share_source=copy_web\&vd_source=697642d055ec82d49734008a241110ad\&spm_id_from=333.788.videopod.episodes\&p=24)
- **章节**: 第五章 - 深度学习基础（TensorFlow 入门）

## 📁 项目结构

```
NLP-5/
├── 基础代码.ipynb          # TensorFlow 1.x 基础语法与兼容模式配置
├── 线性回归.ipynb          # 单变量线性回归（梯度下降实现）
├── 多点线性回归.ipynb      # 多变量线性回归
├── 逻辑回归.ipynb          # 逻辑回归（二分类）
├── 数字图片分类.ipynb      # MNIST 手写数字图片分类
├── requirements.txt        # 依赖库清单
└── README.md               # 本文件
```

## 📖 内容说明

| 文件             | 说明                                                                        |
| -------------- | ------------------------------------------------------------------------- |
| `基础代码.ipynb`   | TensorFlow 1.x 兼容模式配置（`tf.compat.v1.disable_v2_behavior()`），常量、变量、会话等基础操作 |
| `线性回归.ipynb`   | 使用 TensorFlow 1.x 实现单变量线性回归模型                                             |
| `多点线性回归.ipynb` | 多特征线性回归，矩阵运算实现                                                            |
| `逻辑回归.ipynb`   | 使用 Sigmoid 函数实现二分类逻辑回归                                                    |
| `数字图片分类.ipynb` | MNIST 数据集分类（0/1 二分类），包含数据预处理、梯度下降训练、准确率评估                                 |

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
pip install -r requirements.txt
```

### 依赖库

- `tensorflow==2.16.1` - 深度学习框架（使用 1.x 兼容模式）
- `numpy==1.26.4` - 数值计算
- `matplotlib==3.11.1` - 数据可视化

### 注意事项

- 本代码使用 TensorFlow **1.x 兼容模式**（`tf.compat.v1.disable_v2_behavior()`），所有代码风格为 TF 1.x 风格（Session、placeholder 等）
- MNIST 数据集需从 Google 服务器下载，如遇网络问题可手动下载 `mnist.npz` 放入 `~/.keras/datasets/` 目录

## 🚀 使用说明

### 1. 启动 Jupyter

```bash
jupyter notebook
```

### 2. 运行示例

依次运行 notebook 中的代码单元格即可。

## 📝 练习内容

代码包包含以下练习：

1. TensorFlow 1.x 基础语法（常量、变量、占位符、会话）
2. 线性回归模型构建与训练
3. 多变量线性回归
4. 逻辑回归（二分类）
5. MNIST 手写数字分类实践

## 📧 联系方式

如有问题，欢迎交流学习！
