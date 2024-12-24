# 药品评论情感分析系统 (Drug Review Sentiment Analysis)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Gradio](https://img.shields.io/badge/Gradio-5.9x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

基于深度学习的药品评论情感分析系统，可以自动分析药品评论的情感倾向（积极、中性、消极）。本项目采用 LSTM + BERT 词向量的混合架构，并提供了友好的 Web 界面。

## 🌟 功能特点

- 支持单条评论和批量评论的情感分析
- 提供直观的 Web 界面，便于使用
- 显示详细的情感概率分布
- 支持 CSV 文件批量处理
- 集成 TensorBoard 可视化训练过程
- 多语言支持

## 🛠️ 技术架构

- **前端界面**: Gradio
- **深度学习框架**: PyTorch
- **预训练模型**: BERT (bert-base-uncased)
- **模型结构**: LSTM + 注意力机制
- **数据处理**: Pandas, NumPy
- **可视化**: TensorBoard

## 📦 安装说明

1. 克隆仓库
```bash
git clone https://github.com/guangzhaoli/drug-review-sentiment.git
cd drug-review-sentiment
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

## 🚀 使用方法

### 训练模型

```bash
python train.py \
    --train_file review/drugsComTrain_raw.csv \
    --val_file review/drugsComTest_raw.csv \
    --model_dir checkpoints \
    --batch_size 32 \
    --epochs 10
```

### 启动 Web 界面

```bash
python app.py
```

### 命令行推理

```bash
# 单条评论分析
python inference.py text \
    --model_path checkpoints/best_model.pth \
    --text "This medicine helped me a lot with minimal side effects."

# 批量分析
python inference.py batch \
    --test_file data/test.csv \
    --model_path checkpoints/best_model.pth \
    --output_file predictions.csv
```

## 📊 模型性能

- 准确率: 85%+
- F1 分数: 0.83
- 支持 3 类情感分析（积极、中性、消极）

## 📁 项目结构

```
drug-review-sentiment/
├── app.py              # Web 界面实现
├── train.py           # 模型训练脚本
├── inference.py       # 推理脚本
├── model.py           # 模型定义
├── dataset.py         # 数据集处理
├── requirements.txt   # 项目依赖
├── checkpoints/       # 模型检查点
└── data/             # 数据目录
```

## 📝 数据格式

训练数据需要包含以下列：
- `review`: 评论文本
- `rating`: 评分 (1-10)

## 🔧 参数配置

主要可调参数：
- `max_length`: 文本最大长度 (默认: 128)
- `embedding_dim`: 词嵌入维度 (默认: 100)
- `lstm_units`: LSTM 单元数 (默认: 64)
- `dropout_rate`: Dropout 比率 (默认: 0.2)
- `learning_rate`: 学习率 (默认: 0.001)

## 📈 可视化

启动 TensorBoard:
```bash
tensorboard --logdir=checkpoints/tensorboard_logs
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来帮助改进项目。

## 📄 开源协议

本项目采用 [MIT 许可证](LICENSE)。

## 👥 作者

- 作者名字 - [@Guangzhao Li](https://github.com/guangzhaoli)

## 🙏 致谢

- BERT 预训练模型
- Gradio 框架
- PyTorch 社区