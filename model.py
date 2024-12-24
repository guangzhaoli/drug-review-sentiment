import torch
import torch.nn as nn

class SentimentModel(nn.Module):
    """情感分析模型类"""
    def __init__(self, config):
        super(SentimentModel, self).__init__()
        self.config = config
        # 词嵌入层
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=0  # 假设0为padding token的索引
        )
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.lstm_units,
            batch_first=True  # 输入形状为 (batch, seq, feature)
        )# Dropout层
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # 全连接输出层
        self.fc = nn.Linear(
            in_features=config.lstm_units,
            out_features=config.num_classes
        )
        
        # Softmax层
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, attention_mask=None):
        """
        前向传播
        Args:
            x: 输入张量，形状为 (batch_size, seq_length)
            attention_mask: 注意力掩码，形状与x相同
        """
        # 词嵌入
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # 如果提供了attention mask，处理填充部分
        if attention_mask is not None:
            # 扩展维度以匹配embedded的形状
            attention_mask = attention_mask.unsqueeze(-1)
            embedded = embedded * attention_mask
        
        # LSTM处理
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # 我们只使用最后一个时间步的输出
        
        # Dropout
        dropped = self.dropout(hidden[-1])
        
        # 全连接层
        logits = self.fc(dropped)# Softmax
        output = self.softmax(logits)
        
        return output

def create_model(config):
    """
    创建模型实例
    """
    model = SentimentModel(config)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate
    )
    
    return model, criterion, optimizer

# 配置类示例
class ModelConfig:
    def __init__(self):
        self.vocab_size = 10000  # 词汇表大小
        self.embedding_dim = 100  # 词嵌入维度
        self.lstm_units = 64     # LSTM单元数
        self.dropout_rate = 0.2  # Dropout比率
        self.num_classes = 3     # 类别数（消极、中性、积极）
        self.learning_rate = 1e-3  # 学习率
        self.max_len = 128       # 序列最大长度