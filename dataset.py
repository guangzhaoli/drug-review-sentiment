import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer

class DrugReviewDataset(Dataset):
    def __init__(self, file_path, max_length=128, is_test=False):
        """
        初始化数据集
        Args:
            file_path: CSV文件路径
            max_length: 序列最大长度
            is_test: 是否为测试集
        """
        # 读取CSV文件
        self.data = pd.read_csv(file_path)
        
        # 提取评论和评分
        self.reviews = self.data['review'].values
        if not is_test:
            # 将评分转换为情感标签（训练集）
            self.ratings = self.data['rating'].values
            self.labels = self._convert_ratings_to_sentiment(self.ratings)
        
        # 初始化tokenizer (使用bert-base-uncased作为示例)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.is_test = is_test

    def _convert_ratings_to_sentiment(self, ratings):
        """
        将评分转换为情感标签
        1-4: 消极 (0)
        5-6: 中性 (1)
        7-10: 积极 (2)
        """
        sentiments = []
        for rating in ratings:
            if rating <= 4:
                sentiments.append(0)  # 消极
            elif rating <= 6:
                sentiments.append(1)  # 中性
            else:
                sentiments.append(2)  # 积极
        return np.array(sentiments)

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        
        # 对文本进行编码
        encoding = self.tokenizer(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 准备返回数据
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if not self.is_test:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return item

def create_data_loader(dataset, batch_size, shuffle=True):
    """
    创建数据加载器
    """
    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

# 使用示例
def main():
    # 创建训练集
    train_dataset = DrugReviewDataset(
        file_path='review/drugsComTrain_raw.csv',
        max_length=128,
        is_test=False
    )
    
    # 创建测试集
    test_dataset = DrugReviewDataset(
        file_path='review/drugsComTest_raw.csv',
        max_length=128,
        is_test=True
    )
    
    # 创建数据加载器
    train_loader = create_data_loader(train_dataset, batch_size=32)
    test_loader = create_data_loader(test_dataset, batch_size=32, shuffle=False)
    
    print(train_dataset[0])

if __name__ == "__main__":
    main()