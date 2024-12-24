import argparse
import torch
import logging
from torch.utils.data import DataLoader
from dataset import DrugReviewDataset, create_data_loader
from model import SentimentModel, ModelConfig
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description='Train the drug review sentiment model')
    parser.add_argument('--train_file', type=str, required=True, help='Path to training data file')
    parser.add_argument('--val_file', type=str, help='Path to validation data file')
    parser.add_argument('--model_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--embedding_dim', type=int, default=100, help='Dimension of word embeddings')
    parser.add_argument('--lstm_units', type=int, default=64, help='Number of LSTM units')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--device', type=str, default='cpu' if torch.cuda.is_available() else 'cpu',help='Device to use for training (cuda/cpu)')
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )

def train(args):
    # 设置设备
    device = torch.device(args.device)
    logging.info(f'Using device: {device}')

    # 加载数据
    train_dataset = DrugReviewDataset(args.train_file, max_length=args.max_length)
    train_loader = create_data_loader(train_dataset, args.batch_size)
    
    if args.val_file:
        val_dataset = DrugReviewDataset(args.val_file, max_length=args.max_length)
        val_loader = create_data_loader(val_dataset, args.batch_size, shuffle=False)

    # 创建配置
    config = ModelConfig()
    config.vocab_size = len(train_dataset.tokenizer)
    config.embedding_dim = args.embedding_dim
    config.lstm_units = args.lstm_units
    config.dropout_rate = args.dropout_rate
    config.learning_rate = args.learning_rate

    # 创建模型
    model = SentimentModel(config).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 创建模型保存目录
    os.makedirs(args.model_dir, exist_ok=True)

    # 创建 TensorBoard writer
    writer = SummaryWriter(os.path.join(args.model_dir, 'tensorboard_logs'))

    # 训练循环
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # 训练阶段
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # 记录每个 batch 的训练数据
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Training/BatchLoss', loss.item(), global_step)
            writer.add_scalar('Training/BatchAccuracy', 100.*correct/total, global_step)
            progress_bar.set_postfix({'loss': total_loss/len(train_loader),'acc': 100.*correct/total})

        # 记录每个 epoch 的训练数据
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        writer.add_scalar('Training/EpochLoss', epoch_loss, epoch)
        writer.add_scalar('Training/EpochAccuracy', epoch_acc, epoch)

        # 验证阶段
        if args.val_file:
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    # 计算验证损失
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

            val_acc = 100. * val_correct / val_total
            val_loss = val_loss / len(val_loader)
            
            # 记录验证数据
            writer.add_scalar('Validation/Loss', val_loss, epoch)
            writer.add_scalar('Validation/Accuracy', val_acc, epoch)
            
            logging.info(f'Validation Accuracy: {val_acc:.2f}%')

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join(args.model_dir, 'best_model.pth'))

    # 关闭 TensorBoard writer
    writer.close()

    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(args.model_dir, 'final_model.pth'))

if __name__ == '__main__':
    args = parse_args()
    setup_logging()
    train(args)