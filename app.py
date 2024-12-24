# app.py
import gradio as gr
import torch
import pandas as pd
from model import SentimentModel, ModelConfig
from transformers import BertTokenizer
import logging
import os

class SentimentAnalyzer:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # 初始化模型
        config = ModelConfig()
        config.vocab_size = self.tokenizer.vocab_size
        self.model = SentimentModel(config).to(self.device)
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict_text(self, text, max_length=128):
        """预测单个文本的情感"""
        # 预处理文本
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        # 获取预测结果和概率
        probs = probabilities[0].cpu().numpy()
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        prediction = sentiment_map[probs.argmax()]
        
        # 构建结果字符串
        result = f"Prediction: {prediction}\n\n"
        result += "Probabilities:\n"
        result += f"Negative: {probs[0]:.2%}\n"
        result += f"Neutral: {probs[1]:.2%}\n"
        result += f"Positive: {probs[2]:.2%}"
        
        return result

    def predict_file(self, file):
        """预测CSV文件中的评论"""
        try:
            df = pd.read_csv(file.name)
            if 'review' not in df.columns:
                return "Error: CSV file must contain a 'review' column"
            
            results = []
            for text in df['review']:
                encoding = self.tokenizer(
                    str(text),
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(input_ids, attention_mask)
                    _, predicted = torch.max(outputs, 1)
                    results.append(predicted.item())
            
            # 添加预测结果到DataFrame
            sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            df['predicted_sentiment'] = [sentiment_map[pred] for pred in results]
            
            # 保存结果
            output_path = "predictions.csv"
            df.to_csv(output_path, index=False)
            return f"Predictions saved to {output_path}"
        except Exception as e:
            return f"Error processing file: {str(e)}"

def create_interface():
    analyzer = SentimentAnalyzer("checkpoints/best_model.pth")  # 修改为你的模型路径
    
    with gr.Blocks(title="Drug Review Sentiment Analysis") as interface:
        gr.Markdown("""
        # Drug Review Sentiment Analysis
        Analyze the sentiment of drug reviews using deep learning.
        """)
        with gr.Tabs():
            # 单文本分析标签页
            with gr.TabItem("Single Text Analysis"):
                text_input = gr.Textbox(
                    label="Enter your drug review",
                    placeholder="Type your review here...",
                    lines=4
                )
                text_button = gr.Button("Analyze")
                text_output = gr.Textbox(label="Analysis Result", lines=6)
                text_button.click(
                    fn=analyzer.predict_text,
                    inputs=text_input,
                    outputs=text_output
                )
            
            #文件分析标签页
            with gr.TabItem("Batch File Analysis"):
                file_input = gr.File(
                    label="Upload CSV file",
                    file_types=[".csv"]
                )
                file_button = gr.Button("Analyze File")
                file_output = gr.Textbox(label="Processing Result")
                file_button.click(
                    fn=analyzer.predict_file,
                    inputs=file_input,
                    outputs=file_output
                )
        
        gr.Markdown("""
        ### Instructions:
        - For single text analysis: Simply enter your review text and click 'Analyze'
        - For batch analysis: Upload a CSV file with a 'review' column and click 'Analyze File'
        """)
    
    return interface

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建并启动界面
    interface = create_interface()
    interface.launch(
        share=True,  # 设置为False可以禁用公共URL
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860  # 指定端口
    )