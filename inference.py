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
        if not text.strip():
            return "请输入评论内容后再进行分析。"
            
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
        sentiment_map = {0: '消极', 1: '中性', 2: '积极'}
        prediction = sentiment_map[probs.argmax()]
        
        # 构建结果字符串
        result = f"✨ 情感分析结果：{prediction}\n\n"
        result += "📊 概率分布：\n"
        result += f"消极: {probs[0]:.2%}\n"
        result += f"中性: {probs[1]:.2%}\n"
        result += f"积极: {probs[2]:.2%}"# 添加情感标签的颜色提示
        color_map = {
            '消极': '🔴',
            '中性': '⚪',
            '积极': '🟢'
        }
        result = f"{color_map[prediction]} {result}"
        
        return result

    def predict_file(self, file):
        """预测CSV文件中的评论"""
        if file is None:
            return "请先上传CSV文件。"
            
        try:
            df = pd.read_csv(file.name)
            if 'review' not in df.columns:
                return "错误：CSV文件必须包含'review'列"
            
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
            sentiment_map = {0: '消极', 1: '中性', 2: '积极'}
            df['预测情感'] = [sentiment_map[pred] for pred in results]
            
            # 保存结果
            output_path = "预测结果.csv"
            df.to_csv(output_path, index=False)
            return f"✅ 预测完成！结果已保存至 {output_path}\n\n📊 情感分布统计：\n" + \
                   f"积极评价：{(df['预测情感']=='积极').sum()} 条\n" + \
                   f"中性评价：{(df['预测情感']=='中性').sum()} 条\n" + \
                   f"消极评价：{(df['预测情感']=='消极').sum()} 条"
        except Exception as e:
            return f"❌ 处理文件时出错：{str(e)}"

def create_interface():
    analyzer = SentimentAnalyzer("checkpoints/best_model.pth")  # 修改为你的模型路径
    
    # 自定义CSS样式
    custom_css = """
    .gradio-container {
        font-family: "Microsoft YaHei", "微软雅黑", sans-serif !important;
    }
    .markdown-text {
        font-size: 16px !important;
        line-height: 1.5 !important;
    }
    .tab-selected {
        background-color: #2196F3 !important;
        color: white !important;
    }
    """
    
    with gr.Blocks(title="药品评论情感分析", css=custom_css) as interface:
        gr.Markdown("""
        # 🏥 药品评论情感分析系统
        
        本系统使用深度学习技术，帮助您分析药品评论的情感倾向。
        """)
        
        with gr.Tabs():
            # 单文本分析标签页
            with gr.TabItem("✍️ 单条评论分析"):
                with gr.Column():
                    text_input = gr.Textbox(
                        label="请输入药品评论",
                        placeholder="在此输入您要分析的评论内容...",
                        lines=4
                    )
                    text_button = gr.Button("开始分析", variant="primary")
                    text_output = gr.Textbox(
                        label="分析结果",
                        lines=6
                    )
                
                gr.Markdown("""
                ### 💡 使用说明
                1. 在文本框中输入您想要分析的药品评论
                2. 点击"开始分析"按钮
                3. 系统将显示情感分析结果和概率分布
                """)
            
            # 文件分析标签页
            with gr.TabItem("📁 批量评论分析"):
                with gr.Column():
                    file_input = gr.File(
                        label="上传CSV文件",
                        file_types=[".csv"]
                    )
                    file_button = gr.Button("开始分析", variant="primary")
                    file_output = gr.Textbox(
                        label="处理结果",
                        lines=6
                    )
                
                gr.Markdown("""
                ### 💡 使用说明
                1. 准备包含'review'列的CSV文件
                2. 上传文件后点击"开始分析"
                3. 系统将生成分析报告并保存结果
                
                ### 📋 文件格式要求
                - 文件格式：CSV
                - 必需列：review（评论内容）
                - 编码：UTF-8
                """)
        
        gr.Markdown("""
        ### 🔍 关于情感分类
        - 🟢 积极：表示用户对药品持正面评价
        - ⚪ 中性：表示用户评价较为中立
        - 🔴 消极：表示用户对药品持负面评价
        
        ### 👨‍💻 技术支持
        如有问题请联系技术支持团队
        """)
        
        # 绑定事件
        text_button.click(
            fn=analyzer.predict_text,
            inputs=text_input,
            outputs=text_output
        )
        
        file_button.click(
            fn=analyzer.predict_file,
            inputs=file_input,
            outputs=file_output
        )
    
    return interface

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建并启动界面
    interface = create_interface()
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        favicon_path="favicon.ico"  # 如果有图标的话
    )