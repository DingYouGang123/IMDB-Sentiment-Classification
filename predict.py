import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DebertaV2Tokenizer
import argparse

def load_model_and_tokenizer(model_path: str):
    """加载训练好的模型和Tokenizer"""
    # 判断模型类型，加载对应Tokenizer
    if "deberta" in model_path.lower():
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 处理Pad Token缺失问题
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()  # 切换到评估模式
    return model, tokenizer

def predict_sentiment(comment: str, model, tokenizer, max_length=512):
    """预测单条评论的情感倾向"""
    # 分词
    inputs = tokenizer(
        comment,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # 预测
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
    
    # 映射标签
    sentiment = "正面" if predicted_class_id == 1 else "负面"
    return sentiment

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="IMDB电影评论情感预测工具")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="训练好的模型路径（如：./results/deberta-v3-base）"
    )
    parser.add_argument(
        "--comment",
        type=str,
        required=True,
        help="待预测的电影评论（英文）"
    )
    
    args = parser.parse_args()
    
    # 加载模型和Tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(args.model_path)
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return
    
    # 预测并输出结果
    sentiment = predict_sentiment(args.comment, model, tokenizer)
    print(f"\n📝 输入评论：{args.comment}")
    print(f"❤️  预测情感：{sentiment}")

if __name__ == "__main__":
    main()