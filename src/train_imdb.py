import torch
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    Trainer, TrainingArguments, DebertaV2Tokenizer
)
from swanlab.integration.transformers import SwanLabCallback
import swanlab
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# ========== 配置项 ==========
CONFIG = {
    "models": {  # 待对比模型列表
        "bert-base-uncased": "BERT",
        "roberta-base": "RoBERTa",
        "distilbert-base-uncased": "DistilBERT",
        "albert-base-v2": "ALBERT",
        "deberta-v3-base": "DeBERTa",
    },
    "dataset": "imdb",
    "max_length": 512,
    "learning_rate": 2e-5,
    "batch_size": 8,
    "epochs": 3,
    "weight_decay": 0.01,
    "logging_steps": 100,
    "output_dir": "./results",
    "swanlab_project": "PLM-IMDB-Comparison"
}

# 创建结果保存目录
os.makedirs(CONFIG["output_dir"], exist_ok=True)

def load_and_split_dataset() -> tuple:
    """加载IMDB数据集并分割为训练集/验证集/测试集"""
    try:
        dataset = load_dataset(CONFIG["dataset"])
        print(f"✅ 成功加载{CONFIG['dataset']}数据集，共{len(dataset['train'])+len(dataset['test'])}条样本")
    except Exception as e:
        raise RuntimeError(f"❌ 数据集加载失败：{e}") from e
    
    # 分割测试集为验证集和测试集
    test_dataset = dataset["test"]
    val_size = len(test_dataset) // 2
    val_dataset = test_dataset.select(range(val_size))
    test_dataset = test_dataset.select(range(val_size, len(test_dataset)))
    
    # 整理数据集
    dataset_split = DatasetDict({
        "train": dataset["train"],
        "val": val_dataset,
        "test": test_dataset
    })
    print(f"📊 数据集分割完成：训练集{len(dataset_split['train'])}条 | 验证集{len(dataset_split['val'])}条 | 测试集{len(dataset_split['test'])}条")
    return dataset_split["train"], dataset_split["val"], dataset_split["test"]

def get_tokenizer(model_name: str):
    """根据模型名称加载对应的Tokenizer，处理特殊情况（如Pad Token缺失）"""
    try:
        if "deberta" in model_name.lower():
            tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 处理无Pad Token的情况
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"⚠️  模型{model_name}无Pad Token，已使用EOS Token替代")
        return tokenizer
    except Exception as e:
        raise RuntimeError(f"❌ Tokenizer加载失败（模型：{model_name}）：{e}") from e

def tokenize_function(batch, tokenizer):
    """批量分词函数"""
    return tokenizer(
        batch["text"],
        padding="max_length",  # 固定长度填充
        truncation=True,
        max_length=CONFIG["max_length"]
    )

def compute_metrics(eval_pred):
    """计算评估指标：准确率、精确率、召回率、F1分数"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        "accuracy": round(accuracy_score(labels, predictions), 4),
        "precision": round(precision_score(labels, predictions, average="binary"), 4),
        "recall": round(recall_score(labels, predictions, average="binary"), 4),
        "f1": round(f1_score(labels, predictions, average="binary"), 4)
    }

def train_and_evaluate_model(
    model_name: str,
    model_display_name: str,
    train_dataset,
    val_dataset,
    test_dataset
) -> dict:
    """训练单个模型并返回测试集结果"""
    print(f"\n{'='*60}")
    print(f"🚀 开始训练：{model_display_name}（模型：{model_name}）")
    print(f"{'='*60}")
    
    # 计时开始
    start_time = time.time()
    
    # 1. 加载Tokenizer并分词
    tokenizer = get_tokenizer(model_name)
    tokenized_train = train_dataset.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        desc=f"分词：{model_display_name}训练集"
    )
    tokenized_val = val_dataset.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        desc=f"分词：{model_display_name}验证集"
    )
    tokenized_test = test_dataset.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        desc=f"分词：{model_display_name}测试集"
    )
    
    # 2. 格式化数据集
    for ds in [tokenized_train, tokenized_val, tokenized_test]:
        ds = ds.rename_column("label", "labels")
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    # 3. 加载模型
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2  # 二分类任务
        )
    except Exception as e:
        raise RuntimeError(f"❌ 模型加载失败（模型：{model_name}）：{e}") from e
    
    # 4. 配置训练参数
    training_args = TrainingArguments(
        output_dir=os.path.join(CONFIG["output_dir"], model_name.replace("/", "-")),
        eval_strategy="epoch",  # 每个epoch评估一次
        save_strategy="epoch",  # 每个epoch保存一次模型
        learning_rate=CONFIG["learning_rate"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        num_train_epochs=CONFIG["epochs"],
        weight_decay=CONFIG["weight_decay"],
        logging_steps=CONFIG["logging_steps"],
        report_to="none",  # 禁用默认日志工具，使用SwanLab
        load_best_model_at_end=True,  # 训练结束加载最优模型
        metric_for_best_model="accuracy",  # 以准确率为最优模型判定标准
        save_total_limit=1  # 只保存最优模型
    )
    
    # 5. 配置SwanLab回调
    swanlab_callback = SwanLabCallback(
        project=CONFIG["swanlab_project"],
        experiment_name=f"{model_display_name}-IMDB",
        config={k: v for k, v in CONFIG.items() if k != "models"},  # 记录实验配置
        tags=["sentiment-analysis", "pre-trained-model", model_display_name]
    )
    
    # 6. 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        callbacks=[swanlab_callback],
        compute_metrics=compute_metrics
    )
    
    # 7. 开始训练
    trainer.train()
    
    # 8. 测试集评估
    test_results = trainer.evaluate(tokenized_test)
    training_time = round(time.time() - start_time, 2)
    test_results["training_time"] = training_time
    
    # 9. 打印结果
    print(f"\n✅ {model_display_name}训练完成！")
    print(f"📈 测试集结果：准确率{test_results['eval_accuracy']} | F1{test_results['eval_f1']} | 训练时间{training_time}s")
    return {
        "model_name": model_display_name,
        "accuracy": test_results["eval_accuracy"],
        "precision": test_results["eval_precision"],
        "recall": test_results["eval_recall"],
        "f1": test_results["eval_f1"],
        "training_time": training_time
    }

def plot_results(results: list):
    """绘制模型性能对比图（保存到results目录）"""
    models = [r["model_name"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    f1_scores = [r["f1"] for r in results]
    training_times = [r["training_time"] for r in results]
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1：准确率与F1分数
    x = np.arange(len(models))
    width = 0.35
    ax1.bar(x - width/2, accuracies, width, label="准确率", color="#2E86AB")
    ax1.bar(x + width/2, f1_scores, width, label="F1分数", color="#A23B72")
    ax1.set_xlabel("模型")
    ax1.set_ylabel("分数（越高越好）")
    ax1.set_title("各模型分类性能对比")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    
    # 子图2：训练时间
    ax2.bar(models, training_times, color="#F18F01", alpha=0.7)
    ax2.set_xlabel("模型")
    ax2.set_ylabel("训练时间（s，越低越好）")
    ax2.set_title("各模型训练效率对比")
    ax2.grid(axis="y", alpha=0.3)
    
    # 调整布局并保存
    plt.tight_layout()
    save_path = os.path.join(CONFIG["output_dir"], "model_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 性能对比图已保存至：{save_path}")

def main():
    """主函数：加载数据→训练所有模型→评估→可视化结果"""
    # 1. 加载并分割数据集
    train_dataset, val_dataset, test_dataset = load_and_split_dataset()
    
    # 2. 训练所有模型
    all_results = []
    for model_name, display_name in CONFIG["models"].items():
        try:
            result = train_and_evaluate_model(
                model_name, display_name, train_dataset, val_dataset, test_dataset
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ {display_name}训练失败：{e}")
            continue
    
    # 3. 输出汇总结果
    print(f"\n{'='*80}")
    print("📋 所有模型性能汇总")
    print(f"{'='*80}")
    print(f"{'模型':<10} {'准确率':<10} {'F1分数':<10} {'训练时间(s)':<15}")
    print("-"*80)
    for res in all_results:
        print(f"{res['model_name']:<10} {res['accuracy']:<10.4f} {res['f1']:<10.4f} {res['training_time']:<15.2f}")
    
    # 4. 绘制可视化图表
    if all_results:
        plot_results(all_results)
    
    # 5. 保存结果到CSV
    import pandas as pd
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(CONFIG["output_dir"], "model_results.csv"), index=False)
    print(f"\n📄 结果数据已保存至：{os.path.join(CONFIG['output_dir'], 'model_results.csv')}")

if __name__ == "__main__":
    main()