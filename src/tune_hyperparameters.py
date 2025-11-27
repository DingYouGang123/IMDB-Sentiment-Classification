import torch
import numpy as np
import optuna
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import swanlab
from swanlab.integration.transformers import SwanLabCallback

# ========== 调优配置 ==========
TUNE_CONFIG = {
    "model_name": "roberta-base",  # 选定调优模型
    "dataset": "imdb",
    "max_length": 512,
    "num_trials": 10,  # 搜索轮数
    "num_epochs": 3,  # 每轮调优训练轮数
    "output_dir": "./results/roberta-tuning",
    "swanlab_project": "PLM-IMDB-Tuning",
    "metric_for_best_model": "accuracy"  # 调优目标指标
}

# 创建调优结果目录
os.makedirs(TUNE_CONFIG["output_dir"], exist_ok=True)

# ========== 复用核心函数 ==========
def load_and_split_dataset() -> tuple:
    """加载并分割数据集（与train_imdb.py完全一致）"""
    dataset = load_dataset(TUNE_CONFIG["dataset"])
    test_dataset = dataset["test"]
    val_size = len(test_dataset) // 2
    val_dataset = test_dataset.select(range(val_size))
    test_dataset = test_dataset.select(range(val_size, len(test_dataset)))
    return dataset["train"], val_dataset, test_dataset

def compute_metrics(eval_pred):
    """评估指标计算（与train_imdb.py完全一致）"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": round(accuracy_score(labels, predictions), 4),
        "precision": round(precision_score(labels, predictions, average="binary"), 4),
        "recall": round(recall_score(labels, predictions, average="binary"), 4),
        "f1": round(f1_score(labels, predictions, average="binary"), 4)
    }

# ========== 调优目标函数 ==========
def objective(trial: optuna.Trial) -> float:
    """Optuna目标函数：搜索最优超参数并返回验证集准确率"""
    # 1. 定义超参数搜索空间
    hyperparameters = {
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [1e-5, 2e-5, 5e-5, 1e-4]  # 学习率范围
        ),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [4, 8, 16]  # 适配GPU显存
        ),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [0.01, 0.05, 0.1, 0.2]  # 正则化强度，防止过拟合
        ),
        "warmup_ratio": trial.suggest_float(
            "warmup_ratio", 0.05, 0.2, step=0.05  # 学习率预热比例，稳定训练初期
        )
    }

    # 2. 加载Tokenizer和数据集
    tokenizer = AutoTokenizer.from_pretrained(TUNE_CONFIG["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset, val_dataset, _ = load_and_split_dataset()
    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=TUNE_CONFIG["max_length"])
    
    tokenized_train = train_dataset.map(tokenize_fn, batched=True)
    tokenized_val = val_dataset.map(tokenize_fn, batched=True)
    
    # 格式化数据集
    for ds in [tokenized_train, tokenized_val]:
        ds = ds.rename_column("label", "labels")
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # 3. 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(
        TUNE_CONFIG["model_name"], num_labels=2
    )

    # 4. 配置训练参数
    training_args = TrainingArguments(
        output_dir=os.path.join(TUNE_CONFIG["output_dir"], f"trial-{trial.number}"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=hyperparameters["learning_rate"],
        per_device_train_batch_size=hyperparameters["per_device_train_batch_size"],
        per_device_eval_batch_size=16,
        num_train_epochs=TUNE_CONFIG["num_epochs"],
        weight_decay=hyperparameters["weight_decay"],
        warmup_ratio=hyperparameters["warmup_ratio"],
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model=TUNE_CONFIG["metric_for_best_model"],
        save_total_limit=1,
        disable_tqdm=True  # 禁用进度条，减少日志冗余
    )

    # 5. 回调函数（早停+实验跟踪+剪枝）
    callbacks = [
        # 早停：验证集性能3轮无提升则停止训练
        EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001),
        # SwanLab：记录每轮调优的超参数和性能
        SwanLabCallback(
            project=TUNE_CONFIG["swanlab_project"],
            experiment_name=f"RoBERTa-Trial-{trial.number}",
            config={**TUNE_CONFIG, **hyperparameters},
            tags=["hyperparameter-tuning", "RoBERTa", "sentiment-analysis"]
        ),
        # Optuna剪枝：性能不佳的实验提前终止，节省资源
        PyTorchLightningPruningCallback(trial, "eval_accuracy")
    ]

    # 6. 训练与评估
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    trainer.train()
    val_results = trainer.evaluate()
    val_accuracy = val_results["eval_accuracy"]

    # 记录当前实验的所有指标
    swanlab.log({
        **hyperparameters,
        "val_accuracy": val_accuracy,
        "val_f1": val_results["eval_f1"],
        "val_precision": val_results["eval_precision"],
        "val_recall": val_results["eval_recall"]
    })

    return val_accuracy

# ========== 调优主函数 ==========
def main():
    print(f"🚀 开始RoBERTa模型超参数调优（共{len(TUNE_CONFIG['num_trials'])}轮）")
    print(f"📌 调优目标：最大化{config['metric_for_best_model']}")
    print(f"🔍 搜索超参数：学习率、批量大小、权重衰减、预热比例")

    # 1. 初始化Optuna研究
    study = optuna.create_study(
        direction="maximize",  # 最大化准确率
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=50),  # 剪枝策略
        study_name="RoBERTa-IMDB-Tuning"
    )

    # 2. 启动超参数搜索
    study.optimize(
        objective,
        n_trials=TUNE_CONFIG["num_trials"],
        show_progress_bar=True,
        catch=(Exception,)  # 捕获异常，避免单轮失败终止整个调优
    )

    # 3. 输出调优结果
    best_trial = study.best_trial
    print(f"\n{'='*80}")
    print(f"🎉 调优完成！最优结果如下：")
    print(f"{'='*80}")
    print(f"最优验证集准确率：{best_trial.value:.4f}")
    print(f"最优超参数组合：")
    for key, value in best_trial.params.items():
        print(f"  - {key}: {value}")

    # 4. 保存最优超参数到文件
    best_hparams_path = os.path.join(TUNE_CONFIG["output_dir"], "best_hyperparameters.json")
    import json
    with open(best_hparams_path, "w") as f:
        json.dump(best_trial.params, f, indent=4)
    print(f"\n📄 最优超参数已保存至：{best_hparams_path}")

    # 5. 用最优超参数在测试集上验证最终性能
    print(f"\n🔧 用最优超参数验证测试集性能...")
    train_best_model(best_trial.params)

def train_best_model(best_hparams: dict):
    """使用最优超参数训练最终模型，并在测试集评估"""
    # 加载数据和Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TUNE_CONFIG["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset, val_dataset, test_dataset = load_and_split_dataset()
    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=TUNE_CONFIG["max_length"])
    
    tokenized_train = train_dataset.map(tokenize_fn, batched=True)
    tokenized_val = val_dataset.map(tokenize_fn, batched=True)
    tokenized_test = test_dataset.map(tokenize_fn, batched=True)
    
    for ds in [tokenized_train, tokenized_val, tokenized_test]:
        ds = ds.rename_column("label", "labels")
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(
        TUNE_CONFIG["model_name"], num_labels=2
    )

    # 最优参数训练配置
    training_args = TrainingArguments(
        output_dir=os.path.join(TUNE_CONFIG["output_dir"], "best-model"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=best_hparams["learning_rate"],
        per_device_train_batch_size=best_hparams["per_device_train_batch_size"],
        per_device_eval_batch_size=16,
        num_train_epochs=TUNE_CONFIG["num_epochs"],
        weight_decay=best_hparams["weight_decay"],
        warmup_ratio=best_hparams["warmup_ratio"],
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model=TUNE_CONFIG["metric_for_best_model"],
        save_total_limit=1
    )

    # 训练最终模型
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    test_results = trainer.evaluate(tokenized_test)

    # 输出测试集最终性能
    print(f"\n📊 最优模型测试集性能：")
    print(f"  - 准确率：{test_results['eval_accuracy']:.4f}")
    print(f"  - F1分数：{test_results['eval_f1']:.4f}")
    print(f"  - 精确率：{test_results['eval_precision']:.4f}")
    print(f"  - 召回率：{test_results['eval_recall']:.4f}")

    # 保存测试集结果
    test_results_path = os.path.join(TUNE_CONFIG["output_dir"], "test_results.json")
    import json
    with open(test_results_path, "w") as f:
        json.dump(test_results, f, indent=4)
    print(f"\n📄 测试集结果已保存至：{test_results_path}")

if __name__ == "__main__":
    main()