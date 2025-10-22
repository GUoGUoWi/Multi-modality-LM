import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import Config
from tqdm import tqdm


class Evaluator:
    """评估器类，用于评估模型性能"""

    def __init__(self, logger=None):
        """
        初始化评估器

        Args:
            logger: 日志记录器
        """
        self.logger = logger
        self.results = {}

    def evaluate_client(self, client, test_loader=None):
        """
        评估客户端模型

        Args:
            client: 客户端实例
            test_loader: 测试数据加载器，如果为None则使用客户端的验证加载器

        Returns:
            评估指标字典
        """
        if test_loader is None:
            test_loader = client.val_loader

        client.model.eval()

        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating Client {client.client_id}"):
                # 预处理输入
                inputs = client.model.preprocess(batch)
                labels = batch["label"].to(Config.DEVICE)

                # 前向传播
                outputs = client.model(inputs)
                probs = torch.softmax(outputs, dim=1)

                # 获取预测
                _, preds = torch.max(outputs, 1)

                # 记录标签和预测
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 计算评估指标
        metrics = self._compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))

        # 记录结果
        self.results[f"client_{client.client_id}"] = metrics

        # 记录日志
        if self.logger:
            self.logger.info(f"Client {client.client_id} Evaluation Results:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def _compute_metrics(self, labels, predictions, probabilities):
        """
        计算评估指标

        Args:
            labels: 真实标签
            predictions: 预测标签
            probabilities: 预测概率

        Returns:
            指标字典
        """
        # 多分类评估指标
        try:
            accuracy = accuracy_score(labels, predictions)
            precision_micro = precision_score(labels, predictions, average='micro')
            precision_macro = precision_score(labels, predictions, average='macro')
            recall_micro = recall_score(labels, predictions, average='micro')
            recall_macro = recall_score(labels, predictions, average='macro')
            f1_micro = f1_score(labels, predictions, average='micro')
            f1_macro = f1_score(labels, predictions, average='macro')

            # 计算混淆矩阵
            cm = confusion_matrix(labels, predictions)

            metrics = {
                "accuracy": accuracy,
                "precision_micro": precision_micro,
                "precision_macro": precision_macro,
                "recall_micro": recall_micro,
                "recall_macro": recall_macro,
                "f1_micro": f1_micro,
                "f1_macro": f1_macro,
                "confusion_matrix": cm
            }
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error computing metrics: {str(e)}")
            metrics = {
                "accuracy": 0.0,
                "error": str(e)
            }

        return metrics

    def plot_confusion_matrix(self, client_id, save_dir=None):
        """
        绘制混淆矩阵

        Args:
            client_id: 客户端ID
            save_dir: 保存目录
        """
        key = f"client_{client_id}"
        if key not in self.results or "confusion_matrix" not in self.results[key]:
            if self.logger:
                self.logger.warning(f"Confusion matrix not available for client {client_id}")
            return

        cm = self.results[key]["confusion_matrix"]

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=False, yticklabels=False)
        plt.title(f'Confusion Matrix - Client {client_id}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_dir is None:
            save_dir = Config.OUTPUT_DIR

        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"client_{client_id}_confusion_matrix.png"))
        plt.close()

    def save_results(self, filepath):
        """
        保存评估结果

        Args:
            filepath: 保存路径
        """
        # 创建可序列化的结果副本
        serializable_results = {}
        for client_id, metrics in self.results.items():
            serializable_results[client_id] = {}
            for metric, value in metrics.items():
                if isinstance(value, np.ndarray):
                    # 跳过大型数组，如混淆矩阵
                    continue
                serializable_results[client_id][metric] = value

        # 保存结果
        import json
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        if self.logger:
            self.logger.info(f"Evaluation results saved to {filepath}")