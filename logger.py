import os
import json
import logging
import time
from datetime import datetime
import torch


class Logger:
    """日志记录器，支持记录实验信息、训练过程和性能指标"""

    def __init__(self, log_dir):
        """
        初始化日志记录器

        Args:
            log_dir: 日志保存目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 创建时间戳，用于区分不同的实验
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(log_dir, self.timestamp)
        os.makedirs(self.experiment_dir, exist_ok=True)

        # 设置日志格式和文件
        self.log_file = os.path.join(self.experiment_dir, "experiment.log")

        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger()

        # 记录实验开始
        self.info(f"Starting experiment: {log_dir}")

        # 记录配置信息
        self.log_config()

        # 用于计时的字典
        self.timing_start = {}
        self.timings = {}

        # 用于记录各个客户端训练历史的字典
        self.client_history = {}

    def _is_json_serializable(self, obj):
        """检查对象是否可以被JSON序列化"""
        try:
            json.dumps(obj)
            return True
        except (TypeError, OverflowError):
            return False

    def _make_json_serializable(self, obj):
        """将对象转换为JSON可序列化的形式"""
        if isinstance(obj, torch.device):
            return str(obj)
        elif hasattr(obj, "__name__"):
            return obj.__name__
        elif hasattr(obj, "__str__"):
            return str(obj)
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        return obj

    def log_config(self):
        """记录实验配置"""
        from config import Config

        # 将配置以JSON格式保存
        config_file = os.path.join(self.experiment_dir, "config.json")

        # 创建配置字典并确保所有值都是JSON可序列化的
        config_dict = {}
        for key in dir(Config):
            if not key.startswith("__") and key.isupper():
                value = getattr(Config, key)
                # 处理非JSON可序列化的值
                if not self._is_json_serializable(value):
                    value = self._make_json_serializable(value)
                config_dict[key] = value

        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

        # 在日志中记录配置
        self.info(f"Configuration: {json.dumps(config_dict, indent=2)}")

    def info(self, message):
        """记录信息日志"""
        self.logger.info(message)

    def warning(self, message):
        """记录警告日志"""
        self.logger.warning(message)

    def error(self, message):
        """记录错误日志"""
        self.logger.error(message)

    def log_timing_start(self, name):
        """开始计时"""
        self.timing_start[name] = time.time()

    def log_timing_end(self, name):
        """结束计时并记录"""
        if name in self.timing_start:
            elapsed = time.time() - self.timing_start[name]
            self.timings[name] = elapsed
            self.info(f"Timing - {name}: {elapsed:.4f} seconds")
            return elapsed
        return None

    def log_timing(self, name, elapsed):
        """记录已计算的时间"""
        self.timings[name] = elapsed
        self.info(f"Timing - {name}: {elapsed:.4f} seconds")

    def log_epoch(self, client_id, epoch, train_loss, train_acc, val_loss=None, val_acc=None):
        """
        记录训练轮次信息

        Args:
            client_id: 客户端ID
            epoch: 轮次
            train_loss: 训练损失
            train_acc: 训练准确率（百分比形式）
            val_loss: 验证损失，可选
            val_acc: 验证准确率（百分比形式），可选
        """
        # 初始化客户端历史记录（如果还不存在）
        if client_id not in self.client_history:
            self.client_history[client_id] = {
                "train_loss": [],
                "train_acc": [],
                "val_loss": [],
                "val_acc": []
            }

        # 记录训练指标
        self.client_history[client_id]["train_loss"].append(train_loss)
        self.client_history[client_id]["train_acc"].append(train_acc)

        # 日志信息，使用百分比格式显示准确率
        log_msg = f"Client {client_id} - Epoch {epoch} - "
        log_msg += f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}%"

        # 如果有验证指标，也记录
        if val_loss is not None and val_acc is not None:
            self.client_history[client_id]["val_loss"].append(val_loss)
            self.client_history[client_id]["val_acc"].append(val_acc)
            log_msg += f", Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}%"

        self.info(log_msg)

        # 保存历史记录
        self._save_history()

    def _save_history(self):
        """保存训练历史记录到文件"""
        history_file = os.path.join(self.experiment_dir, "history.json")
        with open(history_file, 'w') as f:
            json.dump(self.client_history, f, indent=2)

    def finalize(self):
        """完成日志记录，保存最终结果"""
        # 计算并保存总体执行时间
        total_time = sum(self.timings.values())
        self.info(f"Total execution time: {total_time:.4f} seconds")

        # 保存所有计时信息
        timing_file = os.path.join(self.experiment_dir, "timings.json")
        with open(timing_file, 'w') as f:
            json.dump(self.timings, f, indent=2)

        # 保存最终历史记录
        self._save_history()

        self.info("Experiment completed")