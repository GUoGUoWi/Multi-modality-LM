import os
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from config import Config, ModelType
from data_processor_scenario2 import DataProcessor
from client import Client
from logger import Logger
import platform
##任务：两个参与方 都缺描述

def set_random_seed(seed):
    """
    设置随机种子以确保可重复性

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def main():
    # 设置随机种子
    set_random_seed(Config.SEED)

    # 创建日志记录器
    logger = Logger(Config.LOG_DIR)

    # 加载和预处理数据
    processor = DataProcessor(logger)
    data = processor.load_and_preprocess_data()

    # 为客户端划分数据
    logger.log_timing_start("data_splitting")
    client_data = processor.split_data_for_clients(data)
    logger.log_timing_end("data_splitting")

    # 初始化客户端
    logger.info("初始化客户端...")
    clients = []
    for client_id in range(Config.NUM_CLIENTS):
        model_type = Config.get_client_model(client_id)
        logger.info(f"Client {client_id} using model: {model_type.name.lower()}")

        # 获取数据加载器
        train_loader, val_loader = processor.get_client_dataloaders(client_id)

        # 创建客户端
        client = Client(
            client_id=client_id,
            model_type=model_type,
            num_classes=len(processor.category_map),
            train_loader=train_loader,
            val_loader=val_loader,
            logger=logger
        )

        # 记录客户端初始化信息
        logger.info(f"Client {client_id} initialized - Model: {model_type.name.lower()}")

        clients.append(client)

    # 训练客户端模型
    logger.info("开始训练...")

    # 串行训练客户端
    for i, client in enumerate(clients):
        logger.info(f"训练客户端 {i} ({client.model_type.name.lower()})...")
        client.train(epochs=Config.NUM_EPOCHS)
        client.evaluate()

    # 记录实验完成
    logger.info("实验完成!")

if __name__ == "__main__":
    # 在Windows上正确设置多进程启动方法
    if platform.system() == "Windows":
        mp.set_start_method('spawn', force=True)

    main()