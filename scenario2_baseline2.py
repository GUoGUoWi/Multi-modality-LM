import os
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from config import Config, ModelType
from data_processor_scenario2 import DataProcessor
from client import EnhancedClient
from logger import Logger
import platform
import argparse
from generate_caption import generate_missing_captions

def set_random_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Baseline 2: 使用图生文补全训练")
    parser.add_argument("--seed", type=int, default=Config.SEED, help="随机种子")
    args = parser.parse_args()

    # 设置随机种子
    set_random_seed(args.seed)

    # 创建日志记录器
    logger = Logger(Config.LOG_DIR)
    logger.info("Baseline 2: 使用BLIP生成缺失描述的模型训练")

    # 检查生成描述目录是否存在
    generated_caption_dir = os.path.join(Config.OUTPUT_DIR, "generated_captions")
    if not os.path.exists(generated_caption_dir):
        os.makedirs(generated_caption_dir, exist_ok=True)
        logger.warning(f"生成描述目录 {generated_caption_dir} 不存在，已创建")

    # 加载和预处理数据
    logger.info("加载和预处理数据...")
    processor = DataProcessor(logger)
    data = processor.load_and_preprocess_data()

    # 为客户端划分数据
    logger.log_timing_start("data_splitting")
    client_data = processor.split_data_for_clients(data,num_clients=Config.NUM_CLIENTS)
    logger.log_timing_end("data_splitting")

    # 为客户端生成缺失描述
    aggregation = False      # baseline2 使用各自本地的大模型
    generate_missing_captions(client_data, aggregation=aggregation)

    # 统计生成描述数量
    generated_captions_dir = os.path.join(Config.OUTPUT_DIR, "generated_captions")
    gen_captions_count = 0
    for root, dirs, files in os.walk(generated_captions_dir):
        gen_captions_count += sum(1 for f in files if f.lower().endswith(".txt"))
    logger.info(f"找到 {gen_captions_count} 条生成的描述")

    # 客户端都缺描述

    # 初始化客户端
    logger.info("初始化客户端...")
    clients = []
    for client_id in range(Config.NUM_CLIENTS):
        model_type = Config.get_client_model(client_id)
        logger.info(f"Client {client_id}: {model_type.name} - 使用生成描述补全")

        # 获取数据加载器，使用生成图像替代缺失图像
        train_loader, val_loader = processor.get_client_dataloaders(
                client_id=client_id,
                use_generated_captions=True,  # 使用生成的描述
                aggregation=aggregation,            # 本地直接调用
                for_non_aligned=True          # 非对齐数据部分也进行补充
            )
        # 创建客户端
        client = EnhancedClient(
                client_id=client_id,
                model_type=model_type,
                num_classes=len(processor.category_map),
                train_loader=train_loader,
                val_loader=val_loader,
                logger=logger
            )

        clients.append(client)
        

    # 训练客户端模型
    logger.info("开始训练...")

    # 串行训练客户端
    for i, client in enumerate(clients):
        logger.info(f"训练客户端 {i} ({client.model_type.name})...")
        client.train(epochs=Config.NUM_EPOCHS)
        client.evaluate()

    # 记录实验完成
    logger.info("实验完成!")


if __name__ == "__main__":
    # 在Windows上正确设置多进程启动方法
    if platform.system() == "Windows":
        mp.set_start_method('spawn', force=True)

    main()