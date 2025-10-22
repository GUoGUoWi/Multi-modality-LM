import os
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from config import Config, ModelType
from data_processor_scenario2 import DataProcessor
from client import EnhancedClient
from server import Server
from logger import Logger
import platform
import argparse
from generate_caption import generate_missing_captions
import time
import datetime

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


def federated_round(server, clients, processor, round_num, logger):
    """
    执行一轮联邦学习

    Args:
        server: 服务端实例
        clients: 客户端列表
        processor: 数据处理器
        round_num: 当前轮次
        logger: 日志记录器
    """
    # 初始化计时统计
    round_start_time = time.time()

    logger.info(f"\n===== 联邦学习第 {round_num} 轮开始 =====")

    # Step 1: 服务端下发模型组件
    logger.info("Step 1: 服务端下发LLaVA模型组件到客户端...")
    llava_model, llava_processor = server.get_model_components()

    for client in clients:
        client.set_external_model_components(llava_model, llava_processor)

    # # Step 2: 客户端使用对齐数据提取inputs_embeds
    # logger.info("Step 2: 客户端使用对齐数据提取inputs_embeds...")
    # client_features_list = []
    # client_labels_list = []

    # for client in clients:
    #     # 获取客户端的对齐数据加载器
    #     aligned_data = processor.client_data[client.client_id]["train"]["aligned"]
    #     aligned_loader = processor.get_dataloader(aligned_data, batch_size=Config.BATCH_SIZE, is_train=True)

    #     # 使用外部模型提取特征
    #     features, labels,_ = client.extract_features_with_external_model(aligned_loader)
    #     client_features_list.append(features)
    #     client_labels_list.append(labels)

    # # Step 3: 服务端处理自己的对齐数据
    # logger.info("Step 3: 服务端处理对齐数据...")
    # # 服务端使用相同的对齐数据（这里使用第一个客户端的对齐数据，因为内容一致）
    # server_aligned_data = processor.client_data[2]["train"]["aligned"]
    # server_aligned_loader = processor.get_dataloader(server_aligned_data, batch_size=Config.BATCH_SIZE, is_train=True)

    # server_features, server_labels,_ = server.process_aligned_data_for_features(server_aligned_loader)

    # # Step 4: 融合特征并训练
    # logger.info("Step 4: 融合三方特征并训练...")
    # # 确保所有标签一致（因为对齐数据相同）
    # fused_features = server.fuse_features(server_features, client_features_list)

    # # 使用融合特征训练
    # loss, acc = server.train_with_fused_features(fused_features, server_labels, epochs=Config.NUM_EPOCHS)

    # Step 5: 服务端使用非对齐数据继续训练
    logger.info("Step 5: 服务端使用非对齐数据训练...")
    server.train_with_local_data(epochs=Config.NUM_EPOCHS)

    # Step 6: 评估服务端性能
    logger.info("Step 6: 评估服务端性能...")
    server_val_loss, server_val_acc = server.evaluate()
    logger.info(f"服务端验证结果 - Loss: {server_val_loss:.4f}, Acc: {server_val_acc:.2f}%")

    # Step 7: 服务端再次下发更新后的模型组件
    logger.info("Step 7: 服务端下发更新后的模型组件...")
    llava_model, llava_processor = server.get_model_components()

    for client in clients:
        client.set_external_model_components(llava_model, llava_processor)

    # Step 8: 客户端使用全部数据更新本地模型
    logger.info("Step 8: 客户端使用全部数据更新本地模型...")
    for client in clients:
        logger.info(f"训练客户端 {client.client_id}...")

        # 训练一个epoch
        client.train(epochs=Config.NUM_EPOCHS)

        # 评估客户端性能
        val_loss, val_acc = client.evaluate()
        logger.info(f"客户端 {client.client_id} 验证结果 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    # 计算总时间
    total_round_time = time.time() - round_start_time

    time_delta = datetime.timedelta(seconds=total_round_time)
    time_str = str(time_delta).split('.')[0]  

    logger.info(f"===== 联邦学习第 {round_num} 轮结束 ===== 共用时{time_str}\n")


def main():
    parser = argparse.ArgumentParser(description="Baseline 3: 引入主动方的垂直联邦学习")
    parser.add_argument("--seed", type=int, default=Config.SEED, help="随机种子")
    parser.add_argument("--rounds", type=int, default=Config.NUM_ROUNDS, help="联邦学习轮数")
    args = parser.parse_args()
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)

    # 设置随机种子
    set_random_seed(args.seed)

    # 创建日志记录器
    logger = Logger(Config.LOG_DIR)
    logger.info("Baseline 4: 引入主动方（LLaVA-7B）的垂直联邦学习， 并使用Bli生成缺失文本的模型训练")

    # 加载和预处理数据
    logger.info("加载和预处理数据...")
    processor = DataProcessor(logger)
    data = processor.load_and_preprocess_data()

    # 为客户端划分数据
    logger.log_timing_start("data_splitting")
    client_data = processor.split_data_for_clients(data, num_clients=Config.NUM_CLIENTS+1)
    logger.log_timing_end("data_splitting")

    # 为客户端生成缺失描述
    aggregation = True      # baseline4 使用聚合后大模型
    generate_missing_captions(client_data, aggregation=aggregation)

    # 统计生成描述数量
    generated_captions_dir = os.path.join(Config.OUTPUT_DIR, "generated_captions")
    gen_captions_count = 0
    for root, dirs, files in os.walk(generated_captions_dir):
        gen_captions_count += sum(1 for f in files if f.lower().endswith(".txt"))
    logger.info(f"找到 {gen_captions_count} 条生成的描述")

    # 初始化客户端
    logger.info("初始化客户端...")
    clients = []
    for client_id in range(Config.NUM_CLIENTS + 1):
        if client_id == 2:
            logger.info("为服务端准备数据...")
            server_train_loader,server_val_loader = processor.get_client_dataloaders(
                client_id=client_id,
                use_generated_captions=True,  # 使用生成的描述
                aggregation=True,       #选择聚合
                for_non_aligned=False,
                for_server=True,
                for_local_non_aligned=False
                )
            # 初始化服务端
            logger.info("初始化服务端（主动方）...")
            server = Server(
                num_classes=len(processor.category_map),
                train_loader=server_train_loader,
                val_loader=server_val_loader,
                logger=logger
            )
            continue
        model_type = Config.get_client_model(client_id)
        logger.info(f"Client {client_id}: {model_type.name} - 使用生成图像补全")
        # 获取数据加载器
        train_loader, val_loader = processor.get_client_dataloaders(
            client_id=client_id,
            use_generated_captions=True,  # 使用生成的描述
            aggregation=True,       #选择聚合
            for_non_aligned=False,
            for_local_non_aligned=False
        )
        # 创建增强版客户端
        client = EnhancedClient(
            client_id=client_id,
            model_type=model_type,
            num_classes=len(processor.category_map),
            train_loader=train_loader,
            val_loader=val_loader,
            logger=logger
        )
        clients.append(client)

    # 执行联邦学习
    logger.info(f"开始联邦学习，共 {args.rounds} 轮...")

    for round_num in range(1, args.rounds + 1):
        federated_round(server, clients, processor, round_num, logger)

    # 最终评估
    logger.info("\n===== 最终评估 =====")

    # 评估服务端
    server_val_loss, server_val_acc = server.evaluate()
    logger.info(f"服务端最终性能 - Loss: {server_val_loss:.4f}, Acc: {server_val_acc:.2f}%")

    # 评估各客户端
    for client in clients:
        val_loss, val_acc = client.evaluate()
        logger.info(f"客户端 {client.client_id} 最终性能 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    # 记录通信统计
    server.log_communication_stats()

    # 记录实验完成
    logger.info("实验完成!")
    logger.finalize()



if __name__ == "__main__":
    # 在Windows上正确设置多进程启动方法
    if platform.system() == "Windows":
        mp.set_start_method('spawn', force=True)

    main()