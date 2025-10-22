import os
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from config_cub import Config, ModelType
from CUB_DataProcessor2 import DataProcessor
from client_CUB import EnhancedClient
from server_CUB import Server
from logger import Logger
import platform
import argparse
from generate_caption_CUB import generate_missing_captions
import time
import datetime
from collections import defaultdict

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

def federated_round(server, clients, processor, round_num, logger, client_acc, server_acc):
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

    # Step 2: 客户端使用非对齐数据提取inputs_embeds
    logger.info("Step 2: 客户端使用非对齐数据提取inputs_embeds...")
    client_non_aligned_features = defaultdict(list)
    client_non_aligned_labels = defaultdict(list)
    client_non_aligned_image_ids = defaultdict(list)
    class_prototypes = defaultdict(list)
    for client in clients:
        logger.info(f"提取客户端{client.client_id}的非对齐数据embedding")
        non_aligned_data = processor.client_data[client.client_id]["train"]["non_aligned"]
        non_aligned_loader = processor.get_dataloader(non_aligned_data, batch_size=Config.BATCH_SIZE, is_train=True)

        # 使用外部模型提取特征
        features, labels, image_id = client.extract_features_with_external_model(non_aligned_loader)
        client_non_aligned_features[client.client_id].append(features)
        client_non_aligned_labels[client.client_id].append(labels)
        client_non_aligned_image_ids[client.client_id].append(image_id)

        logger.info(f"提取客户端{client.client_id}的类别原型")
        class_prototypes[client.client_id] = processor.get_class_prototype(
            embeddings=features, 
            client_id=client.client_id,
            labels=labels,
            image_ids=image_id
        )
    # Step 3:提取服务端的非对齐数据并生成类别原型
    logger.info(f"Step 3:提取服务端的非对齐数据embedding")
    server_non_aligned_data = processor.client_data[Config.NUM_CLIENTS]["train"]["non_aligned"]
    server_non_aligned_loader = processor.get_dataloader(server_non_aligned_data, batch_size=Config.BATCH_SIZE, is_train=True)

    server_non_aligned_features, server_non_aligned_labels, server_non_aligned_image_ids = server.process_aligned_data_for_features(server_non_aligned_loader)

    logger.info("提取服务端的类别原型")
    class_prototypes[Config.NUM_CLIENTS] = processor.get_class_prototype(
        for_server=True, 
        embeddings=server_non_aligned_features,
        labels=server_non_aligned_labels,
        image_ids=server_non_aligned_image_ids
        )
    # Step 4:根据类别原型生成超原型
    logger.info("Step 4:根据类别原型生成超原型")
    prototypes, prototypes_labels = processor.get_prototype_from_cp_new(class_prototypes=class_prototypes)

    # Step 5:根据超原型计算各相似度矩阵
    logger.info("Step 5:根据超原型计算各相似度矩阵")
    sim_matrix = defaultdict(list)
    for client in clients:
        sim_matrix[client.client_id] = processor.get_sim_matrix(prototypes, client_non_aligned_features[client.client_id], zero_aligned=True)
    sim_matrix[Config.NUM_CLIENTS] = processor.get_sim_matrix(prototype=prototypes, client_data =server_non_aligned_features, for_server=True, zero_aligned=True)
    
    # Step 6:只提取文本特征
    logger.info("Step 6: 客户端使用非对齐数据提取文本特征...")
    text_client_non_aligned_features = defaultdict(list)
    text_client_non_aligned_labels = defaultdict(list)
    text_client_non_aligned_image_ids = defaultdict(list)
    text_client_non_aligned_embeddings = defaultdict(list)
    text_class_prototypes = defaultdict(list)
    text_class_prototypes_logits = defaultdict(list)
    text_class_prototypes_embeddding = defaultdict(list)
    for client in clients:
        logger.info(f"提取客户端{client.client_id}的非对齐数据文本特征")
        non_aligned_data = processor.client_data[client.client_id]["train"]["non_aligned"]
        non_aligned_loader = processor.get_dataloader(non_aligned_data, batch_size=Config.BATCH_SIZE, is_train=True)

        # 使用外部模型提取特征
        embeddings, features, labels, image_id, logits = client.extract_captions_with_external_model_new(non_aligned_loader)
        text_client_non_aligned_features[client.client_id].append(features)
        text_client_non_aligned_labels[client.client_id].append(labels)
        text_client_non_aligned_image_ids[client.client_id].append(image_id)
        text_client_non_aligned_embeddings[client.client_id].append(embeddings)

        logger.info(f"提取客户端{client.client_id}的文本类别原型")
        text_class_prototypes[client.client_id] = processor.get_class_prototype(
            embeddings=features, 
            client_id=client.client_id,
            labels=labels,
            image_ids=image_id
        )
        text_class_prototypes_logits[client.client_id] = processor.get_class_logits(
            text_logits=logits, 
            client_id=client.client_id,
            labels=labels,
            image_ids=image_id
        )
        text_class_prototypes_embeddding[client.client_id] = processor.get_class_embeddings(
            text_embeddings=embeddings, 
            client_id=client.client_id,
            labels=labels,
            image_ids=image_id
        )
    logger.info("提取服务端的文本特征")
    text_server_non_aligned_embeddings, text_server_non_aligned_features, text_server_non_aligned_labels, text_server_non_aligned_image_ids, text_server_non_aligned_logits = server.extract_captions(server_non_aligned_loader)

    logger.info("提取服务端的文本类别原型")
    text_class_prototypes[Config.NUM_CLIENTS] = processor.get_class_prototype(
        for_server=True, 
        embeddings=text_server_non_aligned_features,
        labels=text_server_non_aligned_labels,
        image_ids=text_server_non_aligned_image_ids
        )
    text_class_prototypes_logits[Config.NUM_CLIENTS] = processor.get_class_logits(
            for_server=True,
            text_logits=text_server_non_aligned_logits, 
            labels=text_server_non_aligned_labels,
            image_ids=text_server_non_aligned_image_ids
        )
    text_class_prototypes_embeddding[Config.NUM_CLIENTS] = processor.get_class_embeddings(
            for_server=True,
            text_embeddings=text_server_non_aligned_embeddings, 
            labels=text_server_non_aligned_labels,
            image_ids=text_server_non_aligned_image_ids
        )
    
    # Step 7:根据文本类别原型生成文本超原型
    logger.info("Step 7:根据类别原型生成超原型")
    text_prototypes, text_prototypes_labels = processor.get_prototype_from_cp_new(class_prototypes=text_class_prototypes)
    text_prototypes_logits = processor.get_prototype_logits_from_cp(class_prototypes_logits=text_class_prototypes_logits)
    text_prototypes_embeddings = processor.get_prototype_logits_from_cp(class_prototypes_logits=text_class_prototypes_embeddding)

    # Step 8:利用冻结模型生成参考logits
    logger.info("Step 8: 客户端利用参考模型使用非对齐数据提取文本特征...")

    llava_model_fronzen, llava_processor_fronzen = server.get_model_components_fronzen()

    for client in clients:
        client.set_external_model_components(llava_model_fronzen, llava_processor_fronzen)
    
    text_client_non_aligned_labels_fronzen = defaultdict(list)
    text_client_non_aligned_image_ids_fronzen = defaultdict(list)

    text_class_prototypes_logits_fronzen = defaultdict(list)
    for client in clients:
        logger.info(f"用参考模型提取客户端{client.client_id}的非对齐数据文本特征")
        non_aligned_data = processor.client_data[client.client_id]["train"]["non_aligned"]
        non_aligned_loader = processor.get_dataloader(non_aligned_data, batch_size=Config.BATCH_SIZE, is_train=True)

        # 使用外部模型提取特征
        _, _, labels, image_id, logits = client.extract_captions_with_external_model_new(non_aligned_loader)
        text_client_non_aligned_labels_fronzen[client.client_id].append(labels)
        text_client_non_aligned_image_ids_fronzen[client.client_id].append(image_id)

        logger.info(f"提取客户端{client.client_id}的文本类别原型logits")
        
        text_class_prototypes_logits_fronzen[client.client_id] = processor.get_class_logits(
            text_logits=logits, 
            client_id=client.client_id,
            labels=labels,
            image_ids=image_id
        )
    logger.info("提取服务端的文本特征")
    _, _, text_server_non_aligned_labels_fronzen, text_server_non_aligned_image_ids_fronzen, text_server_non_aligned_logits = server.extract_captions(server_non_aligned_loader)

    logger.info("提取服务端冻结的文本类别原型logits")
    text_class_prototypes_logits_fronzen[Config.NUM_CLIENTS] = processor.get_class_logits(
            for_server=True,
            text_logits=text_server_non_aligned_logits, 
            labels=text_server_non_aligned_labels,
            image_ids=text_server_non_aligned_image_ids
        )
    text_prototypes_logits_fronzen = processor.get_prototype_logits_from_cp(class_prototypes_logits=text_class_prototypes_logits_fronzen)

    # Step 9:服务端训练生成模型
    model_preferences, final_embeddings = server.train_for_caption_generator_with_infoNCE_loss_new(
        prototypes=prototypes, 
        sim_matrix=sim_matrix,
        dataloader=server_non_aligned_loader, 
        text_prototypes=text_prototypes,
        text_logits=text_prototypes_logits,
        text_logits_fronzen=text_prototypes_logits_fronzen,
        text_embedding = text_prototypes_embeddings
    )

    # Step 10:服务端下发组件各方生成描述
    llava_model, llava_processor = server.get_model_components()

    for client in clients:
        client.set_external_model_components(llava_model, llava_processor)
        non_aligned_data = processor.client_data[client.client_id]["train"]["non_aligned"]
        non_aligned_loader = processor.get_dataloader(non_aligned_data, batch_size=Config.BATCH_SIZE, is_train=True)
        client.generate_captions(dataloader=non_aligned_loader)

    non_aligned_data = processor.client_data[Config.NUM_CLIENTS]["train"]["non_aligned"]
    non_aligned_loader = processor.get_dataloader(non_aligned_data, batch_size=Config.BATCH_SIZE, is_train=True)

    server.generate_captions_new(dataloader=non_aligned_loader)
    
    # Step 11:根据各相似度矩阵生成伪对齐数据
    logger.info("Step 11:根据各相似度矩阵生成伪对齐数据")
    aligned_pairs=processor.match_aligned_data_new(sim_matrix)

    attention_mask, attention_mask_aligned = processor.append_aligned_data_new(
            aligned_pairs, 
            prototypes=prototypes, 
            client_features = client_non_aligned_features,
            server_features = server_non_aligned_features,
            text_prototypes = text_prototypes,
            text_client_features = text_client_non_aligned_features,
            text_server_features = text_server_non_aligned_features,
            prototypes_labels = prototypes_labels,
            text_prototypes_labels = text_prototypes_labels
        )

    logger.info("迭代扩充超原型个数")
    i = 0 
    while len(prototypes) < Config.NUM_PROTOTYPE:
        i += 1
        if i == Config.max_iter:
            break
        sim_matrix = defaultdict(list)
        for client in clients:
            sim_matrix[client.client_id] = processor.get_sim_matrix(prototypes, client_non_aligned_features[client.client_id])
        sim_matrix[Config.NUM_CLIENTS] = processor.get_sim_matrix(prototype=prototypes, client_data =server_non_aligned_features, for_server=True)

        # 根据各相似度矩阵生成伪对齐数据
        aligned_pairs=processor.match_aligned_data_new(sim_matrix)
        attention_mask, attention_mask_aligned = processor.append_aligned_data_new(
            aligned_pairs, 
            prototypes=prototypes, 
            attention_mask = attention_mask,
            attention_mask_aligned = attention_mask_aligned,
            client_features = client_non_aligned_features,
            server_features = server_non_aligned_features,
            text_prototypes = text_prototypes,
            text_client_features = text_client_non_aligned_features,
            text_server_features = text_server_non_aligned_features,
            prototypes_labels = prototypes_labels,
            text_prototypes_labels = text_prototypes_labels
        )

    for client in clients:       
        logger.info(f"为客户端{client.client_id}新准备数据...")
        model_type = Config.get_client_model(client.client_id)
        logger.info(f"Client {client.client_id}: {model_type.name}")
        # 获取数据加载器
        train_loader, val_loader = processor.get_client_dataloaders(
            client_id=client.client_id,
            use_generated_captions=True,  # 使用生成的描述
            aggregation=False,       #选择不聚合
            for_non_aligned=True,
            for_local_non_aligned=False
            )
        # 创建增强版客户端
        client.upgate_dataloaders(train_loader, val_loader)
    logger.info("为服务端重新准备数据...")
    server_train_loader,server_val_loader = processor.get_client_dataloaders(
        client_id=Config.NUM_CLIENTS,
        use_generated_captions=True,  # 使用生成的描述
        aggregation=False,       #选择不聚合
        for_non_aligned=True,
        for_server=True,
        for_local_non_aligned=False
        )
    # 初始化服务端
    logger.info(f"更新服务端(主动方)...")
    server.upgate_dataloaders(server_train_loader, server_val_loader)


    # # Step 12: 客户端重新使用对齐数据提取inputs_embeds
    # logger.info("Step 12: 客户端使用扩充后的对齐数据提取inputs_embeds...")
    # client_features_list = []
    # client_labels_list = []
    # llava_model, llava_processor = server.get_model_components()
    # for client in clients:
    #     client.set_external_model_components(llava_model, llava_processor)
    # for client in clients:
    #     # 获取客户端的对齐数据加载器
    #     aligned_data = processor.client_data[client.client_id]["train"]["aligned"]
    #     aligned_loader = processor.get_dataloader(aligned_data, batch_size=Config.BATCH_SIZE, is_train=True)

    #     # 使用外部模型提取特征
    #     features, labels, image_ids = client.extract_features_with_external_model(aligned_loader)
    #     client_features_list.append(features)
    #     client_labels_list.append(labels)

    # # Step 13: 服务端重新处理自己的对齐数据
    # logger.info("Step 13: 服务端处理扩充后的对齐数据...")
    # # 服务端使用相同的对齐数据（这里使用第一个客户端的对齐数据，因为内容一致）
    # server_aligned_data = processor.client_data[Config.NUM_CLIENTS]["train"]["aligned"]
    # server_aligned_loader = processor.get_dataloader(server_aligned_data, batch_size=Config.BATCH_SIZE, is_train=True)

    # server_features, server_labels, image_ids = server.process_aligned_data_for_features(server_aligned_loader)

    # # Step 14: 融合特征并训练
    # logger.info("Step 14: 融合三方特征并训练...")
    # # 确保所有标签一致（因为对齐数据相同）
    # fused_features = server.fuse_features(server_features, client_features_list)

    # # 使用融合特征训练
    # loss, acc = server.train_with_fused_features(fused_features, server_labels)

    # Step 15: 服务端使用非对齐数据继续训练
    logger.info("Step 15: 服务端使用非对齐数据训练...")
    server.train_with_local_data_new(model_preferences = model_preferences, final_embeddings = final_embeddings, round = round_num)

    # Step 16: 评估服务端性能
    logger.info("Step 16: 评估服务端性能...")
    server_val_loss, server_val_acc = server.evaluate()
    logger.info(f"服务端验证结果 - Loss: {server_val_loss:.4f}, Acc: {server_val_acc:.2f}%")

    # Step 17: 服务端再次下发更新后的模型组件
    logger.info("Step 17: 服务端下发更新后的模型组件...")
    llava_model, llava_processor = server.get_model_components()

    for client in clients:
        client.set_external_model_components(llava_model, llava_processor)

    # Step 18: 客户端使用全部数据更新本地模型
    logger.info("Step 18: 客户端使用全部数据更新本地模型...")
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
    server.log_communication_stats()

    logger.info(f"===== 联邦学习第 {round_num} 轮结束 =====共用时{time_str}\n")

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
    logger.info("Baseline 4: 引入主动方（LLaVA-7B）的垂直联邦学习， 并使用Llava生成缺失描述的模型训练")

    # 加载和预处理数据
    logger.info("加载和预处理数据...")
    processor = DataProcessor(logger)
    data = processor.load_and_preprocess_data()

    # 为客户端划分数据
    logger.log_timing_start("data_splitting")
    client_data = processor.split_data_for_clients(data, num_clients=Config.NUM_CLIENTS+1)
    logger.log_timing_end("data_splitting")

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
                num_classes=len(processor.class_id_to_name),
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
            num_classes=len(processor.class_id_to_name),
            train_loader=train_loader,
            val_loader=val_loader,
            logger=logger
        )
        clients.append(client)

    # 执行联邦学习
    logger.info(f"开始联邦学习，共 {args.rounds} 轮...")

    client_acc = defaultdict(lambda: {"train_acc": [], "val_acc": []})
    server_acc = {"train_acc": {"aligned": [],"non_aligned": []}, "val_acc": []}

    for round_num in range(1, args.rounds + 1):
        federated_round(server, clients, processor, round_num, logger, client_acc=client_acc, server_acc=server_acc)

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
