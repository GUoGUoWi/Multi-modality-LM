import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config_cub import Config
from models_CUB import get_model
from PIL import Image
import os
from torchvision import transforms


def calculate_accuracy(outputs, targets):
    """
    计算分类准确率

    Args:
        outputs: 模型输出
        targets: 目标标签

    Returns:
        准确率（百分比形式，如75.0000表示75%）
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return (correct / total) * 100.0  # 转换为百分比


class Client:
    """联邦学习客户端"""

    def __init__(self, client_id, model_type, num_classes, train_loader, val_loader, logger=None):
        """
        初始化客户端

        Args:
            client_id: 客户端ID
            model_type: 模型类型
            num_classes: 分类类别数
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            logger: 日志记录器
        """
        self.client_id = client_id
        self.model_type = model_type
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.device = Config.DEVICE

        # 初始化模型
        self.model = get_model(model_type, num_classes, device=Config.DEVICE)

        # 记录模型参数数量统计
        if logger:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"客户端 {client_id} 模型参数: {trainable_params:,} 可训练 / {total_params:,} 总计")

        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

    def extract_features(self, dataloader):
        """
        从数据加载器中提取特征和标签

        Args:
            dataloader: 数据加载器

        Returns:
            特征和标签
        """
        self.model.eval()
        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"客户端 {self.client_id} 提取特征"):
                # 预处理输入
                inputs = self.model.preprocess(batch)

                # 提取特征 (这里调用的是模型的_extract_features方法)
                features = self.model._extract_features(inputs)

                # 获取标签
                labels = batch["label"].to(self.model.device)

                # 收集特征和标签
                all_features.append(features)
                all_labels.append(labels)

        # 连接所有特征和标签
        if all_features and all_labels:
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            if self.logger:
                self.logger.info(
                    f"客户端 {self.client_id} 提取了 {all_features.shape[0]} 个特征向量，维度为 {all_features.shape[1]}")

            return all_features, all_labels
        else:
            if self.logger:
                self.logger.warning(f"客户端 {self.client_id} 没有提取到特征，可能是数据集为空")
            # 返回空的特征和标签
            return torch.tensor([]), torch.tensor([])

    def extract_modal_features(self, dataloader):
        """
        从数据加载器中分别提取文本和图像pooled特征以及标签

        Args:
            dataloader: 数据加载器

        Returns:
            文本pooled特征、图像pooled特征和标签
        """
        self.model.eval()
        all_text_features = []
        all_image_features = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"客户端 {self.client_id} 提取模态特征"):
                try:
                    # 预处理输入
                    inputs = self.model.preprocess(batch)

                    # 提取文本和图像pooled特征 - 使用模型的_extract_modal_features方法
                    text_features, image_features = self.model._extract_modal_features(inputs)

                    # 获取标签
                    labels = batch["label"].to(self.model.device)

                    # 收集特征和标签
                    all_text_features.append(text_features)
                    all_image_features.append(image_features)
                    all_labels.append(labels)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"提取模态特征时出错: {str(e)}")
                    print(f"提取模态特征时出错: {str(e)}")
                    continue

        # 连接所有特征和标签
        if all_text_features and all_image_features and all_labels:
            all_text_features = torch.cat(all_text_features, dim=0)
            all_image_features = torch.cat(all_image_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            if self.logger:
                self.logger.info(
                    f"客户端 {self.client_id} 提取了 {all_text_features.shape[0]} 个文本特征和 {all_image_features.shape[0]} 个图像特征")

            return all_text_features, all_image_features, all_labels
        else:
            if self.logger:
                self.logger.warning(f"客户端 {self.client_id} 没有提取到特征，可能是数据集为空")
            # 返回空的特征和标签
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

    def train_with_features(self, mapped_features_dict, labels, epochs=5):
        """
        使用映射回的多模态特征训练分类器

        Args:
            mapped_features_dict: 包含映射回的特征的字典，格式为 {modality: tensor}
            labels: 标签
            epochs: 训练轮数

        Returns:
            训练历史
        """
        history = {'loss': [], 'acc': []}

        # 确保数据在正确的设备上
        for modality in mapped_features_dict:
            mapped_features_dict[modality] = mapped_features_dict[modality].to(self.model.device)

        labels = labels.to(self.model.device)

        # 只训练分类头
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        # 创建针对分类头的优化器
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=Config.LEARNING_RATE)

        # 准备批量处理
        data_size = labels.size(0)
        batch_size = min(32, data_size)  # 默认批次大小

        # 使用tqdm包装epochs
        for epoch in tqdm(range(epochs), desc=f"客户端 {self.client_id} 基于映射特征训练", ncols=100):
            # 设置模型为训练模式
            self.model.classifier.train()

            # 随机打乱数据顺序
            indices = torch.randperm(data_size, device=self.model.device)

            epoch_loss = 0.0
            epoch_correct = 0

            # 分批次训练
            for i in range(0, data_size, batch_size):
                # 获取当前批次索引
                batch_indices = indices[i:min(i + batch_size, data_size)]

                # 准备批次数据
                batch_features = {}
                for modality in mapped_features_dict:
                    batch_features[modality] = mapped_features_dict[modality][batch_indices]

                batch_labels = labels[batch_indices]

                # 前向传播
                outputs = self.model.classify_with_features(batch_features)
                loss = self.criterion(outputs, batch_labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 计算准确率
                _, predicted = torch.max(outputs, 1)
                batch_correct = (predicted == batch_labels).sum().item()

                # 累积损失和正确数
                epoch_loss += loss.item() * len(batch_indices)
                epoch_correct += batch_correct

            # 计算平均损失和准确率
            epoch_loss /= data_size
            epoch_acc = 100.0 * epoch_correct / data_size

            # 记录指标
            history['loss'].append(epoch_loss)
            history['acc'].append(epoch_acc)

            # 打印进度
            if self.logger:
                self.logger.info(
                    f"客户端 {self.client_id} 基于映射特征训练 - Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}, Acc: {epoch_acc:.4f}%")

        # 恢复所有参数可训练状态
        for param in self.model.parameters():
            param.requires_grad = True

        return history

    def get_classifier_function(self):
        """
        获取分类器函数，用于评估映射准确率

        Returns:
            分类器函数
        """

        def classifier_fn(features):
            # 确保模型处于评估模式
            self.model.eval()
            # 使用模型的分类头进行分类
            with torch.no_grad():
                return self.model.classifier(features)

        return classifier_fn
    
    def get_inner_dataset_and_local_idx(self, concat_dataset, idx):
        for dataset in concat_dataset.datasets:
            if idx < len(dataset):
                return dataset, idx
            else:
                idx -= len(dataset)
        raise IndexError(f"Index {idx} out of range")


    def train(self, epochs=Config.NUM_EPOCHS):
        """
        训练模型

        Args:
            epochs: 训练轮数

        Returns:
            训练历史记录
        """
        self.model.train()
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        for epoch in range(1, epochs + 1):
            train_loss = 0.0
            train_acc = 0.0

            # 训练一个epoch
            self.model.train()
            train_start = time.time()

            try:
                with tqdm(self.train_loader, desc=f"Client {self.client_id} - Epoch {epoch}/{epochs}") as pbar:
                    for batch_idx, batch in enumerate(pbar):
                        # 检查批次数据是否有效
                        if "label" not in batch or batch["label"] is None or batch["label"].numel() == 0:
                            print(f"跳过批次 {batch_idx} - 批次数据无效")
                            continue

                        try:
                            # idxs = batch["idx"]
                            # for i in range(len(batch["idx"])):
                            #     if batch["text"][i] == "":
                            #         dataset, local_idx = self.get_inner_dataset_and_local_idx(self.train_loader.dataset, batch["idx"][i].item())
                            #         dataset.data_list[local_idx]["has_caption"] = False
                            #         dataset.data_list[local_idx]["is_generated_caption"] = False
                            #         batch["has_caption"][i] = False
                            #         batch["is_generated_caption"][i] = False
                            # 预处理输入
                            inputs = self.model.preprocess(batch)
                            labels = batch["label"].to(self.model.device)

                            # 前向传播
                            self.optimizer.zero_grad()
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, labels)

                            # 反向传播
                            loss.backward()
                            self.optimizer.step()

                            # 计算准确率
                            acc = calculate_accuracy(outputs, labels)

                            # 累加损失和准确率
                            train_loss += loss.item()
                            train_acc += acc

                            # 更新进度条 - 只显示损失，不显示准确率
                            pbar.set_postfix({
                                "loss": f"{loss.item():.4f}"
                            })
                        except Exception as e:
                            if self.logger:
                                self.logger.error(f"处理批次 {batch_idx} 时出错: {str(e)}")
                            print(f"处理批次 {batch_idx} 时出错: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            continue
            except Exception as e:
                if self.logger:
                    self.logger.error(f"训练过程中出错: {str(e)}")
                print(f"训练过程中出错: {str(e)}")
                import traceback
                traceback.print_exc()

            # 计算平均损失和准确率
            num_batches = len(self.train_loader)
            if num_batches > 0:
                train_loss = train_loss / num_batches
                train_acc = train_acc / num_batches

            # 记录训练用时
            train_time = time.time() - train_start
            if self.logger:
                self.logger.log_timing(f"client_{self.client_id}_epoch_{epoch}_train", train_time)

            # 验证
            if epoch % Config.VALIDATION_INTERVAL == 0:
                val_loss, val_acc = self.evaluate()
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                # 记录日志
                if self.logger:
                    self.logger.log_epoch(
                        self.client_id,
                        epoch,
                        train_loss,
                        train_acc,
                        val_loss,
                        val_acc
                    )
            else:
                # 仅记录训练日志
                if self.logger:
                    self.logger.log_epoch(
                        self.client_id,
                        epoch,
                        train_loss,
                        train_acc
                    )

            # 更新历史记录
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

        return history

    def evaluate(self):
        """
        评估模型

        Returns:
            损失和准确率
        """
        self.model.eval()
        val_loss = 0.0
        val_acc = 0.0

        # 开始评估
        val_start = time.time()

        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    # 预处理输入
                    # idxs = batch["idx"]
                    # for i in range(len(batch["idx"])):
                    #     if batch["text"][i] == "":
                    #         dataset, local_idx = self.get_inner_dataset_and_local_idx(self.train_loader.dataset, batch["idx"][i].item())
                    #         dataset.data_list[local_idx]["has_caption"] = False
                    #         dataset.data_list[local_idx]["is_generated_caption"] = False
                    #         batch["has_caption"][i] = False
                    #         batch["is_generated_caption"][i] = False
                    inputs = self.model.preprocess(batch)
                    labels = batch["label"].to(self.model.device)

                    # 前向传播
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    # 计算准确率
                    acc = calculate_accuracy(outputs, labels)

                    # 累加损失和准确率
                    val_loss += loss.item()
                    val_acc += acc
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"评估批次时出错: {str(e)}")
                    print(f"评估批次时出错: {str(e)}")
                    continue

        # 计算平均损失和准确率
        num_batches = len(self.val_loader)
        if num_batches > 0:
            val_loss = val_loss / num_batches
            val_acc = val_acc / num_batches

        # 记录评估用时
        val_time = time.time() - val_start
        if self.logger:
            self.logger.log_timing(f"client_{self.client_id}_evaluate", val_time)

        return val_loss, val_acc

    def get_model_state(self):
        """
        获取模型状态

        Returns:
            模型状态字典
        """
        return self.model.state_dict()

    def set_model_state(self, state_dict):
        """
        设置模型状态

        Args:
            state_dict: 模型状态字典
        """
        self.model.load_state_dict(state_dict)


class EnhancedClient(Client):
    """增强版客户端，支持使用外部LLaVA模型"""

    def __init__(self, client_id, model_type, num_classes, train_loader, val_loader, logger=None):
        super().__init__(client_id, model_type, num_classes, train_loader, val_loader, logger)

        # 存储外部模型组件
        self.external_model = None
        self.external_processor = None
        self.use_external_model = False
        self.config = None
        self.pad_token_id = -1

    def upgate_dataloaders(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_external_model_components(self, llava_model, processor):
        """
        设置外部模型组件

        Args:
            llava_model: LLaVA模型实例
            processor: LLaVA处理器
        """
        self.external_model = llava_model
        self.external_processor = processor
        self.use_external_model = True
        # 保存LLaVA模型的配置
        self.config = llava_model.config
        # 更新pad_token_id
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        if self.logger:
            self.logger.info(f"客户端 {self.client_id} 设置了外部LLaVA模型组件")

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        image_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def extract_features_with_external_model(self, dataloader):
        """
        使用外部LLaVA模型提取特征

        Args:
            dataloader: 数据加载器

        Returns:
            模型特征和标签
        """
        if not self.use_external_model:
            raise ValueError("未设置外部模型组件")

        all_features = []
        all_labels = []
        all_image_ids = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"客户端 {self.client_id} 使用LLaVA提取特征"):
                # 获取批次数据
                texts = batch["text"]
                images = batch["image"]  # [batch_size, 3, H, W]
                labels = batch["label"]
                image_id = batch["image_id"]

                has_image = batch.get("has_image", torch.ones(len(texts), dtype=torch.bool))
                has_caption = batch.get("has_caption", torch.ones(len(images), dtype=torch.bool))
                batch_features = []
                # 逐个处理样本
                for i in range(len(texts)):
                    text = texts[i]
                    
                    if has_image[i] and has_caption[i]:
                        # 有文本与图像的情况
                        # 将张量转换为PIL图像
                        img_tensor = images[i].cpu()
                        # 反归一化
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_tensor = img_tensor * std + mean
                        img_tensor = torch.clamp(img_tensor, 0, 1)
                        # 转换为PIL
                        import numpy as np
                        from PIL import Image as PILImage
                        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        img_pil = PILImage.fromarray(img_np)
                        # 使用LLaVA的prompt模板
                        prompt = f"USER: <image>\n{text}\nASSISTANT:"
                        # 使用LLaVA处理器
                        inputs = self.external_processor(
                            text=prompt,
                            images=img_pil,
                            return_tensors="pt"
                        )
                    elif not has_image[i] and has_caption[i]:
                        # 纯文本情况
                        prompt = f"USER: {text}\nASSISTANT:"
                        inputs = self.external_processor(
                            text=prompt,
                            return_tensors="pt"
                        )
                    else:
                        #纯图像情况
                        # 将张量转换为PIL图像
                        img_tensor = images[i].cpu()
                        # 反归一化
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_tensor = img_tensor * std + mean
                        img_tensor = torch.clamp(img_tensor, 0, 1)
                        # 转换为PIL
                        import numpy as np
                        from PIL import Image as PILImage
                        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        img_pil = PILImage.fromarray(img_np)
                        # 使用LLaVA的prompt模板
                        prompt = f"USER: <image>\nASSISTANT:"
                        # 使用LLaVA处理器
                        inputs = self.external_processor(
                            text=prompt,
                            images=img_pil,
                            return_tensors="pt"
                        )

                    # 将inputs移到正确的设备
                    inputs = {k: v.to(self.external_model.device) if isinstance(v, torch.Tensor) else v
                              for k, v in inputs.items()}  ###字典推导式{k: <>for k,v in 迭代}
                    
                    # 使用get_model_input_embeddings获取融合后的输入嵌入
                    inputs_embeds = self.external_model.get_model_input_embeddings(
                        input_ids=inputs.get('input_ids'),
                        pixel_values=inputs.get('pixel_values'),
                        attention_mask=inputs.get('attention_mask')
                    )
                    # 通过模型获取隐藏状态输出
                    # 构建新的输入字典，使用inputs_embeds替代input_ids和pixel_values
                    forward_inputs = {
                        'inputs_embeds': inputs_embeds,
                        'attention_mask': inputs.get('attention_mask'),
                        'output_hidden_states': True,  # 确保输出所有隐藏状态
                        'return_dict': True
                    }
                    # 前向传播获取隐藏状态
                    outputs = self.external_model.language_model(**forward_inputs)
                    # 获取最后一层的隐藏状态
                    last_hidden_states = outputs.hidden_states[-1]
                    features = last_hidden_states[:, -1, :]  # 使用最后一个token

                    batch_features.append(features.squeeze(0))

                # 堆叠批次特征
                batch_features = torch.stack(batch_features)

                all_features.append(batch_features)
                all_labels.append(labels.to(self.device))
                all_image_ids.append(image_id.to(self.device))

        # 连接所有批次
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_image_ids = torch.cat(all_image_ids, dim=0)

        return all_features, all_labels, all_image_ids
    
    def extract_captions_with_external_model(self, dataloader):
        """
        使用外部LLaVA模型提取非对齐数据 文本特征

        Args:
            dataloader: 数据加载器

        Returns:
            模型特征和标签
        """
        if not self.use_external_model:
            raise ValueError("未设置外部模型组件")

        all_features = []
        all_labels = []
        all_image_ids = []
        all_logits = []

        with torch.no_grad():
            for batch_index, batch in enumerate(tqdm(dataloader, desc=f"客户端 {self.client_id} 使用LLaVA提取文本特征")):
                # 获取批次数据
                texts = batch["text"]
                images = batch["image"]  # [batch_size, 3, H, W]
                labels = batch["label"]
                image_id = batch["image_id"]

                has_caption = batch.get("has_caption", torch.ones(len(images), dtype=torch.bool))
                batch_features = []
                # 逐个处理样本
                for i in range(len(texts)):
                    text = texts[i]
                    
                    if has_caption[i]:

                        prompt = f"USER: {text}\nASSISTANT:"
                        inputs = self.external_processor(
                            text=prompt,
                            return_tensors="pt"
                        )

                        # 将inputs移到正确的设备
                        inputs = {k: v.to(self.external_model.device) if isinstance(v, torch.Tensor) else v
                                for k, v in inputs.items()}  ###字典推导式{k: <>for k,v in 迭代}
                        
                        # 使用get_model_input_embeddings获取融合后的输入嵌入
                        inputs_embeds = self.external_model.get_model_input_embeddings(
                            input_ids=inputs.get('input_ids'),
                            pixel_values=inputs.get('pixel_values'),
                            attention_mask=inputs.get('attention_mask')
                        )
                        # 通过模型获取隐藏状态输出
                        # 构建新的输入字典，使用inputs_embeds替代input_ids和pixel_values
                        forward_inputs = {
                            'inputs_embeds': inputs_embeds,
                            'attention_mask': inputs.get('attention_mask'),
                            'output_hidden_states': True,  # 确保输出所有隐藏状态
                            'return_dict': True
                        }
                        # 前向传播获取隐藏状态
                        outputs = self.external_model.language_model(**forward_inputs)
                        # 获取最后一层的隐藏状态
                        last_hidden_states = outputs.hidden_states[-1]
                        features = last_hidden_states[:, -1, :]  # 使用最后一个token
                        logits = self.external_model.language_model.lm_head(features)
                        logits_for_generate =self.external_model.language_model.lm_head(last_hidden_states).mean(dim=1)
                        # [1,26,32064]--->[1,32064]
                        #未进行softmax

                        batch_features.append(features.squeeze(0))
                        all_labels.append(labels[i].to(self.device))
                        all_image_ids.append(image_id[i].to(self.device))
                        all_logits.append(logits.to(self.device))
                    else:
                        continue
                if batch_features:  
                    batch_features = torch.stack(batch_features)
                    all_features.append(batch_features)
                else:
                    print(f"[Warning] 客户端 {self.client_id} 批次 {batch_index} 没有有效样本，已跳过")


        # 连接所有批次
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.stack(all_labels)  
        all_image_ids = torch.stack(all_image_ids)
        all_logits = torch.stack(all_logits) #[num, 1, 26, 32064]

        return all_features, all_labels, all_image_ids, all_logits
    
    def extract_captions_with_external_model_new(self, dataloader):
        """
        使用外部LLaVA模型提取非对齐数据 文本特征

        Args:
            dataloader: 数据加载器

        Returns:
            模型特征和标签
        """
        if not self.use_external_model:
            raise ValueError("未设置外部模型组件")

        all_features = []
        all_labels = []
        all_image_ids = []
        all_logits = []
        all_embeddings = []

        with torch.no_grad():
            for batch_index, batch in enumerate(tqdm(dataloader, desc=f"客户端 {self.client_id} 使用LLaVA提取文本特征")):
                # 获取批次数据
                texts = batch["text"]
                images = batch["image"]  # [batch_size, 3, H, W]
                labels = batch["label"]
                image_id = batch["image_id"]

                has_caption = batch.get("has_caption", torch.ones(len(images), dtype=torch.bool))
                batch_features = []
                batch_embeddings = []
                # 逐个处理样本
                for i in range(len(texts)):
                    text = texts[i]
                    
                    if has_caption[i]:

                        prompt = f"USER: {text}\nASSISTANT:"
                        inputs = self.external_processor(
                            text=prompt,
                            return_tensors="pt"
                        )

                        # 将inputs移到正确的设备
                        inputs = {k: v.to(self.external_model.device) if isinstance(v, torch.Tensor) else v
                                for k, v in inputs.items()}  ###字典推导式{k: <>for k,v in 迭代}
                        
                        # 使用get_model_input_embeddings获取融合后的输入嵌入
                        inputs_embeds = self.external_model.get_model_input_embeddings(
                            input_ids=inputs.get('input_ids'),
                            pixel_values=inputs.get('pixel_values'),
                            attention_mask=inputs.get('attention_mask')
                        )
                        # 通过模型获取隐藏状态输出
                        # 构建新的输入字典，使用inputs_embeds替代input_ids和pixel_values
                        forward_inputs = {
                            'inputs_embeds': inputs_embeds,
                            'attention_mask': inputs.get('attention_mask'),
                            'output_hidden_states': True,  # 确保输出所有隐藏状态
                            'return_dict': True
                        }
                        # 前向传播获取隐藏状态
                        outputs = self.external_model.language_model(**forward_inputs)
                        # 获取最后一层的隐藏状态
                        last_hidden_states = outputs.hidden_states[-1]
                        features = last_hidden_states[:, -1, :]  # 使用最后一个token
                        embeddings = last_hidden_states.mean(dim=1)
                        logits = self.external_model.language_model.lm_head(features)
                        logits_for_generate =self.external_model.language_model.lm_head(last_hidden_states).mean(dim=1)
                        # [1,26,32064]--->[1,32064]
                        #未进行softmax
                        batch_embeddings.append(embeddings.squeeze(0))
                        batch_features.append(features.squeeze(0))
                        all_labels.append(labels[i].to(self.device))
                        all_image_ids.append(image_id[i].to(self.device))
                        all_logits.append(logits.to(self.device))
                    else:
                        batch_embeddings.append(torch.zeros(Config.hidden_dim, device=self.device))
                        batch_features.append(torch.zeros(Config.hidden_dim, device=self.device))
                        all_labels.append(torch.tensor(-1, device=self.device))     # 标记无效样本
                        all_image_ids.append(torch.tensor(-1, device=self.device))  # 标记无效样本
                        all_logits.append(torch.zeros((1, Config.vocabulary_size), device=self.device)) #1,vocabulary_size

                if batch_features:  
                    batch_features = torch.stack(batch_features)
                    all_features.append(batch_features)
                    batch_embeddings = torch.stack(batch_embeddings)
                    all_embeddings.append(batch_embeddings)
                else:
                    print(f"[Warning] 客户端 {self.client_id} 批次 {batch_index} 没有有效样本，已跳过")


        # 连接所有批次
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.stack(all_labels)  
        all_image_ids = torch.stack(all_image_ids)
        all_logits = torch.stack(all_logits) #[num, 1, 26, 32064]
        all_embeddings = torch.cat(all_embeddings, dim=0)

        return all_embeddings, all_features, all_labels, all_image_ids, all_logits
    
    # def extract_features_with_external_model(self, dataloader):
    #     """
    #     使用外部LLaVA模型提取特征

    #     Args:
    #         dataloader: 数据加载器

    #     Returns:
    #         模型特征和标签
    #     """
    #     if not self.use_external_model:
    #         raise ValueError("未设置外部模型组件")

    #     all_features = []
    #     all_labels = []

    #     with torch.no_grad():
    #         for batch in tqdm(dataloader, desc=f"客户端 {self.client_id} 使用LLaVA提取特征"):
    #             # 获取批次数据
    #             texts = batch["text"]
    #             images = batch["image"]  # [batch_size, 3, H, W]
    #             labels = batch["label"]

    #             has_image = batch.get("has_image", torch.ones(len(texts), dtype=torch.bool))
    #             has_caption = batch.get("has_caption", torch.ones(len(images), dtype=torch.bool))
    #             batch_features = []
    #             # 逐个处理样本
    #             for i in range(len(texts)):
    #                 text = texts[i]
                    
    #                 if has_image[i] and has_caption[i]:
    #                     # 有文本与图像的情况
    #                     # 将张量转换为PIL图像
    #                     img_tensor = images[i].cpu()
    #                     # 反归一化
    #                     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    #                     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    #                     img_tensor = img_tensor * std + mean
    #                     img_tensor = torch.clamp(img_tensor, 0, 1)
    #                     # 转换为PIL
    #                     import numpy as np
    #                     from PIL import Image as PILImage
    #                     img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    #                     img_pil = PILImage.fromarray(img_np)
    #                     # 使用LLaVA的prompt模板
    #                     prompt = f"USER: <image>\n{text}\nASSISTANT:"
    #                     # 使用LLaVA处理器
    #                     inputs = self.external_processor(
    #                         text=prompt,
    #                         images=img_pil,
    #                         return_tensors="pt"
    #                     )
    #                 elif not has_image[i] and has_caption[i]:
    #                     # 纯文本情况
    #                     prompt = f"USER: {text}\nASSISTANT:"
    #                     inputs = self.external_processor(
    #                         text=prompt,
    #                         return_tensors="pt"
    #                     )
    #                 else:
    #                     #纯图像情况
    #                     # 将张量转换为PIL图像
    #                     img_tensor = images[i].cpu()
    #                     # 反归一化
    #                     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    #                     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    #                     img_tensor = img_tensor * std + mean
    #                     img_tensor = torch.clamp(img_tensor, 0, 1)
    #                     # 转换为PIL
    #                     import numpy as np
    #                     from PIL import Image as PILImage
    #                     img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    #                     img_pil = PILImage.fromarray(img_np)
    #                     # 使用LLaVA的prompt模板
    #                     prompt = f"USER: <image>\nASSISTANT:"
    #                     # 使用LLaVA处理器
    #                     inputs = self.external_processor(
    #                         text=prompt,
    #                         images=img_pil,
    #                         return_tensors="pt"
    #                     )

    #                 # 将inputs移到正确的设备
    #                 inputs = {k: v.to(self.external_model.device) if isinstance(v, torch.Tensor) else v
    #                           for k, v in inputs.items()}  ###字典推导式{k: <>for k,v in 迭代}
                    
    #                 # 使用get_model_input_embeddings获取融合后的输入嵌入
    #                 inputs_embeds = self.external_model.get_model_input_embeddings(
    #                     input_ids=inputs.get('input_ids'),
    #                     pixel_values=inputs.get('pixel_values'),
    #                     attention_mask=inputs.get('attention_mask')
    #                 )
    #                 # 通过模型获取隐藏状态输出
    #                 # 构建新的输入字典，使用inputs_embeds替代input_ids和pixel_values
    #                 forward_inputs = {
    #                     'inputs_embeds': inputs_embeds,
    #                     'attention_mask': inputs.get('attention_mask'),
    #                     'output_hidden_states': True,  # 确保输出所有隐藏状态
    #                     'return_dict': True
    #                 }
    #                 # 前向传播获取隐藏状态
    #                 outputs = self.external_model.language_model(**forward_inputs)
    #                 # 获取最后一层的隐藏状态
    #                 last_hidden_states = outputs.hidden_states[-1]
    #                 features = last_hidden_states[:, -1, :]  # 使用最后一个token

    #                 batch_features.append(features.squeeze(0))

    #             # 堆叠批次特征
    #             batch_features = torch.stack(batch_features)

    #             all_features.append(batch_features)
    #             all_labels.append(labels.to(self.device))


    #     # 连接所有批次
    #     all_features = torch.cat(all_features, dim=0)
    #     all_labels = torch.cat(all_labels, dim=0)


    #     return all_features, all_labels

    def get_captions_dir(self, output_dir = None, aggregation = False):
        if aggregation:
            server_output_dir = output_dir
        else:
            server_output_dir = os.path.join(output_dir,f"{self.client_id}")
            os.makedirs(server_output_dir, exist_ok=True)
        return server_output_dir

    def sanitize_filename(self, text, max_length=100):
        """
        将文本转换为安全的文件名

        Args:
            text: 原始文本
            max_length: 最大长度限制

        Returns:
            安全的文件名
        """
        # 去除不安全的字符
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            text = text.replace(char, '')

        # 限制长度
        if len(text) > max_length:
            text = text[:max_length]

        # 如果文本为空，使用默认名称
        if not text.strip():
            text = "untitled"

        return text.strip()

    def generate_captions(self, dataloader=None, max_length=77, aggregation=False):
        """
        为dataloader生成描述
        """
        output_dir = os.path.join(Config.OUTPUT_DIR, "generated_captions")
        os.makedirs(output_dir, exist_ok=True)
        save_dir = self.get_captions_dir(output_dir, aggregation)

        with tqdm(dataloader, desc=f"为客户端{self.client_id}生成描述") as pbar:
            for batch in pbar:
                # 处理数据并提取特征
                texts = batch["text"]
                images = batch["image"]
                image_ids = batch["image_id"]

                has_caption = batch.get("has_caption", torch.ones(len(texts), dtype=torch.bool))

                batch_text = []
                for i in range(len(texts)):
                    if has_caption[i]:
                        continue
                    prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"

                    img_tensor = images[i].cpu()
                    # 反归一化
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_tensor = img_tensor * std + mean
                    img_tensor = torch.clamp(img_tensor, 0, 1)
                    # 转换为PIL
                    import numpy as np
                    from PIL import Image as PILImage
                    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img_pil = PILImage.fromarray(img_np)
                    inputs = self.external_processor(
                        text = prompt,
                        images = img_pil,
                        return_tensors = "pt"
                    ).to(self.device)                   

                    outputs = self.external_model.generate(
                        **inputs,
                            max_new_tokens=77
                    )
                    captions = self.external_processor.batch_decode(
                        outputs, 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=False
                    )[0]
                    

                    if "ASSISTANT:" in captions:
                        captions = captions.split("ASSISTANT:")[-1].strip()

                    image_id = image_ids[i]
                    filename = f"{int(image_id):012d}.txt"
                    save_path = os.path.join(save_dir,filename)

                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(captions)

