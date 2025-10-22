import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
from config_cub import Config
from tqdm import tqdm
import sys
import os
import numpy as np
from PIL import Image as PILImage
from collections import defaultdict
from loss_function import Loss_function
import ot
import random
import torch.nn.functional as F
import sample_scheduler as ss

class Server:
    """联邦学习服务端（主动方）"""

    def __init__(self, num_classes, train_loader, val_loader, logger=None):
        """
        初始化服务端

        Args:
            num_classes: 分类类别数
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            logger: 日志记录器
        """
        self.num_classes = num_classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.device = Config.DEVICE

        # 初始化LLaVA模型
        self._init_llava_model()

        # 初始化分类头
        self._init_classifier()

        # 初始化一个冻结的llava
        self._init_llava_fronzen()

        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()

        # 为PEFT模型创建优化器
        self.optimizer = optim.AdamW(
            list(self.model.parameters()) + list(self.classifier.parameters()),
            lr=Config.SERVER_LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )

        # 初始化通信统计
        self.communication_stats = {
            "model_download_bytes": 0,
            "feature_upload_bytes": 0,
            "total_rounds": 0
        }

    def upgate_dataloaders(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def _init_llava_fronzen(self):
        """初始化LLaVA模型 (仅推理，不更新参数)"""
        model_id = Config.LLAVA_MODEL_ID
        local_dir = os.path.join(Config.MODEL_CACHE_DIR, "llava-1.5-7b-hf")

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 加载处理器
        if os.path.exists(local_dir) and os.path.isdir(local_dir):
            if self.logger:
                self.logger.info(f"从本地加载LLaVA模型: {local_dir}")
            self.processor_fronzen = AutoProcessor.from_pretrained(local_dir, local_files_only=True)
            self.model_fronzen = LlavaForConditionalGeneration.from_pretrained(
                local_dir,
                local_files_only=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                device_map=None
            )
        else:
            if self.logger:
                self.logger.info(f"从Hugging Face下载LLaVA模型: {model_id}")
            self.processor_fronzen = AutoProcessor.from_pretrained(model_id)
            self.model_fronzen = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                device_map=None
            )

        # 手动将模型移动到设备
        self.model_fronzen = self.model_fronzen.to(self.device)

        # 设置为推理模式
        self.model_fronzen.eval()
        for param in self.model_fronzen.parameters():
            param.requires_grad = False

        if self.logger:
            total_params = sum(p.numel() for p in self.model_fronzen.parameters())
            self.logger.info(f"LLaVA模型已加载 (推理模式)，参数总计: {total_params:,}")
        hidden_size = self.model_fronzen.config.text_config.hidden_size

        self.classifier_fronzen = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, self.num_classes)
        ).to(self.device)
        for param in self.classifier_fronzen.parameters():
            param.requires_grad = False

    def _init_llava_model(self):
        """初始化LLaVA模型并应用LoRA"""
        model_id = Config.LLAVA_MODEL_ID
        local_dir = os.path.join(Config.MODEL_CACHE_DIR, "llava-1.5-7b-hf")

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 加载处理器
        if os.path.exists(local_dir) and os.path.isdir(local_dir):
            if self.logger:
                self.logger.info(f"从本地加载LLaVA模型: {local_dir}")
            self.processor = AutoProcessor.from_pretrained(local_dir, local_files_only=True)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                local_dir,
                local_files_only=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                device_map=None
            )
        else:
            if self.logger:
                self.logger.info(f"从Hugging Face下载LLaVA模型: {model_id}")
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                device_map=None
            )
            

        # 手动将模型移动到设备
        self.model = self.model.to(self.device)

        # 应用LoRA进行参数高效微调
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )

        self.model = get_peft_model(self.model, peft_config)

        if self.logger:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"LLaVA模型参数: {trainable_params:,} 可训练 / {total_params:,} 总计")

    def _init_classifier(self):
        """初始化分类头"""
        # 获取LLaVA的隐藏层维度
        hidden_size = self.model.config.text_config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, self.num_classes)
        ).to(self.device)

    def get_model_components(self):
        """
        获取模型组件供客户端使用

        Returns:
            LLaVA模型和处理器
        """
        # 估算模型大小（字节）
        model = self.model
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        self.communication_stats["model_download_bytes"] += model_size
        return model, self.processor
    
    def get_model_components_fronzen(self):
        """
        获取模型组件供客户端使用

        Returns:
            LLaVA模型和处理器
        """
        # 估算模型大小（字节）
        model = self.model_fronzen
        return model, self.processor_fronzen

    def process_aligned_data_for_features(self, dataloader):
        import numpy as np
        from PIL import Image as PILImage
        """
        处理对齐数据，提取特征

        Args:
            dataloader: 数据加载器

        Returns:
            特征和标签
        """
        all_features = []
        all_labels = []
        all_images_ids = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="服务端处理对齐数据", leave=True):
                texts = batch["text"]
                images = batch["image"]  # [batch_size, 3, H, W]
                labels = batch["label"]
                image_id = batch["image_id"]

                has_image = batch.get("has_image", torch.ones(len(texts), dtype=torch.bool))
                has_caption = batch.get("has_caption", torch.ones(len(images), dtype=torch.bool))
                batch_features = []
                imgs_pil = []
                prompts = []
                for i in range(len(texts)):
                    text = texts[i]

                    if has_image[i] and has_caption[i]:
                        # 处理图像
                        img_tensor = images[i].cpu()
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_tensor = img_tensor * std + mean
                        img_tensor = torch.clamp(img_tensor, 0, 1)

                        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        img_pil = PILImage.fromarray(img_np)

                        prompt = f"USER: <image>\n{text}\nASSISTANT:"
                        imgs_pil.append(img_pil)
                        prompts.append(prompt)
                    elif not has_image[i] and has_caption[i]:
                        # 纯文本情况
                        prompt = f"USER: {text}\nASSISTANT:"
                        imgs_pil.append(None)
                        prompts.append(prompt)
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
                        imgs_pil.append(img_pil)
                        prompts.append(prompt)
                        # 使用LLaVA处理器


                    # 移到设备
                inputs = self.processor(
                                        text=prompts,
                                        images=imgs_pil,
                                        return_tensors="pt",
                                        padding=True
                                    )
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in inputs.items()}

                    # 使用get_model_input_embeddings获取融合后的输入嵌入
                inputs_embeds = self.model.get_model_input_embeddings(
                    input_ids=inputs.get('input_ids'),
                    pixel_values=inputs.get('pixel_values'),
                    attention_mask=inputs.get('attention_mask')
                )

                    # 通过模型获取隐藏状态输出
                forward_inputs = {
                    'inputs_embeds': inputs_embeds,
                    'attention_mask': inputs.get('attention_mask'),
                    'output_hidden_states': True,
                    'return_dict': True
                }

                    # 前向传播获取隐藏状态
                outputs = self.model.language_model(**forward_inputs)

                # 获取最后一层的隐藏状态
                last_hidden_states = outputs.hidden_states[-1]
                batch_features = last_hidden_states[:, -1, :]  # 使用最后一个token


                all_features.append(batch_features)
                all_labels.append(labels.to(self.device))
                all_images_ids.append(image_id.to(self.device))

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_images_ids = torch.cat(all_images_ids, dim=0)

        return all_features, all_labels, all_images_ids

    # def process_aligned_data_for_features(self, dataloader):
    #     import numpy as np
    #     from PIL import Image as PILImage
    #     """
    #     处理对齐数据，提取特征

    #     Args:
    #         dataloader: 数据加载器

    #     Returns:
    #         特征和标签
    #     """
    #     all_features = []
    #     all_labels = []
    #     all_images_ids = []

    #     with torch.no_grad():
    #         for batch in tqdm(dataloader, desc="服务端处理对齐数据", leave=True):
    #             texts = batch["text"]
    #             images = batch["image"]  # [batch_size, 3, H, W]
    #             labels = batch["label"]
    #             image_id = batch["image_id"]

    #             has_image = batch.get("has_image", torch.ones(len(texts), dtype=torch.bool))
    #             has_caption = batch.get("has_caption", torch.ones(len(images), dtype=torch.bool))
    #             batch_features = []

    #             for i in range(len(texts)):
    #                 text = texts[i]

    #                 if has_image[i] and has_caption[i]:
    #                     # 处理图像
    #                     img_tensor = images[i].cpu()
    #                     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    #                     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    #                     img_tensor = img_tensor * std + mean
    #                     img_tensor = torch.clamp(img_tensor, 0, 1)

    #                     img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    #                     img_pil = PILImage.fromarray(img_np)

    #                     prompt = f"USER: <image>\n{text}\nASSISTANT:"

    #                     inputs = self.processor(
    #                         text=prompt,
    #                         images=img_pil,
    #                         return_tensors="pt"
    #                     )
    #                 elif not has_image[i] and has_caption[i]:
    #                     # 纯文本情况
    #                     prompt = f"USER: {text}\nASSISTANT:"
    #                     inputs = self.processor(
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
    #                     inputs = self.processor(
    #                         text=prompt,
    #                         images=img_pil,
    #                         return_tensors="pt"
    #                     )

    #                 # 移到设备
    #                 inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
    #                           for k, v in inputs.items()}

    #                 # 使用get_model_input_embeddings获取融合后的输入嵌入
    #                 inputs_embeds = self.model.get_model_input_embeddings(
    #                     input_ids=inputs.get('input_ids'),
    #                     pixel_values=inputs.get('pixel_values'),
    #                     attention_mask=inputs.get('attention_mask')
    #                 )

    #                 # 通过模型获取隐藏状态输出
    #                 forward_inputs = {
    #                     'inputs_embeds': inputs_embeds,
    #                     'attention_mask': inputs.get('attention_mask'),
    #                     'output_hidden_states': True,
    #                     'return_dict': True
    #                 }

    #                 # 前向传播获取隐藏状态
    #                 outputs = self.model.language_model(**forward_inputs)

    #                 # 获取最后一层的隐藏状态
    #                 last_hidden_states = outputs.hidden_states[-1]
    #                 features = last_hidden_states[:, -1, :]  # 使用最后一个token

    #                 batch_features.append(features.squeeze(0))

    #             batch_features = torch.stack(batch_features)

    #             all_features.append(batch_features)
    #             all_labels.append(labels.to(self.device))
    #             all_images_ids.append(image_id.to(self.device))

    #     all_features = torch.cat(all_features, dim=0)
    #     all_labels = torch.cat(all_labels, dim=0)
    #     all_images_ids = torch.cat(all_images_ids, dim=0)

    #     return all_features, all_labels, all_images_ids

    def fuse_features(self, server_features, client_features_list):
        """
        融合来自服务端和客户端的特征

        Args:
            server_features: 服务端特征
            client_features_list: 客户端特征列表

        Returns:
            融合后的特征
        """
        # 确保所有特征在同一设备上
        server_features = server_features.to(self.device)
        client_features_list = [f.to(self.device) for f in client_features_list]

        # 估算特征上传大小（字节）
        for features in client_features_list:
            feature_size = features.numel() * features.element_size()
            self.communication_stats["feature_upload_bytes"] += feature_size

        # 简单平均融合策略
        all_features = [server_features] + client_features_list
        fused_features = torch.stack(all_features).mean(dim=0)

        return fused_features

    def train_with_fused_features(self, fused_features, labels, epochs=Config.SERVER_EPOCHS, server_acc = None):
        """
        使用融合特征训练分类器

        Args:
            fused_features: 融合后的特征
            labels: 标签
        """
        self.model.train()
        self.classifier.train()

        # 确保数据在正确的设备上并转换为正确的数据类型
        fused_features = fused_features.to(self.device).float()  # 显式转换为 float32
        labels = labels.to(self.device)
        for epoch in range(epochs):
            # 前向传播
            logits = self.classifier(fused_features)
            loss = self.criterion(logits, labels)
 
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 计算准确率
            _, predicted = torch.max(logits, 1)
            accuracy = (predicted == labels).float().mean().item() * 100.0
            if server_acc:
                server_acc["train_acc"]["aligned"].append(round(accuracy,2))

            if self.logger:
                self.logger.info(
                    f"服务端融合特征训练 Epoch [{epoch + 1}/{epochs}] - Loss: {loss.item():.4f}, Acc: {accuracy:.2f}%"
                )

        return loss.item(), accuracy

    def train_with_local_data(self, epochs=Config.SERVER_EPOCHS, server_acc = None):
        """
        使用本地非对齐数据训练

        Args:
            epochs: 训练轮数
        """
        self.model.train()
        self.classifier.train()

        for epoch in range(epochs):
            total_loss = 0.0
            total_acc = 0.0
            num_batches = 0

            with tqdm(self.train_loader, desc=f"服务端本地训练 Epoch {epoch + 1}/{epochs}") as pbar:
                for batch in pbar:
                    # 处理数据并提取特征
                    texts = batch["text"]
                    images = batch["image"]
                    labels = batch["label"].to(self.device)
                    has_caption = batch.get("has_caption", torch.ones(len(texts), dtype=torch.bool))
                    #idxs = batch["idx"]

                    batch_features = []

#                     for i in range(len(texts)):
#                         img_pil = images[i]
#                         img_tensor = images[i].cpu()
#                         # 反归一化
#                         mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#                         std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#                         img_tensor = img_tensor * std + mean
#                         img_tensor = torch.clamp(img_tensor, 0, 1)
#                         # 转换为PIL
#                         import numpy as np
#                         from PIL import Image as PILImage
#                         img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
#                         img_pil = PILImage.fromarray(img_np)
#                         if has_caption[i]:
#                             text = texts[i]

#                             prompt = f"USER: <image>\n{text}\nASSISTANT:"

#                             inputs = self.processor(
#                                 text=prompt,
#                                 images=img_pil,
#                                 return_tensors="pt"
#                             )
#                         else:
#                             prompt = f"USER: <image>\nASSISTANT:"
#                             inputs = self.processor(
#                                 text=prompt,
#                                 images=img_pil,
#                                 return_tensors="pt"
#                             )

#                         # 移到设备
#                         inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
#                                   for k, v in inputs.items()}

#                         # 通过LLaVA前向传播获取隐藏状态
#                         with torch.no_grad():
#                             # 使用get_model_input_embeddings获取融合后的输入嵌入
#                             inputs_embeds = self.model.get_model_input_embeddings(
#                                 input_ids=inputs.get('input_ids'),
#                                 pixel_values=inputs.get('pixel_values'),
#                                 attention_mask=inputs.get('attention_mask')
#                             )

#                             # 通过模型获取隐藏状态输出
#                             forward_inputs = {
#                                 'inputs_embeds': inputs_embeds,
#                                 'attention_mask': inputs.get('attention_mask'),
#                                 'output_hidden_states': True,
#                                 'return_dict': True
#                             }

#                             # 前向传播获取隐藏状态
#                             outputs = self.model.language_model(**forward_inputs)

#                             # 获取最后一层的隐藏状态
#                             last_hidden_states = outputs.hidden_states[-1]

#                             # 使用最后一个token的隐藏状态作为特征
#                             features = last_hidden_states[:, -1, :]

#                             batch_features.append(features.squeeze(0))
# #1315 loss传出来 label    
#                     batch_features = torch.stack(batch_features)

#                     # 确保数据类型一致 - 转换为float32
#                     batch_features = batch_features.float()
# #last_hidden 

                    prompts = []
                    imgs_pil = []
                    for i in range(len(texts)):
                        img_pil = images[i]
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
                        imgs_pil.append(img_pil)
                        if has_caption[i]:
                            prompts.append(f"USER: <image>\n{texts[i]}\nASSISTANT:")
                        else:
                            prompts.append(f"USER: <image>\nASSISTANT:")
                    inputs = self.processor(
                        text=prompts,
                        images=imgs_pil,
                        return_tensors="pt",
                        padding=True
                    )
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                for k, v in inputs.items()}
                    with torch.no_grad():
                        inputs_embeds = self.model.get_model_input_embeddings(
                            input_ids=inputs.get('input_ids'),
                            pixel_values=inputs.get('pixel_values'),
                            attention_mask=inputs.get('attention_mask')
                        )

                        forward_inputs = {
                            'inputs_embeds': inputs_embeds,
                            'attention_mask': inputs.get('attention_mask'),
                            'output_hidden_states': True,
                            'return_dict': True
                        }

                        outputs = self.model.language_model(**forward_inputs)

                        last_hidden_states = outputs.hidden_states[-1]   # [batch_size, seq_len, hidden_dim]
                        batch_features = last_hidden_states[:, -1, :]          # 取最后一个token作为特征
                    batch_features = batch_features.float()
                    # 分类
                    logits = self.classifier(batch_features)
                    loss = self.criterion(logits, labels)

                    # 反向传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # 计算准确率
                    _, predicted = torch.max(logits, 1)
                    accuracy = (predicted == labels).float().mean().item() * 100.0

                    # if accuracy < Config.use_caption_accuracy * 100:
                    #     is_generated_caption = batch.get("is_generated_caption", torch.ones(len(texts), dtype=torch.bool))
                    #     for i in range(len(texts)):
                    #         if is_generated_caption[i]:
                    #             idx = idxs[i].item() if torch.is_tensor(idxs[i]) else idxs[i]
                    #             self.train_loader.dataset.data_list[idx]["has_caption"] = False
                    #             self.train_loader.dataset.data_list[idx]["is_generated_caption"] = False

                    #             batch["has_caption"][i] = False
                    #             batch["is_generated_caption"][i] = False


                    total_loss += loss.item()
                    total_acc += accuracy
                    num_batches += 1

                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{accuracy:.2f}%"})
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            avg_acc = total_acc / num_batches if num_batches > 0 else 0
            if server_acc:
                server_acc["train_acc"]["non_aligned"].append(round(avg_acc,2))

            if epoch % Config.SERVER_VALIDATION_INTERVAL == 0:   
                val_loss, val_acc = self.evaluate()
                if server_acc:
                    server_acc["val_acc"].append(round(val_acc,2))
                self.logger.info(f"服务端第{epoch + 1}轮本地训练完成 - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.2f}%, Val loss:{val_loss:.4f}, Val_acc:{val_acc:2f}")
            else:
                self.logger.info(f"服务端第{epoch + 1}轮本地训练完成 - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.2f}%")
    
    def train_with_local_data_new(self, epochs=Config.SERVER_EPOCHS, server_acc = None, model_preferences = None, final_embeddings = None, round = 0):
        """
        使用本地非对齐数据训练

        Args:
            epochs: 训练轮数
        """
        self.model.train()
        self.classifier.train()
        i = 0
        model_preferences_new = [x for row in model_preferences for x in row]
        for epoch in range(epochs):
            total_loss = 0.0
            total_acc = 0.0
            num_batches = 0
            epoch_final_embeddings = final_embeddings[epoch*Config.NON_ALIGNED_DATA_SIZE_CLIENT:(epoch+1)*Config.NON_ALIGNED_DATA_SIZE_CLIENT]
            epoch_model_preferences = model_preferences_new[epoch*Config.NON_ALIGNED_DATA_SIZE_CLIENT:(epoch+1)*Config.NON_ALIGNED_DATA_SIZE_CLIENT]
            with tqdm(self.train_loader, desc=f"服务端本地训练 Epoch {epoch + 1}/{epochs}") as pbar:
                for batch_idx, batch in enumerate(pbar):
                    # 处理数据并提取特征

                    start = batch_idx * Config.BATCH_SIZE

                    texts = batch["text"]
                    images = batch["image"]
                    labels = batch["label"].to(self.device)
                    has_caption = batch.get("has_caption", torch.ones(len(texts), dtype=torch.bool))
                    idxs = batch["idx"]

                    end = start + len(images)
                    batch_embeddings = epoch_final_embeddings[start:end]
                    batch_model_preferences = epoch_model_preferences[start:end]

                    batch_features = []

#                     for i in range(len(texts)):
#                         img_pil = images[i]
#                         img_tensor = images[i].cpu()
#                         # 反归一化
#                         mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#                         std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#                         img_tensor = img_tensor * std + mean
#                         img_tensor = torch.clamp(img_tensor, 0, 1)
#                         # 转换为PIL
#                         import numpy as np
#                         from PIL import Image as PILImage
#                         img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
#                         img_pil = PILImage.fromarray(img_np)
#                         if has_caption[i]:
#                             text = texts[i]

#                             prompt = f"USER: <image>\n{text}\nASSISTANT:"

#                             inputs = self.processor(
#                                 text=prompt,
#                                 images=img_pil,
#                                 return_tensors="pt"
#                             )
#                         else:
#                             prompt = f"USER: <image>\nASSISTANT:"
#                             inputs = self.processor(
#                                 text=prompt,
#                                 images=img_pil,
#                                 return_tensors="pt"
#                             )

#                         # 移到设备
#                         inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
#                                   for k, v in inputs.items()}

#                         # 通过LLaVA前向传播获取隐藏状态
#                         with torch.no_grad():
#                             # 使用get_model_input_embeddings获取融合后的输入嵌入
#                             inputs_embeds = self.model.get_model_input_embeddings(
#                                 input_ids=inputs.get('input_ids'),
#                                 pixel_values=inputs.get('pixel_values'),
#                                 attention_mask=inputs.get('attention_mask')
#                             )

#                             # 通过模型获取隐藏状态输出
#                             forward_inputs = {
#                                 'inputs_embeds': inputs_embeds,
#                                 'attention_mask': inputs.get('attention_mask'),
#                                 'output_hidden_states': True,
#                                 'return_dict': True
#                             }

#                             # 前向传播获取隐藏状态
#                             outputs = self.model.language_model(**forward_inputs)

#                             # 获取最后一层的隐藏状态
#                             last_hidden_states = outputs.hidden_states[-1]

#                             # 使用最后一个token的隐藏状态作为特征
#                             features = last_hidden_states[:, -1, :]

#                             batch_features.append(features.squeeze(0))
# #1315 loss传出来 label    
#                     batch_features = torch.stack(batch_features)

#                     # 确保数据类型一致 - 转换为float32
#                     batch_features = batch_features.float()
# #last_hidden 
                    prompts = []
                    imgs_pil = []
                    for i in range(len(texts)):
                        img_pil = images[i]
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
                        imgs_pil.append(img_pil)
                        if has_caption[i]:
                            prompts.append(f"USER: <image>\n{texts[i]}\nASSISTANT:")
                        else:
                            prompts.append(f"USER: <image>\nASSISTANT:")
                    inputs = self.processor(
                        text=prompts,
                        images=imgs_pil,
                        return_tensors="pt",
                        padding=True
                    )
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                for k, v in inputs.items()}
                    with torch.no_grad():
                        inputs_embeds = self.model.get_model_input_embeddings(
                            input_ids=inputs.get('input_ids'),
                            pixel_values=inputs.get('pixel_values'),
                            attention_mask=inputs.get('attention_mask')
                        )

                        forward_inputs = {
                            'inputs_embeds': inputs_embeds,
                            'attention_mask': inputs.get('attention_mask'),
                            'output_hidden_states': True,
                            'return_dict': True
                        }

                        outputs = self.model.language_model(**forward_inputs)

                        last_hidden_states = outputs.hidden_states[-1]   # [batch_size, seq_len, hidden_dim]
                        batch_features = last_hidden_states[:, -1, :]          # 取最后一个token作为特征
                    batch_features = batch_features.float()
                    # 分类
                    logits = self.classifier(batch_features)
                    loss = self.criterion(logits, labels)

                    # 反向传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    embeddings = []
                    for i in range(len(batch_embeddings)):
                        embeddings.append(batch_embeddings[i].detach())
                    #final_embedding对应的应该是batch里面每个sample的值 也就是维度应该是5*32*64*4096
                    embeddings_ = []
                    for i in range(len(batch_model_preferences)):
                        embeddings_.append(batch_model_preferences[i].detach())
                    

                    train_data = ss.Dataset_net_p_q(embeddings, embeddings_)
                    train_loader = ss.DataLoader(train_data, batch_size=32, shuffle=True)
                    net_p = ss.Residual_Network_exploitation(dim=embeddings[0].shape[-1]).to(self.device)
                    net_q = ss.Residual_Network_exploration().to(self.device)
                    if round > 1:
                        net_p.load_state_dict(torch.load(Config.ss_model['net_p_weights_path']))
                        net_q.load_state_dict(torch.load(Config.ss_model['net_q_weights_path']))
                    hidden_states = ss.train_net_p(net_p, train_loader, Config.net_p_q_epoch, Config.net_p_q_lr).detach()
                    train_data = ss.Dataset_net_p_q(hidden_states, embeddings_)
                    train_loader = ss.DataLoader(train_data, batch_size=32, shuffle=True)
                    _ = ss.train_net_q(net_q, train_loader, Config.net_p_q_epoch, Config.net_p_q_lr)
                    torch.save(net_p.state_dict(), Config.ss_model['net_p_weights_path'])
                    torch.save(net_q.state_dict(), Config.ss_model['net_q_weights_path'])

                    # 计算准确率
                    _, predicted = torch.max(logits, 1)
                    accuracy = (predicted == labels).float().mean().item() * 100.0
                    scores = []
                    for i in range(len(embeddings)):
                        score = net_q(net_p(embeddings[i])[1]).squeeze()
                        scores.append(score)
                    scores = torch.stack(scores)
                    scores = (scores-torch.min(scores))/(torch.max(scores)-torch.min(scores))
                    scores = torch.mean(scores)

                    if scores*100 < Config.use_caption_accuracy * 100:
                        is_generated_caption = batch.get("is_generated_caption", torch.ones(len(texts), dtype=torch.bool))
                        for i in range(len(texts)):
                            if is_generated_caption[i]:
                                idx = idxs[i].item() if torch.is_tensor(idxs[i]) else idxs[i]
                                self.train_loader.dataset.data_list[idx]["has_caption"] = False
                                self.train_loader.dataset.data_list[idx]["is_generated_caption"] = False

                                batch["has_caption"][i] = False
                                batch["is_generated_caption"][i] = False


                    total_loss += loss.item()
                    total_acc += accuracy
                    num_batches += 1

                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{accuracy:.2f}%"})
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            avg_acc = total_acc / num_batches if num_batches > 0 else 0
            if server_acc:
                server_acc["train_acc"]["non_aligned"].append(round(avg_acc,2))

            if epoch % Config.SERVER_VALIDATION_INTERVAL == 0:   
                val_loss, val_acc = self.evaluate()
                if server_acc:
                    server_acc["val_acc"].append(round(val_acc,2))
                self.logger.info(f"服务端第{epoch + 1}轮本地训练完成 - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.2f}%, Val loss:{val_loss:.4f}, Val_acc:{val_acc:2f}")
            else:
                self.logger.info(f"服务端第{epoch + 1}轮本地训练完成 - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.2f}%")

    def evaluate(self):
        """评估模型性能"""
        self.model.eval()
        self.classifier.eval()

        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # 处理数据
                texts = batch["text"]
                images = batch["image"]
                labels = batch["label"].to(self.device)
                has_image = batch.get("has_image", torch.ones(len(texts), dtype=torch.bool))
                has_caption = batch.get("has_image", torch.ones(len(texts), dtype=torch.bool))
                batch_features = []

                prompts = []
                imgs_pil = []
                for i in range(len(texts)):
                    img_pil = images[i]
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
                    imgs_pil.append(img_pil)
                    if has_caption[i]:
                        prompts.append(f"USER: <image>\n{texts[i]}\nASSISTANT:")
                    else:
                        prompts.append(f"USER: <image>\nASSISTANT:")
                inputs = self.processor(
                    text=prompts,
                    images=imgs_pil,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in inputs.items()}
                with torch.no_grad():
                    inputs_embeds = self.model.get_model_input_embeddings(
                        input_ids=inputs.get('input_ids'),
                        pixel_values=inputs.get('pixel_values'),
                        attention_mask=inputs.get('attention_mask')
                    )

                    forward_inputs = {
                        'inputs_embeds': inputs_embeds,
                        'attention_mask': inputs.get('attention_mask'),
                        'output_hidden_states': True,
                        'return_dict': True
                    }

                    outputs = self.model.language_model(**forward_inputs)

                    last_hidden_states = outputs.hidden_states[-1]   # [batch_size, seq_len, hidden_dim]
                    batch_features = last_hidden_states[:, -1, :]          # 取最后一个token作为特征
                batch_features = batch_features.float()

                # 分类
                logits = self.classifier(batch_features)
                loss = self.criterion(logits, labels)

                # 计算准确率
                _, predicted = torch.max(logits, 1)
                accuracy = (predicted == labels).float().mean().item() * 100.0

                total_loss += loss.item()
                total_acc += accuracy
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_acc = total_acc / num_batches if num_batches > 0 else 0

        return avg_loss, avg_acc

    def log_communication_stats(self):
        """记录通信统计信息"""
        self.communication_stats["total_rounds"] = Config.NUM_ROUNDS
        
        # 转换为MB
        model_down_mb = self.communication_stats["model_download_bytes"] / (1024 * 1024)
        feature_up_mb = self.communication_stats["feature_upload_bytes"] / (1024 * 1024)
        
        self.logger.info(f"通信统计 (总轮次 {self.communication_stats['total_rounds']}):")
        self.logger.info(f"  - 模型下发总量: {model_down_mb:.2f} MB")
        self.logger.info(f"  - 特征上传总量: {feature_up_mb:.2f} MB")
        self.logger.info(f"  - 总通信量: {model_down_mb + feature_up_mb:.2f} MB")
        self.logger.info(f"  - 平均每轮通信量: {(model_down_mb + feature_up_mb) / self.communication_stats['total_rounds']:.2f} MB/轮")
    
    def set_class_prototype(self, server_class_prototype):
        self.server_class_prototype = server_class_prototype

    def get_prototype_from_class_prototype(self, class_prototypes):
        class_prototype = defaultdict(list)
        for client_id in Config.NUM_CLIENTS[-1]:
            for label, feature in class_prototype[client_id].items():
                class_prototype[label].append[feature]
        prototypes = defaultdict(list)
        for label, features in class_prototype.items():
            class_prototype[label].append[self.server_class_prototype[label]]
            prototypes[label] = torch.stack(class_prototype[label],dim=0).mean(dim=0)
        return prototypes

    def compute_ot_distance(self, source, target):
        """
        source: Tensor[N, D] - 源数据 (1000, 4096)
        target: Tensor[M, D] - 目标原型分布 (100, 4096)
        return: ot_distances[N, 1] - OT距离矩阵 (1000, 1)
        """
        X = source.cpu().numpy()  # [1000, 4096]
        Y = target.cpu().numpy()  # [100, 4096]
        
        N, D = X.shape
        M, _ = Y.shape
        
        # 目标分布的权重（均匀分布）
        target_weights = np.ones(M) / M
        
        # 预计算所有源点到所有目标点的距离
        all_distances = ot.dist(X, Y, metric='euclidean')  # [N, M]
        
        # 标准化
        if all_distances.max() > 0:
            all_distances = all_distances / all_distances.max()
        
        # 计算每个源点的OT距离
        ot_distances = np.zeros(N)
        source_weights = np.ones(1)  # 单点权重
        
        for i in range(N):
            # 使用预计算的距离矩阵中的第i行作为成本矩阵
            cost_matrix = all_distances[i:i+1]  # [1, M] - 保持2D形状
            
            # 计算OT距离（使用EMD - Earth Mover's Distance）
            ot_distance = ot.emd2(source_weights, target_weights, cost_matrix)
            ot_distances[i] = ot_distance
        
        return ot_distances.reshape(-1, 1)
    
    def extract_captions(self, dataloader):
        """
        使用外部LLaVA模型提取非对齐数据 文本特征

        Args:
            dataloader: 数据加载器

        Returns:
            模型特征和标签
        """

        all_features = []
        all_labels = []
        all_image_ids = []
        all_logits = []
        all_embeddings = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"服务端使用LLaVA提取特征"):
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
                        inputs = self.processor(
                            text=prompt,
                            return_tensors="pt"
                        )

                        # 将inputs移到正确的设备
                        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                for k, v in inputs.items()}  ###字典推导式{k: <>for k,v in 迭代}
                        
                        # 使用get_model_input_embeddings获取融合后的输入嵌入
                        inputs_embeds = self.model.get_model_input_embeddings(
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
                        outputs = self.model.language_model(**forward_inputs)
                        # 获取最后一层的隐藏状态
                        last_hidden_states = outputs.hidden_states[-1]
                        embeddings = last_hidden_states.mean(dim = 1)
                        features = last_hidden_states[:, -1, :]  # 使用最后一个token
                        logits = self.model.language_model.lm_head(features)
                        # [1,26,32064]--->[1,32064]
                        #未进行softmax

                        batch_features.append(features.squeeze(0))
                        batch_embeddings.append(embeddings.squeeze(0))
                        all_labels.append(labels[i].to(self.device))
                        all_image_ids.append(image_id[i].to(self.device))
                        all_logits.append(logits.to(self.device))
                    else:
                        batch_embeddings.append(torch.zeros(Config.hidden_dim, device=self.device))
                        batch_features.append(torch.zeros(Config.hidden_dim, device=self.device))
                        all_labels.append(torch.tensor(-1, device=self.device))     # 标记无效样本
                        all_image_ids.append(torch.tensor(-1, device=self.device))  # 标记无效样本
                        all_logits.append(torch.zeros((1, Config.vocabulary_size), device=self.device)) #1,vocabulary_size

                # 堆叠批次特征
                if batch_features:
                    batch_features = torch.stack(batch_features)
                    all_features.append(batch_features)
                    batch_embeddings = torch.stack(batch_embeddings)
                    all_embeddings.append(batch_embeddings)


        # 连接所有批次
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.stack(all_labels)  
        all_image_ids = torch.stack(all_image_ids)
        all_logits = torch.stack(all_logits) #[num, 1, 26, 32064]
        all_embeddings = torch.cat(all_embeddings, dim=0)

        return all_embeddings, all_features, all_labels, all_image_ids, all_logits
        
    def train_for_caption_generator(self, epochs = Config.IMAGE_GENERATOR_EPOCHS, 
                                 beta = 1.0, prototypes=None, sim_matrix=None,
                                 dataloader=None, text_prototypes=None,
                                 text_logits=None, text_logits_fronzen=None):
        """
        训练图像生成器
        z_i对应现有模态, o_i对应超原型信息
        g_q对应生成补全模态, g_p对应超原型相应模态
        """
        self.model.train()
        self.classifier.train()

        # 冻结视觉编码器
        for param in self.model.model.vision_tower.parameters():
            param.requires_grad = False

        # 冻结投影层
        for param in self.model.model.multi_modal_projector.parameters():
            param.requires_grad = False

        for epoch in range(epochs):
            total_loss = 0.0
            total_acc = 0.0
            num_batches = 0

            with tqdm(dataloader, desc=f"服务端图生文训练 Epoch {epoch + 1}/{epochs}") as pbar:
                for batch in pbar:
                    # 处理数据并提取特征
                    texts = batch["text"]
                    images = batch["image"]
                    labels = batch["label"]

                    has_caption = batch.get("has_caption", torch.ones(len(texts), dtype=torch.bool))

                    for i in range(len(texts)):
                        if has_caption[i]:
                            continue
                        # Step1: 得到g_q
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

                        inputs = self.processor(
                            text = prompt,
                            images = img_pil,
                            return_tensors = "pt"
                        ).to(self.device)   

# 目标原型作为正例，其他为负例 dpo损失用不了 用对比损失(形式参考)  dpo创新度

                        # outputs = self.model.generate(
                        #     **inputs,
                        #      max_new_tokens=77
                        # )
                        # captions = self.processor.batch_decode(
                        #                         outputs, 
                        #                         skip_special_tokens=True, 
                        #                         clean_up_tokenization_spaces=False
                        #                     )[0]                

                        outputs = self.model(
                                **inputs,
                                output_hidden_states=True,
                                return_dict=True
                            )
                        
                        last_hidden = outputs.hidden_states[-1]
                        non_preferred_embedding = last_hidden[:, -1, :] #取最后一个token
                        non_preferred_logits = self.model.language_model.lm_head(last_hidden).mean(dim=1)[0]

                        ### Step3: 得到ref_non_preferred_logits



                        inputs = self.processor_fronzen(
                            text = prompt,
                            images = img_pil,
                            return_tensors = "pt"
                        ).to(self.device)                   

                        outputs = self.model_fronzen(
                                **inputs,
                                output_hidden_states=True,
                                return_dict=True
                            )
                        
                        last_hidden = outputs.hidden_states[-1]
                        ref_non_preferred_logits = self.model_fronzen.language_model.lm_head(last_hidden).mean(dim=1)[0]

                        ### Steo3:: 得到g_p和o_i
                        num_i = len(images)*epoch + i #第几个样本
                        row = sim_matrix[Config.NUM_CLIENTS][num_i]
                        j = row.argmin()    # 最相似的超原型
                        sim = row[j]        # 相似度

                        preferred_logits = text_logits[j]
                        ref_logits = text_logits_fronzen[j]

                        o_i_embedding = text_prototypes[j]

                        ### Step4: 计算loss并返回训练

                        loss = self.loss_for_image(
                            g_q_logits=non_preferred_logits,
                            g_p_logits=preferred_logits,
                            g_q_embedding=non_preferred_embedding,
                            o_i_embedding=o_i_embedding,
                            sim=sim,
                            beta=beta,
                            ref_logits_preferred=ref_logits, 
                            ref_logits_non_preferred=ref_non_preferred_logits
                        )

                        # Step5: 反向传播
                        self.optimizer.zero_grad()  # 清空梯度
                        loss.backward()             # 梯度回传
                        self.optimizer.step()       # 更新模型参数
                if self.logger:
                    self.logger.info(
                        f"服务端图生文训练 - Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
                    
    def loss_for_image(self, 
                        g_q_logits=None,g_p_logits=None,
                        g_q_embedding=None,o_i_embedding=None,
                        sim=None,beta=None,
                        ref_logits_preferred=None, ref_logits_non_preferred=None):
        """
        z_i 第i个样本的embedding
        o_i 第i个样本对应模态的超原型也为embedding
        """
        loss_fn = Loss_function()
        loss_prefer, reward_diff = loss_fn.loss_prefer(
                g_p_logits=g_p_logits, g_q_logits=g_q_logits, 
                ref_logits_preferred=ref_logits_preferred, ref_logits_non_preferred=ref_logits_non_preferred,
                beta=1.0
            )
        loss_special = loss_fn.loss_special(g_q_embedding=g_q_embedding, o_i_embedding=o_i_embedding, sim_score=sim)
        total_loss = beta*loss_prefer+loss_special
        return total_loss, reward_diff
    
    def get_captions_dir(self, output_dir = None, aggregation = False):
        if aggregation:
            server_output_dir = output_dir
        else:
            server_output_dir = os.path.join(output_dir,"server")
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
        device = self.device
        output_dir = os.path.join(Config.OUTPUT_DIR, "generated_captions")
        os.makedirs(output_dir, exist_ok=True)
        save_dir = self.get_captions_dir(output_dir, aggregation)

        with tqdm(dataloader, desc=f"为服务端生成描述") as pbar:
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

                    prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"

                    inputs = self.processor(
                        text = prompt,
                        images = img_pil,
                        return_tensors = "pt"
                    ).to(self.device)                   

                    outputs = self.model.generate(
                            **inputs,
                             max_new_tokens=max_length
                        )
                    captions = self.processor.batch_decode(
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

    def train_for_caption_generator_with_infoNCE_loss(self, epochs = Config.IMAGE_GENERATOR_EPOCHS, 
                                 beta = 1.0, prototypes=None, sim_matrix=None,
                                 dataloader=None, text_prototypes=None,
                                 text_logits=None, text_logits_fronzen=None):
        """
        训练图像生成器
        z_i对应现有模态, o_i对应超原型信息
        g_q对应生成补全模态, g_p对应超原型相应模态
        """
        self.model.train()
        self.classifier.train()

        # 冻结视觉编码器
        for param in self.model.model.vision_tower.parameters():
            param.requires_grad = False

        # 冻结投影层
        for param in self.model.model.multi_modal_projector.parameters():
            param.requires_grad = False

        loss_history = []
        loss_0_history = []
        loss_1_history = []
        for epoch in range(epochs):
            total_loss = 0.0
            total_acc = 0.0
            num_batches = 0

#先生成描述 然后再训练    1150开始训练的是生成器        

            with tqdm(dataloader, desc=f"服务端图生文训练 Epoch {epoch + 1}/{epochs}") as pbar:
                for batch_idx, batch in enumerate(pbar, start=1):  

                    # 处理数据并提取特征
                    texts = batch["text"]
                    images = batch["image"]
                    labels = batch["label"]

                    has_caption = batch.get("has_caption", torch.ones(len(texts), dtype=torch.bool))

                    for i in range(len(texts)):
                        if not has_caption[i]:
                            continue
                        # Step1: 得到x
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

                        inputs = self.processor(
                            text = prompt,
                            images = img_pil,
                            return_tensors = "pt"
                        ).to(self.device)   

# 目标原型作为正例，其他为负例 dpo损失用不了 用对比损失(形式参考)  dpo创新度

                        # outputs = self.model.generate(
                        #     **inputs,
                        #      max_new_tokens=77
                        # )
                        # captions = self.processor.batch_decode(
                        #                         outputs, 
                        #                         skip_special_tokens=True, 
                        #                         clean_up_tokenization_spaces=False
                        #                     )[0]                

                        outputs = self.model(
                                **inputs,
                                output_hidden_states=True,
                                return_dict=True
                            )
                        last_hidden = outputs.hidden_states[-1]
                        last_token = last_hidden[:, -1, :]
                        avg_hidden = last_hidden.mean(dim=1) 
                        x = self.model.language_model.lm_head(last_token)

                        ### Step2: 得到y
                        num = int(labels[i])
                        if num >= len(text_logits) or torch.all(text_logits[num] == 0):
                            print(f"Label {num} 对应的text_logits缺失或空向量，跳过")
                            continue   
                        y = text_logits[num]

                        ### Step3: 得到z
                        total_num = len(text_logits)
                        all_indices = list(range(total_num))
                        all_indices.remove(num)  

                        random.seed(42)  

                        k = 5  
                        z_indices = random.sample(all_indices, k)  
                        z = [text_logits[idx] for idx in z_indices]

                        ### Step4: 得到描述对应的x_caption
                        text = texts[i]

                        prompt = f"USER: {text}\nASSISTANT:"

                        inputs = self.processor(
                            text = prompt,
                            return_tensors = "pt"
                        ).to(self.device) 

                        outputs = self.model(
                                **inputs,
                                output_hidden_states=True,
                                return_dict=True
                            )
                        last_hidden = outputs.hidden_states[-1]
                        last_token = last_hidden[:, -1, :]
                        x_text = self.model.language_model.lm_head(last_token)

                        ### Step5: 得到ref_non_preferred_logits
                        prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"

                        inputs = self.processor_fronzen(
                            text = prompt,
                            images = img_pil,
                            return_tensors = "pt"
                        ).to(self.device)                   

                        outputs = self.model_fronzen(
                                **inputs,
                                output_hidden_states=True,
                                return_dict=True
                            )

                        last_hidden = outputs.hidden_states[-1]
                        ref_non_preferred_logits = self.model_fronzen.language_model.lm_head(last_hidden).mean(dim=1)[0]

                        ### Steo6: 得到g_p和o_i
                        num_i = len(images)*epoch + i #第几个样本
                        row = sim_matrix[Config.NUM_CLIENTS][num_i]
                        j = row.argmin()    # 最相似的超原型
                        sim = row[j]        # 相似度

                        preferred_logits = text_logits[j]
                        ref_logits = text_logits_fronzen[j]

                        o_i_embedding = text_prototypes[j]
# x生成的描述对应的向量 然后g_p对应文本的embedding      label = (loss_0)->reward  

# 把sample->reward

                        ### Step7: 计算loss并返回训练
                        loss_0, diff = self.loss_for_image(
                            g_q_logits=x[0],
                            g_p_logits=preferred_logits,
                            g_q_embedding=last_token[0],
                            o_i_embedding=o_i_embedding,
                            sim=sim,
                            beta=beta,
                            ref_logits_preferred=ref_logits, 
                            ref_logits_non_preferred=ref_non_preferred_logits
                        )
                        loss_1 = self.info_NCE_loss(x_text, y, z)

                        loss = loss_0 + beta*loss_1  ##标记*-*

                        loss_history.append(loss.item())
                        loss_0_history.append(loss_0.item())
                        loss_1_history.append(loss_1.item())

                        # Step8: 反向传播
                        self.optimizer.zero_grad()  # 清空梯度
                        loss.backward()             # 梯度回传
                        self.optimizer.step()       # 更新模型参数
                    if batch_idx % 10 == 0:
                        os.makedirs("output/model_config/llava_lora", exist_ok=True)
                        self.model.save_pretrained(f"output/model_config/llava_lora/{batch_idx}") 
                
                if self.logger:
                    self.logger.info(
                        f"服务端图生文训练 - Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        self.get_loss_curve(loss_history, save_num = None)
        self.get_loss_curve(loss_0_history, save_num = 0)
        self.get_loss_curve(loss_1_history, save_num = 1)

    
    def train_for_caption_generator_with_infoNCE_loss_new(self, epochs = Config.IMAGE_GENERATOR_EPOCHS, 
                                 beta = 1.0, prototypes=None, sim_matrix=None,
                                 dataloader=None, text_prototypes=None,
                                 text_logits=None, text_logits_fronzen=None, alpha = 0.5, text_embedding = None):
        """
        训练图像生成器
        z_i对应现有模态, o_i对应超原型信息
        g_q对应生成补全模态, g_p对应超原型相应模态
        """
        self.model.train()
        self.classifier.train()

        # 冻结视觉编码器
        for param in self.model.model.vision_tower.parameters():
            param.requires_grad = False

        # 冻结投影层
        for param in self.model.model.multi_modal_projector.parameters():
            param.requires_grad = False

        loss_history = []
        loss_0_history = []
        loss_1_history = []
        model_preferences = [] 
        final_embeddings = []
        for epoch in range(epochs):
            total_loss = 0.0
            total_acc = 0.0
            num_batches = 0
#先生成描述 然后再训练    1150开始训练的是生成器     
            pre_b=None
            
            with tqdm(dataloader, desc=f"服务端图生文训练 Epoch {epoch + 1}/{epochs}") as pbar:
                for batch_idx, batch in enumerate(pbar, start=1):  
                    
                    # 处理数据并提取特征
                    texts = batch["text"]
                    images = batch["image"]
                    labels = batch["label"]

                    has_caption = batch.get("has_caption", torch.ones(len(texts), dtype=torch.bool))
                    y_score = []
                    batch_sample = []
                    for i in range(len(texts)):

                        if not has_caption[i]:
                            y_score.append(torch.tensor(0,device=self.device))
                            batch_sample.append(torch.tensor(0,device=self.device))
                            final_embeddings.append(torch.zeros(Config.CAT_DIM,device=self.device))
                            #asdf
                            continue
                        # Step1: 得到x
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

                        inputs = self.processor(
                            text = prompt,
                            images = img_pil,
                            return_tensors = "pt"
                        ).to(self.device)   

# 目标原型作为正例，其他为负例 dpo损失用不了 用对比损失(形式参考)  dpo创新度

                        # outputs = self.model.generate(
                        #     **inputs,
                        #      max_new_tokens=77
                        # )
                        # captions = self.processor.batch_decode(
                        #                         outputs, 
                        #                         skip_special_tokens=True, 
                        #                         clean_up_tokenization_spaces=False
                        #                     )[0]                

                        outputs = self.model(
                                **inputs,
                                output_hidden_states=True,
                                return_dict=True
                            )
                        last_hidden = outputs.hidden_states[-1]

                        last_token = last_hidden[:, -1, :]
                        x = self.model.language_model.lm_head(last_token)

                        ### Step2: 得到y
                        num = int(labels[i])
                        if num >= len(text_logits) or torch.all(text_logits[num] == 0):
                            y_score.append(torch.tensor(0,device=self.device))
                            batch_sample.append(torch.tensor(0,device=self.device))
                            final_embeddings.append(torch.zeros(Config.CAT_DIM,device=self.device))
                            print(f"Label {num} 对应的text_logits缺失或空向量，跳过")
                            continue   
                        y = text_logits[num]

                        ### Step3: 得到z
                        total_num = len(text_logits)
                        all_indices = list(range(total_num))
                        all_indices.remove(num)  

                        random.seed(42)  

                        k = 5  
                        z_indices = random.sample(all_indices, k)  
                        z = [text_logits[idx] for idx in z_indices]

                        ### Step4: 得到描述对应的x_caption
                        text = texts[i]

                        prompt = f"USER: {text}\nASSISTANT:"

                        inputs = self.processor(
                            text = prompt,
                            return_tensors = "pt"
                        ).to(self.device) 

                        outputs = self.model(
                                **inputs,
                                output_hidden_states=True,
                                return_dict=True
                            )
                        last_hidden = outputs.hidden_states[-1]
                        x_embedding = last_hidden.mean(dim=1) #调度器的第二个embedding
                        last_token = last_hidden[:, -1, :]
                        x_text = self.model.language_model.lm_head(last_token)
                        x_text = x_text.float()
                        ### Step5: 得到ref_non_preferred_logits
                        prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"

                        inputs = self.processor_fronzen(
                            text = prompt,
                            images = img_pil,
                            return_tensors = "pt"
                        ).to(self.device)                   

                        outputs = self.model_fronzen(
                                **inputs,
                                output_hidden_states=True,
                                return_dict=True
                            )

                        last_hidden = outputs.hidden_states[-1]
                        ref_non_preferred_logits = self.model_fronzen.language_model.lm_head(last_hidden).mean(dim=1)[0]

                        ### Steo6: 得到g_p和o_i
                        num_i = len(images)*epoch + i #第几个样本
                        row = sim_matrix[Config.NUM_CLIENTS][num_i]
                        j = row.argmin()    # 最相似的超原型
                        sim = row[j]        # 相似度

                        preferred_logits = text_logits[j]
                        preferred_embedding = text_embedding[j] #训练调度器的第一个embedding
                        final_embedding = torch.cat((x_embedding[0],preferred_embedding),dim=0) #调度器sample级别对应的embedding
                        final_embeddings.append(final_embedding)

                        ref_logits = text_logits_fronzen[j]

                        o_i_embedding = text_prototypes[j]
# x生成的描述对应的向量 然后g_p对应文本的embedding      label = (loss_0)->reward  

# 把sample->reward

                        ### Step7: 计算loss并返回训练
                        loss_0, diff = self.loss_for_image(
                            g_q_logits=x[0],
                            g_p_logits=preferred_logits,
                            g_q_embedding=last_token[0],
                            o_i_embedding=o_i_embedding,
                            sim=sim,
                            beta=beta,
                            ref_logits_preferred=ref_logits, 
                            ref_logits_non_preferred=ref_non_preferred_logits
                        )
                        
                        y_score.append(loss_0)
                        
                        log_probs_sft_y = F.log_softmax(x[0], dim=-1)
                        r_sample=self.min_max_norm(diff)+1-self.min_max_norm(log_probs_sft_y) #vocab_size
                        batch_sample.append(r_sample.mean()) #batch_sample存的是每个batch里embedding
                        # r_sample=r_samples.mean() #1 
                        loss_1 = self.info_NCE_loss(x_text, y, z)

                        loss = loss_0 + beta*loss_1  ##标记*-*

                        loss_history.append(loss.item())
                        loss_0_history.append(loss_0.item())
                        loss_1_history.append(loss_1.item())

                        # Step8: 反向传播
                        self.optimizer.zero_grad()  # 清空梯度
                        loss.backward()             # 梯度回传
                        self.optimizer.step()       # 更新模型参数
                        
                    # 
                    # if batch_idx % 10 == 0:
                    #     os.makedirs("output/model_config/llava_lora", exist_ok=True)
                    #     self.model.save_pretrained(f"output/model_config/llava_lora/{batch_idx}")
                    r_batch_sample=batch_sample #存的是所有batch的拼接
                    # r_batch
                    # Y = [y1,y2,...,yn]  ->  Y_score #sample_size,1
                    # batch_size = Config.BATCH_SIZE
                    
                    # for i in range(0,len(y_score),batch_size):
                        #if i+batch_size>len(y_score):
                    if batch_idx==1:
                        cur_b=torch.mean(torch.exp(torch.stack(y_score)))
                        cur_len=len(y_score)
                        first_cur_b=cur_b
                        first_len=cur_len
                        first_r_batch_sample=torch.stack(r_batch_sample)
                    elif batch_idx==len(dataloader):
                        hid_tensor=torch.stack(y_score)
                        hid_tensor=torch.cat([hid_tensor,torch.zeros(first_len-len(y_score),device=hid_tensor.device)],dim=0)
                        cur_b=torch.mean(torch.exp(hid_tensor))
                        r_batch=[(cur_b-pre_b)/max(cur_b,pre_b)]*(len(y_score))
                        r_batch_sample=torch.stack(r_batch_sample)
                        r_batch=torch.stack(r_batch)
                        model_preference=alpha*torch.sigmoid(r_batch_sample)+(1-alpha)*torch.sigmoid(r_batch)
                        model_preferences.append(model_preference) #append的是每个batch对应的model_preference
                                                                    #最终维度应该为5*32()√
                        r_batch=[(first_cur_b-cur_b)/max(first_cur_b,cur_b)]*first_len
                        r_batch=torch.stack(r_batch)
                        model_preference=alpha*torch.sigmoid(first_r_batch_sample)+(1-alpha)*torch.sigmoid(r_batch)
                        model_preferences=[model_preference]+model_preferences
                    else:
                        cur_b = torch.mean(torch.exp(torch.stack(y_score)))
                        r_batch=[(cur_b-pre_b)/max(cur_b,pre_b)]*len(y_score)
                        r_batch_sample=torch.stack(r_batch_sample)
                        r_batch=torch.stack(r_batch)
                        model_preference=alpha*torch.sigmoid(r_batch_sample)+(1-alpha)*torch.sigmoid(r_batch)
                        model_preferences.append(model_preference) #append的是每个batch对应的model_preference
                                                                    #最终维度应该为5*32()√
                    pre_b=cur_b
                        # all_tensors = [t for sublist in r_batch_sample for t in sublist]  # 展平二维 list
                        # all_tensors_stacked = torch.stack(all_tensors)
                # import pdb; pdb.set_trace()
                    

                
                if self.logger:
                    self.logger.info(
                        f"服务端图生文训练 - Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        self.get_loss_curve(loss_history, save_num = None)
        self.get_loss_curve(loss_0_history, save_num = 0)
        self.get_loss_curve(loss_1_history, save_num = 1)
        return model_preferences, final_embeddings
        
###bfloat16--->float16 加一个文本约束(单独加)
        
    def min_max_norm(self, x):
        min_x=torch.min(x,dim=-1,keepdim=True)[0]
        max_x=torch.max(x,dim=-1,keepdim=True)[0]
        return (x- min_x) / (max_x- min_x)
    
    def info_NCE_loss(self, x, y, z):
        loss = Loss_function()
        return loss.info_nce_loss(x, y, z)
    
    def get_loss_curve(self,loss_history, save_num = None):
        import os
        import matplotlib.pyplot as plt
        save_dir = "/root/autodl-tmp/46_FedMMDG/outputs/loss_curve"
        os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.plot(loss_history, label="Loss", marker='o')
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)
        if save_num != None:
            save_path = os.path.join(save_dir, f"loss_curve_{save_num}.png")
        else:
            save_path = os.path.join(save_dir, "loss_curve.png")
        plt.savefig(save_path, bbox_inches='tight') 
        plt.close() 
    
    def generate_captions_new(self, dataloader=None, max_length=77, aggregation=False):
        """
        为dataloader生成描述
        """
        device = self.device
        output_dir = os.path.join(Config.OUTPUT_DIR, "generated_captions")
        os.makedirs(output_dir, exist_ok=True)
        save_dir = self.get_captions_dir(output_dir, aggregation)

        with tqdm(dataloader, desc=f"为服务端生成描述") as pbar:
            for batch_idx, batch in enumerate(pbar):

                # 处理数据并提取特征
                texts = batch["text"]
                images = batch["image"]

                image_ids = batch["image_id"]

                has_caption = batch.get("has_caption", torch.ones(len(texts), dtype=torch.bool))

                batch_text = []
                for i in range(len(texts)):
                    if has_caption[i]:
                        continue

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

                    prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"

                    inputs = self.processor(
                        text = prompt,
                        images = img_pil,
                        return_tensors = "pt"
                    ).to(self.device)                   

                    outputs = self.model.generate(
                            **inputs,
                             max_new_tokens=max_length
                        )
                    captions = self.processor.batch_decode(
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
                        
    def get_captions_dir_new(self, output_dir = None, aggregation = False):
        if aggregation:
            server_output_dir = output_dir
        else:
            server_output_dir = os.path.join(output_dir,"server_cub")
            os.makedirs(server_output_dir, exist_ok=True)
        return server_output_dir

    def generate_captions_cub_new(self, dataloader=None, max_length=77, aggregation=False):
            """
            为dataloader生成描述
            """
            device = self.device
            output_dir = os.path.join(Config.OUTPUT_DIR, "generated_captions")
            os.makedirs(output_dir, exist_ok=True)
            save_dir = self.get_captions_dir_new(output_dir, aggregation)
    
            with tqdm(dataloader, desc=f"为服务端生成描述") as pbar:
                for batch_idx, batch in enumerate(pbar):
    
                    # 处理数据并提取特征
                    texts = batch["text"]
                    images = batch["image"]
    
                    image_ids = batch["image_id"]
    
                    has_caption = batch.get("has_caption", torch.ones(len(texts), dtype=torch.bool))
    
                    batch_text = []
                    for i in range(len(texts)):
                        if has_caption[i]:
                            continue
    
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
    
                        prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
    
                        inputs = self.processor(
                            text = prompt,
                            images = img_pil,
                            return_tensors = "pt"
                        ).to(self.device)                   
    
                        outputs = self.model.generate(
                                **inputs,
                                 max_new_tokens=max_length
                            )
                        captions = self.processor.batch_decode(
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
                            
    

                    



                

                    


                    

