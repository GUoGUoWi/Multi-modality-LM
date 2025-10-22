import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, BlipModel, BlipProcessor
from config_cub import Config, ModelType, CaptionModelType
import os
from PIL import Image
import numpy as np


class ModelInterface(nn.Module):
    """模型接口基类，为不同模型提供统一的API"""

    def __init__(self, model_type, num_classes, device=None):
        """
        初始化模型接口

        Args:
            model_type: 模型类型
            num_classes: 分类类别数
            device: 计算设备
        """
        super(ModelInterface, self).__init__()
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = device if device is not None else Config.DEVICE

        # 初始化模型和处理器
        self._init_model_and_processor()

        # 初始化分类头 - 添加BatchNorm和Dropout提高性能
        hidden_size = self._get_hidden_size()

        # self.classifier = nn.Sequential(
        #     nn.BatchNorm1d(hidden_size),
        #     nn.Dropout(0.2),
        #     nn.Linear(hidden_size, num_classes)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, self.num_classes)
        ).to(self.device)

        self.classifier.to(self.device)

    def _init_model_and_processor(self):
        """初始化预训练模型和处理器，需要在子类中实现"""
        raise NotImplementedError

    def _get_hidden_size(self):
        """获取特征维度，需要在子类中实现"""
        raise NotImplementedError

    def _extract_features(self, inputs):
        """提取特征，需要在子类中实现"""
        raise NotImplementedError

    def _extract_modal_features(self, inputs):
        """
        分别提取文本和图像特征

        Args:
            inputs: 预处理后的输入

        Returns:
            文本特征和图像特征
        """
        raise NotImplementedError

    def classify_with_features(self, features_dict):
        """
        使用融合特征进行分类（适配新的融合特征格式）

        Args:
            features_dict: 特征字典，包含不同模态的特征

        Returns:
            分类logits
        """
        text_features = features_dict.get('text', None)
        image_features = features_dict.get('image', None)

        batch_size = 0

        # 确定batch_size
        if text_features is not None:
            batch_size = text_features.size(0)
        elif image_features is not None:
            batch_size = image_features.size(0)
        else:
            raise ValueError("特征字典中缺少文本和图像特征")

        # 获取本模型的隐藏层大小
        hidden_size = self._get_hidden_size()

        # 创建合并特征tensor
        combined_features = torch.zeros((batch_size, hidden_size), device=self.device)

        # 处理融合特征的分类
        if text_features is not None and image_features is not None:
            # 如果两种模态都存在，进行融合
            if text_features.size(1) == image_features.size(1):
                # 如果维度相同，取平均
                fused_features = (text_features + image_features) / 2
            else:
                # 如果维度不同，拼接
                fused_features = torch.cat([text_features, image_features], dim=1)

            # 如果融合特征维度与模型隐藏层维度不匹配，进行适配
            if fused_features.size(1) == hidden_size:
                combined_features = fused_features
            elif fused_features.size(1) == hidden_size // 2:
                # 复制特征来匹配维度
                combined_features = torch.cat([fused_features, fused_features], dim=1)
            else:
                # 使用线性变换来适配维度差异
                if not hasattr(self, 'feature_adapter'):
                    self.feature_adapter = nn.Linear(fused_features.size(1), hidden_size).to(self.device)
                combined_features = self.feature_adapter(fused_features)

        elif text_features is not None:
            # 只有文本特征
            if text_features.size(1) == hidden_size:
                combined_features = text_features
            elif text_features.size(1) == hidden_size // 2:
                combined_features = torch.cat([text_features, text_features], dim=1)
            else:
                if not hasattr(self, 'text_adapter'):
                    self.text_adapter = nn.Linear(text_features.size(1), hidden_size).to(self.device)
                combined_features = self.text_adapter(text_features)

        elif image_features is not None:
            # 只有图像特征
            if image_features.size(1) == hidden_size:
                combined_features = image_features
            elif image_features.size(1) == hidden_size // 2:
                combined_features = torch.cat([image_features, image_features], dim=1)
            else:
                if not hasattr(self, 'image_adapter'):
                    self.image_adapter = nn.Linear(image_features.size(1), hidden_size).to(self.device)
                combined_features = self.image_adapter(image_features)

        # 分类
        logits = self.classifier(combined_features)

        return logits

    def forward(self, inputs):
        """
        前向传播

        Args:
            inputs: 包含图像和文本的字典

        Returns:
            分类logits
        """
        # 提取特征
        features = self._extract_features(inputs)

        # 分类
        logits = self.classifier(features)

        return logits

    def preprocess(self, batch):
        """
        预处理输入数据

        Args:
            batch: 数据批次

        Returns:
            预处理后的输入数据
        """
        raise NotImplementedError


class CLIPInterface(ModelInterface):
    """CLIP模型接口"""

    def _init_model_and_processor(self):
        """初始化CLIP模型和处理器"""
        if self.model_type == ModelType.CLIP_BASE:
            model_name = "openai/clip-vit-base-patch32"
            local_dir = os.path.join(Config.MODEL_CACHE_DIR, "clip-vit-base-patch32")
        elif self.model_type == ModelType.CLIP_LARGE:
            model_name = "openai/clip-vit-large-patch14"
            local_dir = os.path.join(Config.MODEL_CACHE_DIR, "clip-vit-large-patch14")
        else:
            raise ValueError(f"不支持的CLIP模型类型: {self.model_type}")

        # 优先使用本地模型，如果本地模型目录存在
        try:
            if os.path.exists(local_dir) and os.path.isdir(local_dir):
                print(f"从本地加载CLIP模型: {local_dir}")
                self.processor = CLIPProcessor.from_pretrained(local_dir, local_files_only=True,
                                                               clean_up_tokenization_spaces=False)
                self.model = CLIPModel.from_pretrained(local_dir, local_files_only=True)
            else:
                # 如果本地模型不存在，尝试从Hugging Face下载并给出提示信息
                print(f"本地未找到模型 {local_dir}，尝试从Hugging Face下载...")
                print(f"如果下载失败，请手动下载模型到目录: {local_dir}")
                print(f"可以使用命令: huggingface-cli download {model_name} --local-dir {local_dir}")
                self.processor = CLIPProcessor.from_pretrained(model_name, clean_up_tokenization_spaces=False)
                self.model = CLIPModel.from_pretrained(model_name)

            # 验证模型加载
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"CLIP模型参数数量: {param_count:,}")

        except Exception as e:
            print(f"加载CLIP模型时出错: {e}")
            raise

        # 冻结CLIP模型
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(self.device)

    def _get_hidden_size(self):
        """获取CLIP特征维度"""
        # 拼接特征，所以维度是两倍的隐藏维度
        return self.model.config.text_config.hidden_size * 2

    def _extract_features(self, inputs):
        """
        从输入中提取CLIP特征

        Args:
            inputs: 预处理后的CLIP输入

        Returns:
            融合后的特征
        """
        # 确保所有输入在同一设备上 (主GPU)
        device = self.device
        processed_inputs = {}

        # 处理嵌套字典
        for key, value in inputs.items():
            if isinstance(value, dict):
                processed_inputs[key] = {
                    inner_key: inner_value.to(device) if isinstance(inner_value, torch.Tensor) else inner_value
                    for inner_key, inner_value in value.items()}
            elif isinstance(value, torch.Tensor):
                processed_inputs[key] = value.to(device)
            else:
                processed_inputs[key] = value

        # 保存has_image标志并从inputs中移除，以免传递给模型
        has_image = processed_inputs.pop("has_image", None)
        has_caption = processed_inputs.pop("has_caption",None)
        # 获取图像和文本特征
        with torch.no_grad():
            outputs = self.model(**processed_inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds

        # 如果has_image未提供，假设所有样本都有图像
        batch_size = text_features.shape[0]
        if has_image is None:
            has_image = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            has_image = has_image.to(device)

        #若has_caption未提供，则默认所有样本都有描述
        if has_caption is None:
            has_caption = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            has_caption = has_caption.to(device)

        # 创建结果特征张量
        combined_features = torch.zeros((batch_size, self._get_hidden_size()), device=device)

        # 有图像 & 有描述
        if torch.any(has_image & has_caption):
            indices = torch.where(has_image & has_caption)[0]
            for idx in indices:
                combined_features[idx] = torch.cat([text_features[idx], image_features[idx]], dim=0)

        # 没图像 & 有描述
        if torch.any(~has_image & has_caption):
            indices = torch.where(~has_image & has_caption)[0]
            for idx in indices:
                combined_features[idx] = torch.cat([text_features[idx], text_features[idx]], dim=0)

        # 有图像 & 没描述
        if torch.any(has_image & ~has_caption):
            indices = torch.where(has_image & ~has_caption)[0]
            for idx in indices:
                combined_features[idx] = torch.cat([image_features[idx], image_features[idx]], dim=0)

        # 都没有（可选 fallback）
        if torch.any(~has_image & ~has_caption):
            indices = torch.where(~has_image & ~has_caption)[0]
            for idx in indices:
                combined_features[idx] = torch.zeros_like(torch.cat([text_features[idx], text_features[idx]], dim=0))

        # 使用L2归一化，提高特征质量
        combined_features = F.normalize(combined_features, p=2, dim=1)

        return combined_features

    def _extract_modal_features(self, inputs):
        """
        分别提取文本和图像特征

        Args:
            inputs: 预处理后的CLIP输入

        Returns:
            文本特征和图像特征
        """
        # 确保所有输入在同一设备上
        device = self.device
        processed_inputs = {}

        # 处理嵌套字典
        for key, value in inputs.items():
            if isinstance(value, dict):
                processed_inputs[key] = {
                    inner_key: inner_value.to(device) if isinstance(inner_value, torch.Tensor) else inner_value
                    for inner_key, inner_value in value.items()}
            elif isinstance(value, torch.Tensor):
                processed_inputs[key] = value.to(device)
            else:
                processed_inputs[key] = value

        # 保存has_image标志并从inputs中移除
        has_image = processed_inputs.pop("has_image", None)
        has_caption = processed_inputs.pop("has_caption", None)
        # 获取图像和文本特征
        with torch.no_grad():
            outputs = self.model(**processed_inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds

        # 如果has_image未提供，假设所有样本都有图像
        batch_size = text_features.shape[0]
        if has_image is None:
            has_image = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            has_image = has_image.to(device)

        # 如果has_caption未提供，假设所有样本都有图像
        if has_caption is None:
            has_caption = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            has_caption = has_caption.to(device)

        # 对于没有图像的样本，将图像特征设为零向量
        if torch.any(~has_image):
            no_image_indices = torch.where(~has_image)[0]
            for idx in no_image_indices:
                image_features[idx] = torch.zeros_like(image_features[idx])

        # 对于没有文本的样本，将文本特征设为空描述
        if torch.any(~has_caption):
            no_caption_indices = torch.where(~has_caption)[0]
            for idx in no_caption_indices:
                text_features[idx] = torch.zeros_like(text_features[idx])

        return text_features, image_features

    def preprocess(self, batch):
        """
        预处理CLIP输入

        Args:
            batch: 数据批次

        Returns:
            预处理后的CLIP输入
        """
        texts = batch["text"]
        has_image = batch.get("has_image", torch.ones(len(texts), dtype=torch.bool))
        has_caption = batch.get("has_caption", torch.ones(len(texts), dtype=torch.bool))
        # 对于没有描述的样本，使用空字符串
        processed_texts = []
        for i, text in enumerate(texts):
            if has_caption[i]:
                processed_texts.append(text)
            else:
                # 对于没有描述的样本，使用空描述或通用描述
                processed_texts.append("")
        # 从张量生成图像列表 - 这是关键修改
        # CLIP处理器要求原始PIL图像或numpy数组，但我们只有张量
        # 因此，我们将张量转换为numpy数组
        images = []
        for i, tensor in enumerate(batch["image"]):
            if has_image[i]:
                # 创建一个副本，防止修改原始张量
                img_tensor = tensor.clone().cpu().detach()

                # 如果是归一化的图像，我们需要反归一化它
                # 这是基于ImageNet的标准归一化参数
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = img_tensor * std + mean

                # 裁剪到[0, 1]范围
                img_tensor = torch.clamp(img_tensor, 0, 1)

                # 转换为PIL图像
                # 将通道从(C,H,W)移动到(H,W,C)，然后转换为numpy数组
                img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                images.append(img_pil)
            else:
                # 对于没有图像的样本，创建一个灰色图像而非黑色图像
                # 这样可能产生更有意义的特征
                dummy_img = Image.new('RGB', (224, 224), (128, 128, 128))
                images.append(dummy_img)

        # 使用处理器处理文本和图像
        inputs = self.processor(
            text=processed_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77  # CLIP的标准文本长度
        )

        # 添加has_image标志
        inputs["has_image"] = has_image
        inputs["has_caption"] = has_caption

        return inputs

class BLIPInterface(ModelInterface):
    """BLIP模型接口"""

    def _init_model_and_processor(self):
        """初始化BLIP模型和处理器"""
        if self.model_type == ModelType.BLIP_BASE or self.model_type==CaptionModelType.BLIP_BASE:
            model_name = "Salesforce/blip-image-captioning-base"
            local_dir = os.path.join(Config.MODEL_CACHE_DIR, "blip-image-captioning-base")
        elif self.model_type == ModelType.BLIP_LARGE:
            model_name = "Salesforce/blip-image-captioning-large"
            local_dir = os.path.join(Config.MODEL_CACHE_DIR, "blip-image-captioning-large")
        else:
            raise ValueError(f"不支持的BLIP模型类型: {self.model_type}")

        # 优先使用本地模型，如果本地模型目录存在
        if os.path.exists(local_dir) and os.path.isdir(local_dir):
            print(f"从本地加载BLIP模型: {local_dir}")
            self.processor = BlipProcessor.from_pretrained(local_dir, local_files_only=True,
                                                           clean_up_tokenization_spaces=False)
            self.model = BlipModel.from_pretrained(local_dir, local_files_only=True)
        else:
            # 如果本地模型不存在，尝试从Hugging Face下载并给出提示信息
            print(f"本地未找到模型 {local_dir}，尝试从Hugging Face下载...")
            print(f"如果下载失败，请手动下载模型到目录: {local_dir}")
            print(f"可以使用命令: huggingface-cli download {model_name} --local-dir {local_dir}")
            self.processor = BlipProcessor.from_pretrained(model_name, clean_up_tokenization_spaces=False)
            self.model = BlipModel.from_pretrained(model_name)

        # 冻结BLIP模型
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(self.device)

    def _get_hidden_size(self):
        """获取BLIP特征维度"""
        # BLIP的特征维度
        return self.model.config.text_config.hidden_size * 2

    def _extract_features(self, inputs):
        """
        从输入中提取BLIP特征

        Args:
            inputs: 预处理后的BLIP输入

        Returns:
            融合后的特征
        """
        # 确保所有输入在同一设备上
        device = self.device
        processed_inputs = {}

        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                processed_inputs[key] = value.to(device)
            else:
                processed_inputs[key] = value

        # 保存has_image标志并从inputs中移除，以免传递给模型
        has_image = processed_inputs.pop("has_image", None)
        has_caption = processed_inputs.pop("has_caption",None)
        # 获取图像和文本特征
        with torch.no_grad():
            outputs = self.model(**processed_inputs)

            # BLIP模型的输出结构不同，text_embeds和image_embeds是不同的
            # 我们需要从不同的输出结构中提取特征

            # 从text_model_output中获取文本特征
            if hasattr(outputs, 'text_model_output') and outputs.text_model_output is not None:
                if hasattr(outputs.text_model_output, 'last_hidden_state'):
                    # 使用[CLS] token作为文本特征
                    text_features = outputs.text_model_output.last_hidden_state[:, 0, :]
                elif hasattr(outputs.text_model_output, 'pooler_output'):
                    # 使用pooler_output作为文本特征
                    text_features = outputs.text_model_output.pooler_output
                else:
                    raise ValueError("无法从BLIP输出中提取文本特征")
            else:
                # 尝试直接从outputs获取text_embeds
                if hasattr(outputs, 'text_embeds'):
                    text_features = outputs.text_embeds
                else:
                    raise ValueError("BLIP输出中没有文本特征")

            # 从vision_model_output中获取图像特征
            if hasattr(outputs, 'vision_model_output') and outputs.vision_model_output is not None:
                if hasattr(outputs.vision_model_output, 'last_hidden_state'):
                    # 使用[CLS] token作为图像特征
                    image_features = outputs.vision_model_output.last_hidden_state[:, 0, :]
                elif hasattr(outputs.vision_model_output, 'pooler_output'):
                    # 使用pooler_output作为图像特征
                    image_features = outputs.vision_model_output.pooler_output
                else:
                    raise ValueError("无法从BLIP输出中提取图像特征")
            else:
                # 尝试直接从outputs获取image_embeds
                if hasattr(outputs, 'image_embeds'):
                    image_features = outputs.image_embeds
                else:
                    raise ValueError("BLIP输出中没有图像特征")

        # 如果has_image未提供，假设所有样本都有图像
        batch_size = text_features.shape[0]
        if has_image is None:
            has_image = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        else:
            has_image = has_image.to(self.device)
        #如果has_caption未提供，假设所有样本都有描述
        batch_size = text_features.shape[0]
        if has_caption is None:
            has_caption = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        else:
            has_caption = has_caption.to(self.device)
        
        # 创建结果特征张量
        combined_features = torch.zeros((batch_size, self._get_hidden_size()), device=self.device)

        # 对于有图像的样本，使用文本和图像特征的拼接
        if torch.any(has_image):
            has_image_indices = torch.where(has_image)[0]
            for idx in has_image_indices:
                combined_features[idx] = torch.cat([text_features[idx], image_features[idx]], dim=0)

        # 对于没有图像的样本，拼接两次文本特征来保持维度一致
        if torch.any(~has_image):
            no_image_indices = torch.where(~has_image)[0]
            for idx in no_image_indices:
                combined_features[idx] = torch.cat([text_features[idx], text_features[idx]], dim=0)
         # 对于没有文本的样本，拼接两次图像特征来保持维度一致
        if torch.any(~has_caption):
            no_caption_indices = torch.where(~has_caption)[0]
            for idx in no_caption_indices:
                combined_features[idx] = torch.cat([image_features[idx], image_features[idx]], dim=0)
        # 使用L2归一化，提高特征质量
        combined_features = F.normalize(combined_features, p=2, dim=1)

        return combined_features

    def _extract_modal_features(self, inputs):
        """
        分别提取文本和图像特征

        Args:
            inputs: 预处理后的BLIP输入

        Returns:
            文本特征和图像特征
        """
        # 确保所有输入在同一设备上
        device = self.device
        processed_inputs = {}

        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                processed_inputs[key] = value.to(device)
            else:
                processed_inputs[key] = value

        # 保存has_image标志并从inputs中移除
        has_image = processed_inputs.pop("has_image", None)

        # 获取图像和文本特征
        with torch.no_grad():
            outputs = self.model(**processed_inputs)

            # 从text_model_output中获取文本特征
            if hasattr(outputs, 'text_model_output') and outputs.text_model_output is not None:
                if hasattr(outputs.text_model_output, 'last_hidden_state'):
                    text_features = outputs.text_model_output.last_hidden_state[:, 0, :]
                elif hasattr(outputs.text_model_output, 'pooler_output'):
                    text_features = outputs.text_model_output.pooler_output
                else:
                    raise ValueError("无法从BLIP输出中提取文本特征")
            else:
                if hasattr(outputs, 'text_embeds'):
                    text_features = outputs.text_embeds
                else:
                    raise ValueError("BLIP输出中没有文本特征")

            # 从vision_model_output中获取图像特征
            if hasattr(outputs, 'vision_model_output') and outputs.vision_model_output is not None:
                if hasattr(outputs.vision_model_output, 'last_hidden_state'):
                    image_features = outputs.vision_model_output.last_hidden_state[:, 0, :]
                elif hasattr(outputs.vision_model_output, 'pooler_output'):
                    image_features = outputs.vision_model_output.pooler_output
                else:
                    raise ValueError("无法从BLIP输出中提取图像特征")
            else:
                if hasattr(outputs, 'image_embeds'):
                    image_features = outputs.image_embeds
                else:
                    raise ValueError("BLIP输出中没有图像特征")

        # 如果has_image未提供，假设所有样本都有图像
        batch_size = text_features.shape[0]
        if has_image is None:
            has_image = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            has_image = has_image.to(device)

        # 对于没有图像的样本，将图像特征设为零向量
        if torch.any(~has_image):
            no_image_indices = torch.where(~has_image)[0]
            for idx in no_image_indices:
                image_features[idx] = torch.zeros_like(image_features[idx])

        # 使用L2归一化
        text_features = F.normalize(text_features, p=2, dim=1)
        image_features = F.normalize(image_features, p=2, dim=1)

        return text_features, image_features

    def preprocess(self, batch):
        """
        预处理BLIP输入

        Args:
            batch: 数据批次

        Returns:
            预处理后的BLIP输入
        """
        texts = batch["text"]
        has_image = batch.get("has_image", torch.ones(len(texts), dtype=torch.bool))
        has_caption = batch.get("has_caption", torch.ones(len(texts), dtype=torch.bool))
        # 对于没有描述的样本，使用空字符串
        processed_texts = []
        for i, text in enumerate(texts):
            if has_caption[i]:
                processed_texts.append(text)
            else:
                # 对于没有描述的样本，使用空描述或通用描述
                processed_texts.append("")
        # 与CLIP类似，从张量生成图像列表
        images = []
        for i, tensor in enumerate(batch["image"]):
            if has_image[i]:
                # 创建一个副本，防止修改原始张量
                img_tensor = tensor.clone().cpu().detach()

                # 如果是归一化的图像，我们需要反归一化它
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = img_tensor * std + mean

                # 裁剪到[0, 1]范围
                img_tensor = torch.clamp(img_tensor, 0, 1)

                # 转换为PIL图像
                img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                images.append(img_pil)
            else:
                # 对于没有图像的样本，创建一个灰色图像
                dummy_img = Image.new('RGB', (224, 224), (128, 128, 128))
                images.append(dummy_img)

        # 使用处理器处理图像和文本
        inputs = self.processor(
            images=images,
            text=processed_texts,
            return_tensors="pt",
            padding="max_length",  # 确保一致的输出维度
            truncation=True,
            max_length=77
        )

        # 添加has_image标志
        inputs["has_image"] = has_image
        inputs["has_caption"] = has_caption
        return inputs


def get_model(model_type, num_classes, device=None):
    """
    根据模型类型获取相应的模型接口

    Args:
        model_type: 模型类型
        num_classes: 分类类别数
        device: 计算设备

    Returns:
        模型接口实例
    """
    if model_type in [ModelType.CLIP_BASE, ModelType.CLIP_LARGE]:
        return CLIPInterface(model_type, num_classes, device)
    elif model_type in [ModelType.BLIP_BASE, ModelType.BLIP_LARGE, CaptionModelType.BLIP_BASE]:
        return BLIPInterface(model_type, num_classes, device)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")