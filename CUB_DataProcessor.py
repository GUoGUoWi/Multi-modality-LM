import os
import hashlib
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from config_cub import Config  
import random
from collections import defaultdict
import json
from sklearn.cluster import KMeans
import ot

class CUBDataset(Dataset):
    """CUB-200-2011 数据集"""

    def __init__(self, data_list, image_dir, attr_dir=None, transform=None,
                 text_only=False, use_generated_images=False, use_generated_captions=False,
                 aggregation=False, image_id_to_path=None):
        """
        Args:
            data_list: 包含每条数据的字典列表，至少包含 'image_id', 'caption', 'category_id'
            image_dir: 原始图像目录
            attr_dir: 属性或生成的描述目录
            transform: 图像转换
            text_only: 是否仅使用文本
            use_generated_images: 是否使用生成图像
            use_generated_captions: 是否使用生成的描述
            aggregation: 是否使用聚合模式
        """
        self.data_list = data_list
        self.image_dir = image_dir
        self.attr_dir = attr_dir
        self.text_only = text_only
        self.use_generated_images = use_generated_images
        self.use_generated_captions = use_generated_captions
        self.aggregation = aggregation
        self.generated_images_dir = os.path.join(Config.OUTPUT_DIR, "generated_images") if use_generated_images else None
        self.generated_captions_dir = os.path.join(Config.OUTPUT_DIR, "generated_captions") if use_generated_captions else None
        self.image_id_to_path = image_id_to_path

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        self.labels = torch.tensor([item["category_id"] for item in data_list], dtype=torch.long)

    def __len__(self):
        return len(self.data_list)

    def _sanitize_filename(self, text, max_length=100):
        """文本转安全文件名"""
        invalid_chars = '<>:"/\\|?*'
        for c in invalid_chars:
            text = text.replace(c, '')
        if len(text) > max_length:
            text = text[:max_length]
        if not text.strip():
            text = "untitled"
        return text.strip()

    def _sanitize_image_id(self, image_id):
        """image_id 格式化为12位字符串"""
        return f"{int(image_id):012d}"

    def _get_generated_image_path(self, item):
        """获取生成图像路径"""
        if not self.use_generated_images or self.attr_dir is None:
            return None

        caption = item.get("caption", "")
        image_id = item.get("image_id", None)

        gen_dir = os.path.join(Config.OUTPUT_DIR, "generated_images")
        if self.aggregation:
            gen_dir = os.path.join(gen_dir, "aggregation")
            os.makedirs(gen_dir, exist_ok=True)

        safe_caption = self._sanitize_filename(caption)
        path1 = os.path.join(gen_dir, f"{safe_caption}.png")
        if os.path.exists(path1):
            return path1
        if image_id is not None:
            path2 = os.path.join(gen_dir, f"{safe_caption}_{image_id}.png")
            if os.path.exists(path2):
                return path2

        # 哈希回退
        enhanced = f"{caption}, realistic photo, natural lighting, high detail, sharp focus, CUB dataset style"
        hash_str = hashlib.md5(enhanced.encode('utf-8')).hexdigest()[:16]
        if image_id is not None:
            path3 = os.path.join(gen_dir, f"{image_id}_{hash_str}.png")
        else:
            path3 = os.path.join(gen_dir, f"{hash_str}.png")
        return path3 if os.path.exists(path3) else None

    def _get_generated_caption_path(self, item):
        """获取生成描述路径"""
        if not self.use_generated_captions or self.attr_dir is None:
            return None

        image_id = item.get("image_id")
        caption = item.get("caption")
        client_id = item.get("client_id", None)

        gen_dir = os.path.join(Config.OUTPUT_DIR, "generated_captions")
        if self.aggregation:
            gen_dir = os.path.join(gen_dir, "aggregation")
            os.makedirs(gen_dir, exist_ok=True)

            safe_id = self._sanitize_image_id(image_id) 
            path1 = os.path.join(gen_dir, f"{safe_id}.txt")
            if os.path.exists(path1):
                return path1
            
            if image_id is not None:
                alt_path = os.path.join(generated_captions_dir, f"{safe_id}_{image_id}.txt")
                if os.path.exists(alt_path):
                    return alt_path

            # 哈希回退
            enhanced = f"{caption}, realistic photo, natural lighting, high detail, sharp focus, CUB dataset style"
            hash_str = hashlib.md5(enhanced.encode('utf-8')).hexdigest()[:16]
            if image_id is not None:
                path2 = os.path.join(gen_dir, f"{image_id}_{hash_str}.txt")
            else:
                path2 = os.path.join(gen_dir, f"{hash_str}.txt")
            return path2 if os.path.exists(path2) else None
        if client_id is not None:
            if client_id == 2:
                generated_captions_dir = os.path.join(self.generated_captions_dir, "server")
            else:
                generated_captions_dir = os.path.join(self.generated_captions_dir, f"{client_id}")
            os.makedirs(generated_captions_dir, exist_ok=True)
        else:
            generated_captions_dir = self.generated_captions_dir
            os.makedirs(generated_captions_dir, exist_ok=True)
        safe_id = self._sanitize_image_id(image_id) 
        path1 = os.path.join(gen_dir, f"{safe_id}.txt")
        if os.path.exists(path1):
            return path1
        
        if image_id is not None:
            alt_path = os.path.join(generated_captions_dir, f"{safe_id}_{image_id}.txt")
            if os.path.exists(alt_path):
                return alt_path

        # 哈希回退
        enhanced = f"{caption}, realistic photo, natural lighting, high detail, sharp focus, CUB dataset style"
        hash_str = hashlib.md5(enhanced.encode('utf-8')).hexdigest()[:16]
        if image_id is not None:
            path2 = os.path.join(gen_dir, f"{image_id}_{hash_str}.txt")
        else:
            path2 = os.path.join(gen_dir, f"{hash_str}.txt")
        return path2 if os.path.exists(path2) else None

    def get_img_path(self, img_id):
        path = self.image_id_to_path[img_id]
        images_root = '/root/autodl-tmp/46_FedMMDG/CUB-200/CUB_200_2011/images/'
        relative_path = os.path.relpath(path, images_root)
        file, img_path = relative_path.split("/")
        img_dir = Config.CUB_IMAGE_DIR
        image_path = os.path.join(img_dir,file)
        image_path = os.path.join(image_path,img_path)
        return image_path

    def __getitem__(self, idx):
        item = self.data_list[idx]

        text = item.get("caption", "")
        has_caption = item.get("has_caption", bool(text.strip()))
        is_generated_caption = item.get("is_generated_caption", False)

        # 加载图像
        if "image_id" in item and item["image_id"] is not None and not self.text_only:
            img_id = item["image_id"]
            img_path = self.get_img_path(img_id)
            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert("RGB")
                    if self.transform:
                        image = self.transform(image)
                except:
                    image = torch.zeros((3, 224, 224))
                    print(f"Error loading image {img_path}: {e}")
            else:
                image = torch.zeros((3, 224, 224))
        else:
            image = torch.zeros((3, 224, 224))

        # 使用生成描述替代缺失描述
        if not has_caption and self.use_generated_captions:
            gen_path = self._get_generated_caption_path(item)
            if gen_path:
                try:
                    with open(gen_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    has_caption = bool(text.strip())
                    item["has_caption"] = has_caption
                    if has_caption:
                        is_generated_caption = True
                        item["is_generated_caption"] = is_generated_caption
                except Exception as e:
                    print(f"Error loading generated caption {gen_path}: {e}")

        label = item["category_id"]

        sample = {
            "image": image,
            "text": text,
            "label": torch.tensor(label, dtype=torch.long),
            "has_caption": torch.tensor(has_caption, dtype=torch.bool),
            "has_image": torch.tensor(True, dtype=torch.bool),
            "is_generated_caption": torch.tensor(is_generated_caption, dtype=torch.bool),
            "idx": idx,
            "image_id": item.get("image_id", -1),
            "image_path": self.get_img_path(img_id)
        }

        return sample
class DataProcessor:
    def __init__(self, logger = None):
        self.root_dir = Config.CUB_ROOT_DIR
        self.logger = logger
        self.images_dir = os.path.join(self.root_dir, "images")
        self.captions_dir = os.path.join(self.root_dir, "captions/text_flower")

        # 标注文件
        self.images_file = os.path.join(self.root_dir, "images.txt")
        self.labels_file = os.path.join(self.root_dir, "image_class_labels.txt")
        self.classes_file = os.path.join(self.root_dir, "classes.txt")

        self.image_id_to_path = {}
        self.image_id_to_label = {}
        self.class_id_to_name = {}
        self.image_id_to_captions = {}

    def get_image_id_to_path(self):
        return self.image_id_to_path
    
    def get_image_id_to_captions(self):
        return self.image_id_to_captions
    
    def _load_images(self):
            """加载 image_id -> image_path"""
            with open(self.images_file, "r") as f:
                for line in f:
                    img_id, rel_path = line.strip().split()
                    self.image_id_to_path[int(img_id)] = os.path.join(self.images_dir, rel_path)

            if self.logger:
                self.logger.info(f"加载了 {len(self.image_id_to_path)} 张图片路径")

    def _load_labels(self):
        """加载 image_id -> class_id"""
        with open(self.labels_file, "r") as f:
            for line in f:
                img_id, class_id = line.strip().split()
                self.image_id_to_label[int(img_id)] = int(class_id) - 1

        if self.logger:
            self.logger.info(f"加载了 {len(self.image_id_to_label)} 个图像标签")

    def _load_classes(self):
        """加载 class_id -> class_name"""
        with open(self.classes_file, "r") as f:
            for line in f:
                class_id, name = line.strip().split()
                self.class_id_to_name[int(class_id)] = name

        if self.logger:
            self.logger.info(f"加载了 {len(self.class_id_to_name)} 个类别")

    def _load_captions(self):
        """加载 image_id -> captions 列表"""
        for img_id, path in self.image_id_to_path.items():
            parts = path.split("images/", 1)
            path = parts[1]
            file_name, img_path = path.split("/")
            img_txt = img_path.replace(".jpg", ".txt")
            caption_file = os.path.join(self.captions_dir, file_name)
            txt_file = os.path.join(caption_file, img_txt)
            if os.path.exists(txt_file):
                with open(txt_file, "r") as f:
                    captions = [line.strip() for line in f.readlines() if line.strip()]
                    self.image_id_to_captions[img_id] = captions
        if self.logger:
            self.logger.info(f"加载了 {len(self.image_id_to_captions)} 张图片的描述")

    def load_and_preprocess_data(self, train_ratio=0.8):
        """
        加载并预处理 CUB-200-2011 数据集

        Args:
            train_ratio: 训练集比例

        Returns:
            dict: {"train": {...}, "val": {...}}
        """
        self._load_images()
        self._load_labels()
        self._load_classes()
        self._load_captions()

        image_ids = list(self.image_id_to_captions.keys())
        random.seed(Config.SEED)

        random.shuffle(image_ids)

        split_idx = int(len(image_ids) * train_ratio)
        train_image_ids = image_ids[:split_idx]
        val_image_ids = image_ids[split_idx:]

        if self.logger:
            self.logger.info(f"训练集: {len(train_image_ids)} 张图片, 验证集: {len(val_image_ids)} 张图片")

        aligned_size = min(Config.ALIGNED_DATA_SIZE, len(train_image_ids))
        non_aligned_size = min(Config.NON_ALIGNED_DATA_SIZE, len(train_image_ids) - aligned_size)

        val_aligned_size = min(aligned_size, len(val_image_ids))
        val_non_aligned_size = min(non_aligned_size, len(val_image_ids) - val_aligned_size)

        train_aligned_images = train_image_ids[:aligned_size]
        train_non_aligned_images = train_image_ids[aligned_size:aligned_size + non_aligned_size]

        val_aligned_images = val_image_ids[:val_aligned_size]
        val_non_aligned_images = val_image_ids[val_aligned_size:val_aligned_size + val_non_aligned_size]

        if self.logger:
            self.logger.info(f"训练集: 对齐数据 {len(train_aligned_images)} 个图像, "
                             f"非对齐数据 {len(train_non_aligned_images)} 个图像")
            self.logger.info(f"验证集: 对齐数据 {len(val_aligned_images)} 个图像, "
                             f"非对齐数据 {len(val_non_aligned_images)} 个图像")

        return {
            "train": {
                "aligned_images": train_aligned_images,
                "non_aligned_images": train_non_aligned_images,
                "image_captions": self.image_id_to_captions,
                "image_categories":self.image_id_to_label
            },
            "val": {
                "aligned_images": val_aligned_images,
                "non_aligned_images": val_non_aligned_images,
                "image_captions": self.image_id_to_captions,
                "image_categories":self.image_id_to_label
            }
        }
    def get_dataloader(self, data_list, batch_size=Config.BATCH_SIZE, is_train=True):
        """
        从数据列表创建数据加载器

        Args:
            data_list: 数据项列表
            batch_size: 批次大小
            is_train: 是否为训练数据

        Returns:
            数据加载器
        """
        # 确定图像目录
        image_dir = Config.CUB_IMAGE_DIR
        caption_dir = Config.CUB_CAPTION_dir

        # 创建数据集
        dataset = CUBDataset(
            data_list,
            image_dir,
            caption_dir,
            image_id_to_path = self.image_id_to_path
        )

        # 自定义collate函数，确保批次中的所有样本都有有效的字段
        def custom_collate_fn(batch):
            # 筛选有效样本
            valid_batch = [item for item in batch if item is not None and "label" in item and item["label"] is not None]

            if len(valid_batch) == 0:
                # 如果没有有效样本，返回一个空的批次（这不应该发生，因为我们已经验证了数据）
                raise ValueError("批次中没有有效样本，这不应该发生，请检查数据处理流程")

            # 对于有效样本，进行标准的批次收集
            images = torch.stack([item["image"] for item in valid_batch])
            texts = [item["text"] for item in valid_batch]

            # 确保label是tensor
            labels = [item["label"] if isinstance(item["label"], torch.Tensor)
                      else torch.tensor(item["label"], dtype=torch.long) for item in valid_batch]
            labels = torch.stack(labels)

            # 确保has_caption是tensor
            has_captions = [
                item.get("has_caption", torch.tensor(False)) if isinstance(item.get("has_caption", False), torch.Tensor)
                else torch.tensor(item.get("has_caption", False), dtype=torch.bool) for item in valid_batch]
            has_captions = torch.stack(has_captions)

            # 确保is_generated_caption是tensor
            is_generated_caption = [
                item.get("is_generated_caption", torch.tensor(False)) if isinstance(item.get("is_generated_caption", False),
                                                                            torch.Tensor)
                else torch.tensor(item.get("is_generated_caption", False), dtype=torch.bool) for item in valid_batch]
            is_generated_caption = torch.stack(is_generated_caption)

            # 确保image_id是tensor
            image_ids = [item.get("image_id", -1) for item in valid_batch]
            image_ids = torch.tensor(image_ids, dtype=torch.long)
            batch_dict = {
                "image": images,
                "text": texts,
                "label": labels,
                "image_id": image_ids,
                "has_caption": has_captions,
                "is_generated_caption": is_generated_caption
            }

            batch_dict["idx"] = torch.tensor([item["idx"] for item in valid_batch], dtype=torch.long)

            return batch_dict

        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=0,  # 使用0避免Windows上的多进程问题
            pin_memory=True,
            collate_fn=custom_collate_fn
        )

        return dataloader

    def split_data_for_clients(self, data, num_clients=Config.NUM_CLIENTS):
        """
        为多个客户端划分数据

        Args:
            data: 预处理后的数据结构
            num_clients: 客户端数量

        Returns:
            划分好的客户端数据
        """
        random.seed(Config.SEED)
        if self.logger:
            self.logger.info(f"为 {num_clients} 个客户端划分数据...")

        # 初始化客户端数据
        self.client_data = {i: {
            "train": {"aligned": [], "non_aligned": []},
            "val": {"aligned": [], "non_aligned": []}
        } for i in range(num_clients)}

        # 处理训练集对齐数据 - 每个客户端都有相同的对齐图像，与相同的描述
        for img_id in data["train"]["aligned_images"]:
            # 获取该图像的所有描述和类别
            captions = data["train"]["image_captions"][img_id][:5]  # 最多取5个描述
            category_id = data["train"]["image_categories"][img_id]

            # 随机打乱描述，然后从中无放回抽样
            random.shuffle(captions)
            captions = captions[:num_clients]  # 确保有足够的描述给所有客户端
            # return {
            # "train": {
            #     "aligned_images": train_aligned_images,
            #     "non_aligned_images": train_non_aligned_images,
            #     "image_captions": self.image_id_to_captions,
            #     "image_categories":self.image_id_to_label
            # },
            # "val": {
            #     "aligned_images": val_aligned_images,
            #     "non_aligned_images": val_non_aligned_images,
            #     "image_captions": self.image_id_to_captions,
            #     "image_categories":self.image_id_to_label
            # }

            # 为每个客户端分配不同的描述
            for client_id in range(min(num_clients, len(captions))):
                has_caption = random.random() > Config.ALIGNED_CAPTION_MISSING_RATIO
                # 创建样本
                sample = {
                    'image_id': img_id,
                    'caption': captions[client_id] if has_caption else "",
                    'category_id': category_id,
                    'client_id': client_id,  # 添加客户端ID以便生成图像或文本时知道来源
                    "has_caption": has_caption,
                    "is_generated_caption": False
                }

                # 添加到客户端的对齐数据中
                self.client_data[client_id]["train"]["aligned"].append(sample)

        # 处理训练集非对齐数据 - 每个客户端都有各自独有的NON_ALIGNED_DATA_SIZE个非对齐样本
        target_non_aligned_per_client = Config.NON_ALIGNED_DATA_SIZE_CLIENT  # 每个客户端应该有这么多非对齐样本

        # 确保有足够的非对齐图像可用
        total_non_aligned_needed = target_non_aligned_per_client * num_clients
        #如果没有足够的非对齐图像可用
        if len(data["train"]["non_aligned_images"]) < total_non_aligned_needed:
            if self.logger:
                self.logger.warning(
                    f"警告：非对齐图像数量 {len(data['train']['non_aligned_images'])} 小于所需数量 {total_non_aligned_needed}")

            # 如果图像不够，可能需要复用图像，但确保客户端之间的样本不重叠
            all_non_aligned_samples = []
            for img_id in data["train"]["non_aligned_images"]:
                captions = data["train"]["image_captions"][img_id]
                category_id = data["train"]["image_categories"][img_id]

                # 为每个图像创建多个样本，使用不同的描述
                for caption in captions:
                    all_non_aligned_samples.append({
                        'image_id': img_id,
                        'caption': caption,
                        'category_id': category_id,
                        "has_caption": True,
                        "is_generated_caption": False
                    })

            # 打乱所有样本
            random.shuffle(all_non_aligned_samples)

            # 为每个客户端分配样本
            for client_id in range(num_clients):
                client_samples = all_non_aligned_samples[client_id * target_non_aligned_per_client:
                                                         (client_id + 1) * target_non_aligned_per_client]

                # 如果样本数量不足，重复使用已有样本（通过深拷贝避免引用相同对象）
                while len(client_samples) < target_non_aligned_per_client:
                    # 从已有样本中随机选择并创建副本
                    sample_to_duplicate = random.choice(all_non_aligned_samples)
                    duplicate_sample = sample_to_duplicate.copy()

                    # 确保使用不同的描述（如果可能）
                    if img_id in data["train"]["image_captions"] and len(data["train"]["image_captions"][img_id]) > 1:
                        current_caption = duplicate_sample['caption']
                        other_captions = [cap for cap in data["train"]["image_captions"][img_id] if
                                          cap != current_caption]
                        if other_captions:
                            duplicate_sample['caption'] = random.choice(other_captions)

                    # 添加客户端ID
                    duplicate_sample['client_id'] = client_id
                    client_samples.append(duplicate_sample)

                #应用描述缺失
                for sample in client_samples:
                    if random.random() < Config.NON_ALIGNED_CAPTION_MISSING_RATIO:
                        sample["caption"] = ''
                        sample["has_caption"] = False
                    sample['client_id'] = client_id

                self.client_data[client_id]["train"]["non_aligned"] = client_samples
        else:
            # 如果有足够的图像，为每个客户端选择独有的非对齐图像
            all_non_aligned_images = data["train"]["non_aligned_images"].copy()
            random.shuffle(all_non_aligned_images)

            for client_id in range(num_clients):
                start_idx = client_id * target_non_aligned_per_client
                end_idx = (client_id + 1) * target_non_aligned_per_client

                client_non_aligned_images = all_non_aligned_images[start_idx:end_idx]

                for img_id in client_non_aligned_images:
                    captions = data["train"]["image_captions"][img_id]
                    category_id = data["train"]["image_categories"][img_id]

                    # 随机选择一个描述
                    caption = random.choice(captions)
                    has_caption = random.random() > Config.NON_ALIGNED_CAPTION_MISSING_RATIO


                    # 创建样本
                    sample = {
                        'image_id': img_id,
                        'caption': caption if has_caption else "",
                        'category_id': category_id,
                        'client_id': client_id,  # 添加客户端ID
                        "has_caption": has_caption,
                        "is_generated_caption": False
                    }

                    # 添加到客户端的非对齐数据中
                    self.client_data[client_id]["train"]["non_aligned"].append(sample)

                # 如果仍然不够，通过使用不同描述来创建更多样本
                while len(self.client_data[client_id]["train"]["non_aligned"]) < target_non_aligned_per_client:
                    # 随机选择一个已分配的图像
                    img_id = random.choice(client_non_aligned_images)
                    captions = data["train"]["image_captions"][img_id]
                    category_id = data["train"]["image_categories"][img_id]

                    # 选择一个不同的描述（如果可能）
                    existing_captions = [s['caption'] for s in self.client_data[client_id]["train"]["non_aligned"]
                                         if s.get('image_id') == img_id]
                    available_captions = [c for c in captions if c not in existing_captions]

                    if not available_captions:  # 如果没有新的描述可用
                        available_captions = captions  # 重用现有描述

                    caption = random.choice(available_captions)
                    #只有第一个客户端缺文本，客户端2缺图像
                    has_caption = random.random() > Config.NON_ALIGNED_CAPTION_MISSING_RATIO

                    sample = {
                        'image_id': img_id,
                        'caption': caption if has_caption else "",
                        'category_id': category_id,
                        'client_id': client_id,  # 添加客户端ID
                        "has_caption": has_caption,
                        "is_generated_caption": False
                    }

                    self.client_data[client_id]["train"]["non_aligned"].append(sample)

        # 对验证集使用类似的逻辑
        # 处理验证集对齐数据
        for img_id in data["val"]["aligned_images"]:
            captions = data["val"]["image_captions"][img_id][:5]
            category_id = data["val"]["image_categories"][img_id]

            random.shuffle(captions)
            captions = captions[:num_clients]

            for client_id in range(min(num_clients, len(captions))):
                has_caption = random.random() > Config.ALIGNED_CAPTION_MISSING_RATIO
                
                sample = {
                    'image_id': img_id,
                    'caption': captions[client_id] if has_caption else "",
                    'category_id': category_id,
                    'client_id': client_id,  # 添加客户端ID
                    "has_caption": has_caption,
                    "is_generated_caption": False
                }

                self.client_data[client_id]["val"]["aligned"].append(sample)

        # 处理验证集非对齐数据 - 与训练集类似
        target_val_non_aligned_per_client = target_non_aligned_per_client  # 验证集和训练集保持一样的样本数

        # 确保有足够的非对齐图像可用
        total_val_non_aligned_needed = target_val_non_aligned_per_client * num_clients
        if len(data["val"]["non_aligned_images"]) < total_val_non_aligned_needed:
            if self.logger:
                self.logger.warning(
                    f"警告：验证集非对齐图像数量 {len(data['val']['non_aligned_images'])} 小于所需数量 {total_val_non_aligned_needed}")

            # 如果图像不够，可能需要复用图像，但确保客户端之间的样本不重叠
            all_val_non_aligned_samples = []
            for img_id in data["val"]["non_aligned_images"]:
                captions = data["val"]["image_captions"][img_id]
                category_id = data["val"]["image_categories"][img_id]

                # 为每个图像创建多个样本，使用不同的描述
                for caption in captions:
                    all_val_non_aligned_samples.append({
                        'image_id': img_id,
                        'caption': caption,
                        'category_id': category_id,
                        "has_caption": True,
                        "is_generated_caption": False
                    })

            # 打乱所有样本
            random.shuffle(all_val_non_aligned_samples)

            # 为每个客户端分配样本
            for client_id in range(num_clients):
                client_samples = all_val_non_aligned_samples[client_id * target_val_non_aligned_per_client:
                                                             (client_id + 1) * target_val_non_aligned_per_client]

                # 如果样本数量不足，重复使用已有样本（通过深拷贝避免引用相同对象）
                while len(client_samples) < target_val_non_aligned_per_client:
                    # 从已有样本中随机选择并创建副本
                    sample_to_duplicate = random.choice(all_val_non_aligned_samples)
                    duplicate_sample = sample_to_duplicate.copy()

                    # 确保使用不同的描述（如果可能）
                    if img_id in data["val"]["image_captions"] and len(data["val"]["image_captions"][img_id]) > 1:
                        current_caption = duplicate_sample['caption']
                        other_captions = [cap for cap in data["val"]["image_captions"][img_id] if
                                          cap != current_caption]
                        if other_captions:
                            duplicate_sample['caption'] = random.choice(other_captions)

                    # 添加客户端ID
                    duplicate_sample['client_id'] = client_id
                    duplicate_sample["has_caption"] = True
                    duplicate_sample["is_generated_caption"] = False
                    client_samples.append(duplicate_sample)

                # 客户端都缺文本
                for sample in client_samples:
                    if random.random() < Config.NON_ALIGNED_CAPTION_MISSING_RATIO:
                        sample["caption"] = ''
                        sample["has_caption"] = False
                    sample['client_id'] = client_id

                self.client_data[client_id]["val"]["non_aligned"] = client_samples
        else:
            # 如果有足够的图像，为每个客户端选择独有的非对齐图像
            all_val_non_aligned_images = data["val"]["non_aligned_images"].copy()
            random.shuffle(all_val_non_aligned_images)

            for client_id in range(num_clients):
                start_idx = client_id * target_val_non_aligned_per_client
                end_idx = (client_id + 1) * target_val_non_aligned_per_client

                client_non_aligned_images = all_val_non_aligned_images[start_idx:end_idx]

                for img_id in client_non_aligned_images:
                    captions = data["val"]["image_captions"][img_id]
                    category_id = data["val"]["image_categories"][img_id]

                    # 随机选择一个描述
                    caption = random.choice(captions)

                    has_caption = random.random() > Config.ALIGNED_CAPTION_MISSING_RATIO

                    # 创建样本
                    sample = {
                        'image_id': img_id,
                        'caption': caption if has_caption else "",
                        'category_id': category_id,
                        'client_id': client_id,  # 添加客户端ID
                        "has_caption": has_caption,
                        "is_generated_caption": False
                    }

                    # 添加到客户端的非对齐数据中
                    self.client_data[client_id]["val"]["non_aligned"].append(sample)

                # 如果仍然不够，通过使用不同描述来创建更多样本
                while len(self.client_data[client_id]["val"]["non_aligned"]) < target_val_non_aligned_per_client:
                    # 随机选择一个已分配的图像
                    img_id = random.choice(client_non_aligned_images)
                    captions = data["val"]["image_captions"][img_id]
                    category_id = data["val"]["image_categories"][img_id]

                    # 选择一个不同的描述（如果可能）
                    existing_captions = [s['caption'] for s in self.client_data[client_id]["val"]["non_aligned"]
                                         if s.get('image_id') == img_id]
                    available_captions = [c for c in captions if c not in existing_captions]

                    if not available_captions:  # 如果没有新的描述可用
                        available_captions = captions  # 重用现有描述

                    caption = random.choice(available_captions)

                    has_caption = random.random() > Config.ALIGNED_CAPTION_MISSING_RATIO

                    sample = {
                        'image_id': img_id,
                        'caption': caption if has_caption else "",
                        'category_id': category_id,
                        'client_id': client_id,  # 添加客户端ID
                        "has_caption": has_caption,
                        "is_generated_caption": False
                    }

                    self.client_data[client_id]["val"]["non_aligned"].append(sample)

        # 记录客户端数据统计信息
        for client_id in range(num_clients):
            train_data = self.client_data[client_id]["train"]
            val_data = self.client_data[client_id]["val"]

            # 检查训练集和验证集类别分布
            train_categories = [item["category_id"] for item in train_data["aligned"] + train_data["non_aligned"]]
            val_categories = [item["category_id"] for item in val_data["aligned"] + val_data["non_aligned"]]

            train_category_counts = {}
            for cat in train_categories:
                train_category_counts[cat] = train_category_counts.get(cat, 0) + 1

            val_category_counts = {}
            for cat in val_categories:
                val_category_counts[cat] = val_category_counts.get(cat, 0) + 1

            train_unique_categories = set(train_categories)
            val_unique_categories = set(val_categories)

            if self.logger:
                self.logger.info(f"客户端 {client_id} 类别分布:")
                self.logger.info(f"  训练集: {len(train_unique_categories)} 个唯一类别")
                self.logger.info(f"  验证集: {len(val_unique_categories)} 个唯一类别")
                self.logger.info(
                    f"  验证集类别是训练集类别的子集: {val_unique_categories.issubset(train_unique_categories)}")

                if not val_unique_categories.issubset(train_unique_categories):
                    missing_categories = val_unique_categories - train_unique_categories
                    self.logger.warning(
                        f"  验证集中有 {len(missing_categories)} 个类别在训练集中不存在")

            # 计算有图像的样本数量
            train_with_image = sum(
                1 for item in train_data["aligned"] + train_data["non_aligned"] if item.get("image_id") is not None)
            val_with_image = sum(
                1 for item in val_data["aligned"] + val_data["non_aligned"] if item.get("image_id") is not None)

            # 计算唯一类别数量
            train_categories = set(item["category_id"] for item in train_data["aligned"] + train_data["non_aligned"])
            val_categories = set(item["category_id"] for item in val_data["aligned"] + val_data["non_aligned"])

            # 记录统计信息
            train_stats = {
                "total": len(train_data["aligned"]) + len(train_data["non_aligned"]),
                "aligned": len(train_data["aligned"]),
                "non_aligned": len(train_data["non_aligned"]),
                "with_image": train_with_image,
                "image_ratio": train_with_image / (len(train_data["aligned"]) + len(train_data["non_aligned"])) if (
                                                                                                                           len(
                                                                                                                               train_data[
                                                                                                                                   "aligned"]) + len(
                                                                                                                       train_data[
                                                                                                                           "non_aligned"])) > 0 else 0,
                "num_categories": len(train_categories)
            }

            val_stats = {
                "total": len(val_data["aligned"]) + len(val_data["non_aligned"]),
                "aligned": len(val_data["aligned"]),
                "non_aligned": len(val_data["non_aligned"]),
                "with_image": val_with_image,
                "image_ratio": val_with_image / (len(val_data["aligned"]) + len(val_data["non_aligned"])) if (
                                                                                                                     len(
                                                                                                                         val_data[
                                                                                                                             "aligned"]) + len(
                                                                                                                 val_data[
                                                                                                                     "non_aligned"])) > 0 else 0,
                "num_categories": len(val_categories)
            }

            if self.logger:
                self.logger.info(f"客户端 {client_id} 数据统计: 训练数据 {train_stats}, 验证数据 {val_stats}")

        return self.client_data

    def get_client_dataloaders(self, client_id, batch_size=Config.BATCH_SIZE, text_only=False, aggregation=False,
                               use_generated_images=False, use_generated_captions=False ,for_non_aligned=False, 
                               for_local_non_aligned=False, for_server = False):
        """
        获取指定客户端的数据加载器

        Args:
            client_id: 客户端ID
            batch_size: 批次大小
            text_only: 是否仅使用文本特征
            use_generated_images: 是否使用生成的图像
            use_generated_captions: 是否使用生成的描述
            aggregation: 是否聚合
            for_non_aligned: 是否非对齐数据也要替换
            for_server: 是否为主动方
        Returns:
            训练和验证数据加载器
        """
        if not self.client_data:
            raise ValueError("请先调用 split_data_for_clients 函数划分数据")

        if client_id not in self.client_data:
            raise ValueError(f"客户端ID {client_id} 不存在")

        client_data = self.client_data[client_id]
        random.seed(Config.SEED)
        #若为主动方 则只使用非对齐数据
        if for_server == True:
            train_data = client_data["train"]["non_aligned"]
            val_data = client_data["val"]["non_aligned"]

            train_dataset = CUBDataset(
            client_data["train"]["non_aligned"],
            Config.TRAIN_IMAGE_DIR,
            Config.TRAIN_ANNOTATION_FILE,
            text_only=text_only,
            use_generated_images=use_generated_images and for_non_aligned,
            use_generated_captions=use_generated_captions and for_non_aligned,
            aggregation=aggregation and not for_local_non_aligned,
            image_id_to_path = self.image_id_to_path
                                                     )
            
            ###创建验证集的数据集
            val_aligned_dataset = CUBDataset(
                client_data["val"]["aligned"],
                Config.TRAIN_IMAGE_DIR,
                Config.TRAIN_ANNOTATION_FILE,
                text_only=text_only,
                use_generated_images=use_generated_images,
                use_generated_captions=use_generated_captions,
                aggregation=aggregation,
                image_id_to_path = self.image_id_to_path
                                                        )
            
            val_non_aligned_dataset = CUBDataset(
                client_data["val"]["non_aligned"],
                Config.TRAIN_IMAGE_DIR,
                Config.TRAIN_ANNOTATION_FILE,
                text_only=text_only,
                use_generated_images=use_generated_images and for_non_aligned,
                use_generated_captions=use_generated_captions and for_non_aligned,
                aggregation=aggregation and not for_local_non_aligned,
                image_id_to_path = self.image_id_to_path
                                                        )
            # 自定义collate函数，确保批次中的所有样本都有有效的字段
            def custom_collate_fn(batch):
                # 筛选有效样本
                valid_batch = [item for item in batch if item is not None and "label" in item and item["label"] is not None]

                if len(valid_batch) == 0:
                    # 如果没有有效样本，返回一个空的批次（这不应该发生，因为我们已经验证了数据）
                    raise ValueError("批次中没有有效样本，这不应该发生，请检查数据处理流程")

                # 对于有效样本，进行标准的批次收集
                images = torch.stack([item["image"] for item in valid_batch])
                texts = [item["text"] for item in valid_batch]

                # 确保label是tensor
                labels = [item["label"] if isinstance(item["label"], torch.Tensor)
                        else torch.tensor(item["label"], dtype=torch.long) for item in valid_batch]
                labels = torch.stack(labels)

                # 确保has_caption是tensor
                has_captions = [
                    item.get("has_caption", torch.tensor(False)) if isinstance(item.get("has_caption", False), torch.Tensor)
                    else torch.tensor(item.get("has_caption", False), dtype=torch.bool) for item in valid_batch]
                has_captions = torch.stack(has_captions)

                # 确保is_generated_caption是tensor
                is_generated_caption = [
                    item.get("is_generated_caption", torch.tensor(False)) if isinstance(item.get("is_generated_caption", False),
                                                                                torch.Tensor)
                    else torch.tensor(item.get("is_generated_caption", False), dtype=torch.bool) for item in valid_batch]
                is_generated_caption = torch.stack(is_generated_caption)
                
                # 确保image_id是tensor
                image_ids = [item.get("image_id", -1) for item in valid_batch]
                image_ids = torch.tensor(image_ids, dtype=torch.long)

                idx = torch.tensor([item["idx"] for item in valid_batch], dtype=torch.long)

                return {
                    "image": images,
                    "text": texts,
                    "label": labels,
                    "image_id": image_ids,
                    "is_generated_caption": is_generated_caption,
                    "idx": idx,
                    "has_caption": has_captions
                }
            
            # 合并验证集数据集
            val_dataset = torch.utils.data.ConcatDataset([val_aligned_dataset, val_non_aligned_dataset])

            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # 使用0避免Windows上的多进程问题
                pin_memory=True,
                collate_fn=custom_collate_fn
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # 使用0避免Windows上的多进程问题
                pin_memory=True,
                collate_fn=custom_collate_fn
            )

            return train_loader, val_loader
            
        # 合并对齐和非对齐数据
        train_data = client_data["train"]["aligned"] + client_data["train"]["non_aligned"]
        val_data = client_data["val"]["aligned"] + client_data["val"]["non_aligned"]

        # 如果非对齐数据不聚合 exp: baseline2、4

        ###创建训练集的数据集
        train_aligned_dataset = CUBDataset(
            client_data["train"]["aligned"],
            Config.TRAIN_IMAGE_DIR,
            Config.TRAIN_ANNOTATION_FILE,
            text_only=text_only,
            use_generated_images=use_generated_images,
            use_generated_captions=use_generated_captions,
            aggregation=aggregation,
            image_id_to_path = self.image_id_to_path
                                                     )

        train_non_aligned_dataset = CUBDataset(
            client_data["train"]["non_aligned"],
            Config.TRAIN_IMAGE_DIR,
            Config.TRAIN_ANNOTATION_FILE,
            text_only=text_only,
            use_generated_images=use_generated_images and for_non_aligned,
            use_generated_captions=use_generated_captions and for_non_aligned,
            aggregation=aggregation and not for_local_non_aligned,
            image_id_to_path = self.image_id_to_path
                                                     )

        ###创建验证集的数据集
        val_aligned_dataset = CUBDataset(
            client_data["val"]["aligned"],
            Config.TRAIN_IMAGE_DIR,
            Config.TRAIN_ANNOTATION_FILE,
            text_only=text_only,
            use_generated_images=use_generated_images,
            use_generated_captions=use_generated_captions,
            aggregation=aggregation,
            image_id_to_path = self.image_id_to_path
                                                     )
        
        val_non_aligned_dataset = CUBDataset(
            client_data["val"]["non_aligned"],
            Config.TRAIN_IMAGE_DIR,
            Config.TRAIN_ANNOTATION_FILE,
            text_only=text_only,
            use_generated_images=use_generated_images and for_non_aligned,
            use_generated_captions=use_generated_captions and for_non_aligned,
            aggregation=aggregation and not for_local_non_aligned,
            image_id_to_path = self.image_id_to_path
                                                     )


        # # 创建数据集
        # train_dataset = COCOTextImageDataset(
        #     train_data,
        #     Config.TRAIN_IMAGE_DIR,
        #     Config.TRAIN_ANNOTATION_FILE,
        #     text_only=text_only,
        #     use_generated_images=use_generated_images,
        #     use_generated_captions=use_generated_captions
        # )

        # val_dataset = COCOTextImageDataset(
        #     val_data,
        #     Config.VAL_IMAGE_DIR,
        #     Config.VAL_ANNOTATION_FILE,
        #     text_only=text_only,
        #     use_generated_images=use_generated_images,
        #     use_generated_captions=use_generated_captions
        # )

        # 自定义collate函数，确保批次中的所有样本都有有效的字段
        def custom_collate_fn(batch):
            # 筛选有效样本
            valid_batch = [item for item in batch if item is not None and "label" in item and item["label"] is not None]

            if len(valid_batch) == 0:
                # 如果没有有效样本，返回一个空的批次（这不应该发生，因为我们已经验证了数据）
                raise ValueError("批次中没有有效样本，这不应该发生，请检查数据处理流程")

            # 对于有效样本，进行标准的批次收集
            images = torch.stack([item["image"] for item in valid_batch])
            texts = [item["text"] for item in valid_batch]

            # 确保label是tensor
            labels = [item["label"] if isinstance(item["label"], torch.Tensor)
                      else torch.tensor(item["label"], dtype=torch.long) for item in valid_batch]
            labels = torch.stack(labels)

            # 确保has_caption是tensor
            has_captions = [
                item.get("has_caption", torch.tensor(False)) if isinstance(item.get("has_caption", False), torch.Tensor)
                else torch.tensor(item.get("has_caption", False), dtype=torch.bool) for item in valid_batch]
            has_captions = torch.stack(has_captions)

            # 确保is_generated_caption是tensor
            is_generated_caption = [
                item.get("is_generated_caption", torch.tensor(False)) if isinstance(item.get("is_generated_caption", False),
                                                                            torch.Tensor)
                else torch.tensor(item.get("is_generated_caption", False), dtype=torch.bool) for item in valid_batch]
            is_generated_caption = torch.stack(is_generated_caption)
            
            # 确保image_id是tensor
            image_ids = [item.get("image_id", -1) for item in valid_batch]
            image_ids = torch.tensor(image_ids, dtype=torch.long)

            idx = torch.tensor([item["idx"] for item in valid_batch], dtype=torch.long)

            return {
                "image": images,
                "text": texts,
                "label": labels,
                "image_id": image_ids,
                "is_generated_caption": is_generated_caption,
                "idx": idx,
                "has_caption": has_captions
            }

        # 合并训练集数据集，使用ConcatDataset
        train_dataset = torch.utils.data.ConcatDataset([train_aligned_dataset, train_non_aligned_dataset])
        
        # 合并验证集数据集
        val_dataset = torch.utils.data.ConcatDataset([val_aligned_dataset, val_non_aligned_dataset])

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # 使用0避免Windows上的多进程问题
            pin_memory=True,
            collate_fn=custom_collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # 使用0避免Windows上的多进程问题
            pin_memory=True,
            collate_fn=custom_collate_fn
        )

        return train_loader, val_loader
    
    def get_category_names(self):
        """
        获取COCO类别ID到类别名称的映射

        Returns:
            类别名称字典 {category_id: category_name}
        """
        try:
            # 加载训练实例标注
            train_instances = self._load_annotations(Config.INSTANCE_TRAIN_ANNOTATION_FILE)

            # 创建类别映射：内部ID -> 类别名称
            category_names = {}
            for category in train_instances['categories']:
                if category['id'] in self.category_map:
                    internal_id = self.category_map[category['id']]
                    category_names[internal_id] = category['name']

            if self.logger:
                self.logger.info(f"加载了 {len(category_names)} 个类别名称")

            return category_names
        except Exception as e:
            if self.logger:
                self.logger.error(f"获取类别名称失败: {e}")
            # 返回默认类别名称
            return {i: f"category_{i}" for i in range(len(self.category_map))}

    def get_prototype(self, NUM_PROTOTYPE = Config.NUM_PROTOTYPE, 
                    client_features = None, client_labels = None, client_image_id_list = None,
                    server_features = None, server_labels = None, server_image_id_list = None):
        """
            若有对齐数据 则使用client_features(labels)_list 与 server_features(labels) 进行融合 聚类 生成超原型
            返回为 超原型
        """
        if client_features == None or server_features == None:
            raise ValueError("请传入客户端与主动方对齐数据的embedding")

        self.logger.info("进行特征融合")
        # 收集服务端数据
        server_data_dict = {}
        for i, img_id in enumerate(server_image_id_list):
            img_id = int(img_id)
            server_data_dict[img_id] = {
                'feature': server_features[i],
                'label': server_labels[i]
            }
        
        all_data_dict = {}
        
        # 收集客户端数据
        for i, img_id in enumerate(client_image_id_list):
            img_id = int(img_id)
            if img_id not in all_data_dict:
                all_data_dict[img_id] = {
                    'features': [],
                    'labels': []
                }
            all_data_dict[img_id]['features'].append(client_features[i])
            all_data_dict[img_id]['labels'].append(client_labels[i])
        
        # 添加服务端特征
        for img_id, server_data in server_data_dict.items():
            img_id = int(img_id)
            if img_id in all_data_dict:  # 只处理客户端也有的image_id
                all_data_dict[img_id]['features'].append(server_data['feature'])
                all_data_dict[img_id]['labels'].append(server_data['label'])
        
        # 存储融合后的特征和标签
        fused_features = []
        fused_labels = []
        fused_image_id = []
        
        # 对每个image_id的所有特征（客户端+服务端）进行平均融合
        for img_id, data in all_data_dict.items():
            if img_id in server_data_dict and img_id != -1:  # 确保该image_id在服务端也存在

                all_features = torch.stack(data['features'])  # shape: [3, feature_dim]
                fused_feature = all_features.mean(dim=0)  

                fused_image_id.append(img_id)
                fused_features.append(fused_feature)
                fused_labels.append(data['labels'][0])  # 标签取第一个
        
        # 转换为tensor
        if len(fused_features) > 0:
            fused_features = torch.stack(fused_features)
            fused_labels = torch.tensor(fused_labels) if not torch.is_tensor(fused_labels[0]) else torch.stack(fused_labels)
            
            self.logger.info(f"特征融合完成，融合了 {len(fused_features)} 个特征")
    
        self.logger.info("进行超原型生成")

        # 转换为numpy
        features_np = fused_features.detach().cpu().numpy() if torch.is_tensor(fused_features) else np.array(fused_features)
        
        # 进行聚类
        kmeans = KMeans(n_clusters=NUM_PROTOTYPE, random_state=Config.SEED, n_init=10)

        cluster_assignments = kmeans.fit_predict(features_np) #返回的是ndarray of shape (n_samples,)

        clustered_data = {}
        for i in range(NUM_PROTOTYPE):
            clustered_data[i] = {
                'features': [],
                'labels': [],
                'image_ids': []
            }

        # 将每个特征分配到对应的聚类中
        for i, cluster_id in enumerate(cluster_assignments):
            clustered_data[cluster_id]['features'].append(fused_features[i])
            clustered_data[cluster_id]['labels'].append(fused_labels[i])
            clustered_data[cluster_id]['image_ids'].append(fused_image_id[i])

        # 转换为tensor格式
        for cluster_id in clustered_data:
            if len(clustered_data[cluster_id]['features']) > 0:
                clustered_data[cluster_id]['features'] = torch.stack(clustered_data[cluster_id]['features'])
                clustered_data[cluster_id]['labels'] = torch.tensor(clustered_data[cluster_id]['labels']) if not torch.is_tensor(clustered_data[cluster_id]['labels'][0]) else torch.stack(clustered_data[cluster_id]['labels'])

        # for i, data in clustered_data.items():
        #     self.logger.info(f"聚类 {i}: {len(data['features'])} 个特征")
        Prototype = defaultdict(list)
        for i, data in clustered_data.items():
            if len(data['features']) > 1:
                features = data['features'].mean(dim=0)
            else:
                features = data["features"][0]
            Prototype[i].append(features)

        self.logger.info(f"超原型生成完成,总数为{NUM_PROTOTYPE}")
        return Prototype, cluster_assignments


    
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


    def get_sim_matrix(self, prototype=None, client_data=None, for_server=False, zero_aligned = False):
        if prototype is None or client_data is None:
            raise ValueError("请传入prototype与client_data")
        
        if for_server:
            non_aligned = client_data
        else:
            non_aligned = client_data[0]  

        # 构建原型张量
        if zero_aligned:
            prototype_tensor = torch.stack(prototype, dim=0)
        else:
            prototype_tensor = torch.cat([data.unsqueeze(0) for data in prototype], dim=0)

        # 计算距离矩阵
        distance_matrix = self.compute_ot_distance(non_aligned, prototype_tensor)

        return distance_matrix

    def match_aligned_data(self, sim_matrix, distance_threshold=0.0005):
        """
        返回格式为[(i, j, k, label),]
        """
        non_aligned_data = defaultdict(list)
        for client_id in range(Config.NUM_CLIENTS+1):
            non_aligned_data[client_id] = self.client_data[client_id]["train"]["non_aligned"]
    
        aligned_pairs = []
        label_pairs = []
        # 第一次为空所以要初始化

        label_to_indices = [defaultdict(list) for _ in range(3)]

        for client_id in range(3):
            for idx in range(Config.NON_ALIGNED_DATA_SIZE_CLIENT):
                label = non_aligned_data[client_id][idx]["category_id"]
                label_to_indices[client_id][label].append(idx)

        common_labels = set(label_to_indices[0].keys()) & set(label_to_indices[1].keys()) & set(label_to_indices[2].keys())

        for label in common_labels:
            indices_0 = label_to_indices[0][label]
            indices_1 = label_to_indices[1][label]
            indices_2 = label_to_indices[2][label]
            
            for i in indices_0:
                for j in indices_1:
                    for k in indices_2:
                        label_pairs.append((i, j, k))
        for i, j, k in label_pairs:
            if abs(sim_matrix[0][i] - sim_matrix[1][j]) < distance_threshold and abs(sim_matrix[0][i] - sim_matrix[2][k]) < distance_threshold and abs(sim_matrix[1][j] - sim_matrix[2][k]) < distance_threshold:
                aligned_pairs.append((i,j,k))
            continue
        self.logger.info(f"找到 {len(aligned_pairs)} 个匹配对")
        return aligned_pairs
    
    def match_aligned_data_new(self, sim_matrix, distance_threshold=0.0008):
        non_aligned_data = defaultdict(list)
        for client_id in range(Config.NUM_CLIENTS+1):
            non_aligned_data[client_id] = self.client_data[client_id]["train"]["non_aligned"]
    

        aligned_pairs = []
        label_pairs = []

        label_to_indices = [defaultdict(list) for _ in range(3)]

        for client_id in range(3):
            for idx in range(Config.NON_ALIGNED_DATA_SIZE_CLIENT):
                label = non_aligned_data[client_id][idx]["category_id"]
                label_to_indices[client_id][label].append(idx)

        common_labels = set(label_to_indices[0].keys()) & set(label_to_indices[1].keys()) & set(label_to_indices[2].keys())

        for label in common_labels:
            if int(label) == -1:
                continue
            indices_0 = label_to_indices[0][label]
            indices_1 = label_to_indices[1][label]
            indices_2 = label_to_indices[2][label]
            
            for i in indices_0:
                for j in indices_1:
                    for k in indices_2:
                        label_pairs.append((i, j, k, label))


        for i, j, k, label in label_pairs:
            if abs(sim_matrix[0][i] - sim_matrix[1][j]) < distance_threshold and abs(sim_matrix[0][i] - sim_matrix[2][k]) < distance_threshold and abs(sim_matrix[1][j] - sim_matrix[2][k]) < distance_threshold:
                aligned_pairs.append((i,j,k,label))
            continue
        self.logger.info(f"找到 {len(aligned_pairs)} 个匹配对")
        return aligned_pairs
    
    def append_aligned_data(self,aligned_pairs, attention_mask = None,
                            prototypes=None, client_features = None, server_features = None, 
                            text_prototypes=None, text_client_features=None, text_server_features=None):
        self.logger.info("正在使用伪对齐数据补充对齐数据")
        selected_aligned_pairs = []
        if attention_mask:
            for i, j, k, label in aligned_pairs:
                if (i, j, k, label) in attention_mask:
                    continue
                selected_aligned_pairs.append((i,j,k,label))
        else:
            attention_mask = []
            selected_aligned_pairs = aligned_pairs
        for i, j, k in selected_aligned_pairs:
            self.client_data[0]["train"]["aligned"].append(self.client_data[0]["train"]["non_aligned"][i])
            self.client_data[1]["train"]["aligned"].append(self.client_data[1]["train"]["non_aligned"][j])
            self.client_data[2]["train"]["aligned"].append(self.client_data[2]["train"]["non_aligned"][k])
        self.logger.info(f"扩充后客户端对齐数据个数为{len(self.client_data[0]["train"]["aligned"])},共增加{len(selected_aligned_pairs)}个")
        # if len(prototypes) < Config.NUM_PROTOTYPE:
        #     self.logger.info(f"当前超原型个数为{len(prototypes)},进行超原型扩充")
        #     for i, j, k in selected_aligned_pairs:
        #         match_prototype =[] 
        #         match_prototype.append(client_features[0][0][i])
        #         match_prototype.append(client_features[1][0][j])
        #         match_prototype.append(server_features[k])
        #         match_prototype = torch.stack(match_prototype, dim=0).mean(dim=0)
        #         #[4096]
        #         match_prototype_tensor = match_prototype.unsqueeze(0)
        #         #[1, 4096]
        #         prototype_tensor = torch.stack(prototypes, dim=0)
        #         sim = self.compute_ot_distance(source=match_prototype_tensor, target=prototype_tensor).item()

        #         text_match_prototype =[]
        #         text_match_prototype.append(text_client_features[0][0][i])
        #         text_match_prototype.append(text_client_features[1][0][j])
        #         text_match_prototype.append(text_server_features[k])
        #         text_match_prototype = torch.stack(text_match_prototype, dim=0).mean(dim=0)

        #         if sim < Config.SIM_INDEX:
        #             prototypes.append(match_prototype)
        #             text_prototypes.append(text_match_prototype)
        #             attention_mask.append((i, j, k))

                
        #         if len(prototypes) == Config.NUM_PROTOTYPE:
        #             break
        #     self.logger.info(f"超原型扩充完成,扩充后个数为{Config.NUM_PROTOTYPE}")
        return attention_mask
    
    def append_aligned_data_new(self,aligned_pairs, attention_mask = None,
                            prototypes=None, attention_mask_aligned=None, client_features = None, server_features = None, 
                            text_prototypes=None, text_client_features=None, text_server_features=None, 
                            prototypes_labels=None, text_prototypes_labels = None):
        self.logger.info("正在使用伪对齐数据补充对齐数据")
        selected_aligned_pairs = []
        if attention_mask_aligned: #之前增加的对齐数据现在过滤掉
            for i, j, k, label in aligned_pairs:
                if int(label) == -1:
                    continue
                if (i, j, k, label) in attention_mask_aligned:
                    continue
                selected_aligned_pairs.append((i,j,k,label))
                attention_mask_aligned.append((i, j, k, label))
        else:
            attention_mask_aligned = []
            selected_aligned_pairs = aligned_pairs
            for i, j, k, label in selected_aligned_pairs:
                attention_mask_aligned.append((i, j, k, label))#已经增加的过滤掉
        if not attention_mask:
            attention_mask = []

        for i, j, k, label in selected_aligned_pairs:
            self.client_data[0]["train"]["aligned"].append(self.client_data[0]["train"]["non_aligned"][i])
            self.client_data[1]["train"]["aligned"].append(self.client_data[1]["train"]["non_aligned"][j])
            self.client_data[2]["train"]["aligned"].append(self.client_data[2]["train"]["non_aligned"][k])
        self.logger.info(f"扩充后客户端对齐数据个数为{len(self.client_data[0]["train"]["aligned"])},共增加{len(selected_aligned_pairs)}个")
        if len(prototypes) < Config.NUM_PROTOTYPE:
            self.logger.info(f"当前超原型个数为{len(prototypes)},进行超原型扩充")
            for i, j, k, label in selected_aligned_pairs: 
                if text_client_features[0][0][i].any() and text_client_features[1][0][j].any() and text_server_features[k].any():
                    match_prototype =[] 
                    text_match_prototype =[]
                    match_prototype.append(client_features[0][0][i])
                    match_prototype.append(client_features[1][0][j])
                    match_prototype.append(server_features[k])
                    match_prototype = torch.stack(match_prototype, dim=0).mean(dim=0)
                    #[4096]
                    match_prototype_tensor = match_prototype.unsqueeze(0)
                    #[1, 4096]
                    prototype_tensor = torch.stack(prototypes, dim=0)
                    sim = self.compute_ot_distance(source=match_prototype_tensor, target=prototype_tensor).item()

                    text_match_prototype.append(text_client_features[0][0][i])
                    text_match_prototype.append(text_client_features[1][0][j])
                    text_match_prototype.append(text_server_features[k])
                    text_match_prototype = torch.stack(text_match_prototype, dim=0).mean(dim=0)

                    if sim < Config.SIM_INDEX:
                        prototypes.append(match_prototype)
                        prototypes_labels.append(label)
                        text_prototypes.append(text_match_prototype)
                        text_prototypes_labels.append(label)
                        attention_mask.append((i, j, k, label))
                    
                    if len(prototypes) == Config.NUM_PROTOTYPE:
                        break
            self.logger.info(f"超原型扩充完成,扩充后个数为{len(prototypes)}")
        return attention_mask, attention_mask_aligned

    
    def get_class_prototype(self, client_id=None, embeddings=None, labels=None, image_ids=None, for_server=False):
        if client_id:
            self.logger.info(f"正在为客户端{client_id}生成类别原型")
        else:
            self.logger.info(f"正在为服务端(主动方)生成类别原型")

        feature_dict = defaultdict(list)
        image_dict = defaultdict(list)

        for i, label in enumerate(labels):
            if int(label) == -1:
                continue
            label = int(label)
            feature_dict[label].append(embeddings[i])
            image_dict[label].append(image_ids[i])

        class_prototype=defaultdict(list)

        for i, features in feature_dict.items():
            if features:
                fused_features = torch.stack(features, dim=0).mean(dim=0)
                class_prototype[i] = fused_features

        return class_prototype
    
    def get_prototype_from_cp(self, class_prototypes = None):
        if class_prototypes == None:
            raise ValueError("请输入类别原型")
        prototypes_total = defaultdict(list)
        for client_id in class_prototypes:
            class_prototype = class_prototypes[client_id]
            for label, feature in class_prototype.items():
                prototypes_total[label].append(feature)
        prototypes = []
        for feature in prototypes_total.values():
            prototypes.append(torch.stack(feature,dim=0).mean(dim=0))
        return prototypes
    
    def get_prototype_from_cp_new(self, class_prototypes = None):
        if class_prototypes == None:
            raise ValueError("请输入类别原型")
        prototypes_total = defaultdict(list)
        for client_id in class_prototypes:
            class_prototype = class_prototypes[client_id]
            for label, feature in class_prototype.items():
                prototypes_total[label].append(feature)
        prototypes = []
        labels = []
        for label, feature in prototypes_total.items():
            prototypes.append(torch.stack(feature,dim=0).mean(dim=0))
            labels.append(label)
        return prototypes, labels
    
    def get_class_prototype(self, client_id=None, embeddings=None, labels=None, image_ids=None, for_server=False):
        if client_id != None:
            self.logger.info(f"正在为客户端{client_id}生成类别原型")
        else:
            self.logger.info(f"正在为服务端(主动方)生成类别原型")

        feature_dict = defaultdict(list)
        image_dict = defaultdict(list)

        for i, label in enumerate(labels):
            if int(label) == -1:
                continue
            label = int(label)
            feature_dict[label].append(embeddings[i])
            image_dict[label].append(image_ids[i])

        class_prototype=defaultdict(list)

        for i, features in feature_dict.items():
            if features:
                fused_features = torch.stack(features, dim=0).mean(dim=0)
                class_prototype[i] = fused_features

        return class_prototype
    
    def get_class_logits(self, client_id=None, text_logits=None, labels=None, image_ids=None, for_server=False):
        if client_id != None:
            self.logger.info(f"正在为客户端{client_id}生成类别原型匹配的logits")
        else:
            self.logger.info(f"正在为服务端(主动方)生成类别原型匹配的logits")

        logits_dict = defaultdict(list)
        image_dict = defaultdict(list)

        for i, label in enumerate(labels):
            if int(label) == -1:
                continue
            label = int(label)
            logits_dict[label].append(text_logits[i][0])
            image_dict[label].append(image_ids[i])

        text_class_prototype_logits=defaultdict(list)

        for i, features in logits_dict.items():
            if features:
                fused_logits = torch.stack(features, dim=0).mean(dim=0)
                text_class_prototype_logits[i] = fused_logits

        return text_class_prototype_logits
    
    def get_prototype_logits_from_cp(self, class_prototypes_logits = None):
        if class_prototypes_logits == None:
            raise ValueError("请输入类别原型对应logits")
        prototypes_total_logits = defaultdict(list)
        for client_id in class_prototypes_logits:
            class_prototype = class_prototypes_logits[client_id]
            for label, feature in class_prototype.items():
                prototypes_total_logits[label].append(feature)
        prototypes_logits = []
        for feature in prototypes_total_logits.values():
            prototypes_logits.append(torch.stack(feature,dim=0).mean(dim=0))
        return prototypes_logits
    
    def get_class_embeddings(self, client_id=None, text_embeddings=None, labels=None, image_ids=None, for_server=False):
        if client_id:
            self.logger.info(f"正在为客户端{client_id}生成文本类别原型embedding")
        else:
            self.logger.info(f"正在为服务端(主动方)生成文本类别原型embedding")

        feature_dict = defaultdict(list)
        image_dict = defaultdict(list)

        for i, label in enumerate(labels):
            if int(label) == -1:
                continue
            label = int(label)
            feature_dict[label].append(text_embeddings[i])
            image_dict[label].append(image_ids[i])

        text_class_prototypes_embeddding=defaultdict(list)

        for i, features in feature_dict.items():
            if features:
                fused_features = torch.stack(features, dim=0).mean(dim=0)
                text_class_prototypes_embeddding[i] = fused_features

        return text_class_prototypes_embeddding