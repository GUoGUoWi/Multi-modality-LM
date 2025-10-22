import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
from config import Config
import hashlib

class CaptionGenerator:
    """使用 LLaVA-7B 模型为图像生成文本描述的工具类"""
    
    def __init__(self, device=None, model_path="/root/autodl-tmp/46_FedMMDG/model_cache/llava-1.5-7b-hf"):
        """
        初始化描述生成器

        Args:
            device: 计算设备，默认使用Config中配置的设备
            model_path: LLaVA 模型本地路径
        """
        self.device = device if device is not None else Config.DEVICE
        use_fp16 = hasattr(self.device, 'type') and self.device.type == 'cuda'
        
        # 创建输出目录
        self.output_dir = os.path.join(Config.OUTPUT_DIR, "generated_captions")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载 LLaVA 模型
        print(f"正在从本地路径加载 LLaVA 模型: {model_path}")
        try:
            model_id = Config.LLAVA_MODEL_ID
            local_dir = os.path.join(Config.MODEL_CACHE_DIR, "llava-1.5-7b-hf")
            self.processor = AutoProcessor.from_pretrained(local_dir, local_files_only=True)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                local_dir,
                local_files_only=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                device_map=None
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            print("成功加载模型")
        except Exception as e:
            print(f"加载 LLaVA 模型失败: {e}")
            raise

    def sanitize_filename(self, text, max_length=100):
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            text = text.replace(char, '')
        if len(text) > max_length:
            text = text[:max_length]
        if not text.strip():
            text = "untitled"
        return text.strip()

    def get_client_dir(self, client_id):
        client_dir = os.path.join(self.output_dir, f"client_{client_id}")
        os.makedirs(client_dir, exist_ok=True)
        return client_dir

    def generate_caption(self, image_path, client_id=None, image_id=None, aggregation=False, max_new_tokens=50):
        # 生成保存目录
        save_dir = self.get_client_dir(client_id) if client_id is not None and not aggregation else os.path.join(self.output_dir, "aggregation")
        os.makedirs(save_dir, exist_ok=True)

        # 生成文件名
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        safe_filename = self.sanitize_filename(base_name)
        filename = f"{safe_filename}.txt"
        if os.path.exists(os.path.join(save_dir, filename)):
            identifier = image_id if image_id is not None else hashlib.md5(image_path.encode('utf-8')).hexdigest()[:8]
            filename = f"{safe_filename}_{identifier}.txt"
        save_path = os.path.join(save_dir, filename)

        # 避免重复生成
        if os.path.exists(save_path):
            return save_path

        try:
            # 加载图像
            image = Image.open(image_path).convert("RGB")
            prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"

            # processor + 移动到模型所在设备
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                padding=True
            )
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(self.model.device)

            # 生成描述
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

            caption = self.processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            # 截取 ASSISTANT 后的内容
            if "ASSISTANT:" in caption:
                caption = caption.split("ASSISTANT:")[-1].strip()

            # 保存
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(caption)

            return save_path
        except Exception as e:
            print(f"为图像 {image_path} 生成描述失败: {e}")
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("无法生成描述")
            return save_path

    def generate_batch(self, items, client_id=None, image_dir=None, aggregation=False):
        paths = []
        for item in tqdm(items, desc="生成图像描述"):
            # 构建图像路径
            if image_dir is not None and "image_id" in item and item["image_id"] is not None:
                # 使用image_dir和image_id构建路径
                image_path = os.path.join(image_dir, f"{item['image_id']:012d}.jpg")
            elif "image_path" in item:
                # 使用提供的完整路径
                image_path = item["image_path"]
            else:
                print(f"无法确定图像路径: {item}")
                paths.append(None)
                continue

            image_id = item.get("image_id")
            try:
                # 调用单图生成方法
                caption_path = self.generate_caption(
                    image_path=image_path,
                    client_id=client_id,
                    image_id=image_id,
                    aggregation=aggregation
                )
                paths.append(caption_path)
            except Exception as e:
                print(f"生成描述失败: {e}")
                paths.append(None)

        return paths


