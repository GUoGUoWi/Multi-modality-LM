import os
import torch
import hashlib
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image
from config import Config
import torch.nn.functional as F


class ImageGenerator:
    """使用Stable Diffusion XL模型生成图像的工具类"""

    def __init__(self, device=None, model_id="stabilityai/stable-diffusion-xl-base-1.0"):
        """
        初始化图像生成器

        Args:
            device: 计算设备，默认使用Config中配置的设备
            model_id: Stable Diffusion XL模型ID
        """
        self.device = device if device is not None else Config.DEVICE

        # 判断是否使用半精度浮点数以节省显存
        use_fp16 = self.device.type == 'cuda'

        # 创建模型缓存目录
        sd_cache_dir = os.path.join(Config.MODEL_CACHE_DIR, "stable-diffusion-xl")
        os.makedirs(sd_cache_dir, exist_ok=True)

        # 加载模型
        print(f"正在加载Stable Diffusion XL模型: {model_id}")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            sd_cache_dir,
            local_files_only=True,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
            use_safetensors=True,
        )

        # 使用更快的调度器
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        print("成功加载模型")

        # 禁用进度条
        self.pipe.set_progress_bar_config(disable=True)

        # 启用内存优化
        if use_fp16:
            self.pipe.enable_vae_tiling()  # 减少内存使用
            self.pipe.enable_model_cpu_offload()  # 在需要时将模型部分卸载到CPU
        else:
            self.pipe = self.pipe.to(self.device)

        # 创建输出目录
        self.output_dir = os.path.join(Config.OUTPUT_DIR, "generated_images")
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"图像将保存到: {self.output_dir}")

        # 用于训练
        self.is_training_ready = False
        self.optimizer = None
        self.unet = None
        self.vae = None
        self.text_encoder = None

    def enhance_prompt(self, caption):
        """
        增强提示词以提高生成质量

        Args:
            caption: 原始描述文本

        Returns:
            增强后的提示词
        """
        # 改进提示词以生成更真实、更符合COCO数据集风格的图像
        enhanced = (
            f"{caption}, realistic photo, natural lighting, "
            f"high detail, sharp focus, COCO dataset style, "
            f"clear background"
        )
        return enhanced

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

    def get_client_dir(self, client_id):
        """
        获取客户端图像目录

        Args:
            client_id: 客户端ID

        Returns:
            客户端图像目录路径
        """
        client_dir = os.path.join(self.output_dir, f"client_{client_id}")
        os.makedirs(client_dir, exist_ok=True)
        return client_dir

    def generate_image(self, caption, client_id=None, image_id=None, seed=None, num_steps=30, guidance_scale=7.5, aggregation=False):
        """
        根据描述生成图像

        Args:
            caption: 描述文本
            client_id: 客户端ID,用于确定存储目录
            image_id: 图像ID,用于备用文件命名
            seed: 随机种子，确保结果可重现
            num_steps: 生成步数，越高质量越好但越慢
            guidance_scale: 提示词引导强度

        Returns:
            生成的图像路径
        """
        # 确定保存目录
        save_dir = self.output_dir
        if client_id is not None and not aggregation:
            save_dir = self.get_client_dir(client_id)
        else:
            save_dir = os.path.join(save_dir, "aggregation")
            os.makedirs(save_dir, exist_ok=True)

        # 创建安全的文件名
        safe_caption = self.sanitize_filename(caption)
        filename = f"{safe_caption}.png"

        # 如果文件名已存在，添加一个唯一标识符
        if os.path.exists(os.path.join(save_dir, filename)):
            # 使用image_id或哈希值作为唯一标识符
            identifier = image_id if image_id is not None else hashlib.md5(caption.encode('utf-8')).hexdigest()[:8]
            filename = f"{safe_caption}_{identifier}.png"

        save_path = os.path.join(save_dir, filename)

        # 如果图像已存在，直接返回路径
        if os.path.exists(save_path):
            return save_path

        # 增强提示词
        enhanced_prompt = self.enhance_prompt(caption)

        # 设置随机种子
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        # 生成图像
        image = self.pipe(
            prompt=enhanced_prompt,
            height=512,
            width=512,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        # 保存图像
        image.save(save_path)

        return save_path

    def generate_batch(self, items, client_id=None, seed=42, aggregation=False):
        """
        批量生成图像

        Args:
            items: 数据项列表，每项应包含caption和image_id
            client_id: 客户端ID，用于确定存储目录
            seed: 随机种子基数

        Returns:
            生成的图像路径列表
        """
        paths = []

        for i, item in enumerate(tqdm(items, desc="生成图像")):
            caption = item["caption"]
            image_id = item.get("image_id")

            # 为每个样本使用不同的种子
            item_seed = seed + i if seed is not None else None

            try:
                path = self.generate_image(
                    caption=caption,
                    client_id=client_id,
                    image_id=image_id,
                    seed=item_seed,
                    aggregation=aggregation
                )
                paths.append(path)
            except Exception as e:
                print(f"生成图像失败: {e}")
                paths.append(None)

        return paths
    
    def prepare_training(self, lr=1e-5):
        """准备训练：解封装 pipeline 并创建优化器"""
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder

        self.optimizer = torch.optim.AdamW(self.unet.parameters(), lr=lr)
        self.is_training_ready = True
        print("训练模式已准备完成")

    def train_step(self, caption, target_image, loss_fn=None):
        """
        单步训练
        caption: 文本描述
        target_image: 目标图像 (tensor, [B, 3, H, W], 0~1)
        loss_fn: 自定义的loss函数，如果为None则使用MSE
        """
        if not self.is_training_ready:
            raise RuntimeError("请先调用 prepare_training()")

        # 文本编码
        text_inputs = self.pipe.tokenizer(
            caption, padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]

        # 图像编码到 latent space
        target_latents = self.vae.encode(target_image.to(self.device) * 2 - 1).latent_dist.sample()

        # 添加噪声
        noise = torch.randn_like(target_latents)
        timesteps = torch.randint(0, self.pipe.scheduler.config.num_train_timesteps, (1,), device=self.device)
        noisy_latents = self.pipe.scheduler.add_noise(target_latents, noise, timesteps)

        # 预测噪声
        pred_noise = self.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample

        # Loss
        if loss_fn is None:
            loss = F.mse_loss(pred_noise, noise)
        else:
            loss = loss_fn(pred_noise, noise)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_model(self, save_dir):
        """保存训练后的模型"""
        self.pipe.save_pretrained(save_dir)
        print(f"模型已保存到 {save_dir}")