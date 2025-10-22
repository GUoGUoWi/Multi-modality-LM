import os
import random
import numpy as np
import torch
import json
import time
from functools import wraps
from config import Config
from PIL import Image


def set_seed(seed=None):
    """
    设置随机种子，确保实验可重复性

    Args:
        seed: 随机种子，如果为None则使用Config中的SEED
    """
    if seed is None:
        seed = Config.SEED

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 为了更好的可重复性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed


def create_experiment_dir(exp_name=None):
    """
    创建实验目录

    Args:
        exp_name: 实验名称，如果为None则使用时间戳

    Returns:
        实验目录路径
    """
    if exp_name is None:
        exp_name = f"exp_{time.strftime('%Y%m%d_%H%M%S')}"

    exp_dir = os.path.join(Config.OUTPUT_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 创建子目录
    models_dir = os.path.join(exp_dir, "models")
    logs_dir = os.path.join(exp_dir, "logs")
    results_dir = os.path.join(exp_dir, "results")

    for directory in [models_dir, logs_dir, results_dir]:
        os.makedirs(directory, exist_ok=True)

    return exp_dir


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    将归一化的图像张量反归一化为原始像素值范围

    Args:
        tensor: 归一化的图像张量，形状为[C, H, W]
        mean: 归一化时使用的均值
        std: 归一化时使用的标准差

    Returns:
        反归一化后的图像张量，值在[0, 1]之间
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)

    # 确保tensor是一个三维张量 [C, H, W]
    if len(tensor.shape) == 4:
        tensor = tensor[0]

    # 反归一化
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    tensor = tensor.clone()
    tensor = tensor * std + mean

    # 将值限制在[0, 1]范围内
    tensor = torch.clamp(tensor, 0, 1)

    return tensor


def tensor_to_pil(tensor):
    """
    将图像张量转换为PIL图像

    Args:
        tensor: 图像张量，值应在[0, 1]范围内

    Returns:
        PIL图像
    """
    if not isinstance(tensor, torch.Tensor):
        return None

    # 确保值在[0, 1]范围内
    tensor = torch.clamp(tensor, 0, 1)

    # 转换为numpy数组，并调整为[0, 255]范围
    if tensor.dim() == 3:
        # [C, H, W] -> [H, W, C]
        img_np = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np)
    elif tensor.dim() == 2:
        # 灰度图像
        img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np, mode='L')
    else:
        return None


def save_config(config_dict, filepath):
    """
    保存配置到文件

    Args:
        config_dict: 配置字典
        filepath: 保存路径
    """
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config(filepath):
    """
    从文件加载配置

    Args:
        filepath: 配置文件路径

    Returns:
        配置字典
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def timer(func):
    """
    函数执行时间计时装饰器

    Args:
        func: 要计时的函数

    Returns:
        包装后的函数
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to run")
        return result

    return wrapper


def calculate_accuracy(outputs, targets):
    """
    计算分类准确率

    Args:
        outputs: 模型输出
        targets: 真实标签

    Returns:
        准确率
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy


def get_device_stats():
    """
    获取设备信息和状态

    Returns:
        设备信息字典
    """
    device_info = {
        "device": str(Config.DEVICE),
        "cuda_available": torch.cuda.is_available()
    }

    if torch.cuda.is_available():
        device_info.update({
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_current_device": torch.cuda.current_device(),
            "cuda_device_name": torch.cuda.get_device_name(0)
        })

    return device_info


def check_data_dirs():
    """
    检查COCO数据集目录是否存在

    Returns:
        布尔值，表示是否所有必要的数据目录都存在
    """
    required_dirs = [
        Config.DATA_DIR,
        Config.TRAIN_IMAGE_DIR,
        Config.VAL_IMAGE_DIR
    ]

    required_files = [
        Config.TRAIN_ANNOTATION_FILE,
        Config.VAL_ANNOTATION_FILE,
        Config.INSTANCE_TRAIN_ANNOTATION_FILE,
        Config.INSTANCE_VAL_ANNOTATION_FILE
    ]

    all_exists = True

    for directory in required_dirs:
        if not os.path.isdir(directory):
            print(f"Directory not found: {directory}")
            all_exists = False

    for file in required_files:
        if not os.path.isfile(file):
            print(f"File not found: {file}")
            all_exists = False

    return all_exists