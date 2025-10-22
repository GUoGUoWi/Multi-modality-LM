import os
import torch
from enum import Enum


class ModelType(Enum):
    """模型类型枚举"""
    CLIP_BASE = "clip-base"
    CLIP_LARGE = "clip-large"
    BLIP_BASE = "blip-base"
    BLIP_LARGE = "blip-large"
    SERVER_MODEL = "server-model"  # 添加服务端模型类型

class CaptionModelType(Enum):
    BLIP_BASE="blip-base"

class Config:
    """配置类，包含所有可调参数"""
    # 路径配置
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    MODEL_CACHE_DIR = os.path.join(ROOT_DIR, "model_cache")
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    # 确保目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    # 数据集配置
    DATA_DIR = os.path.join(ROOT_DIR, "CUB-200", "CUB_200_2011")

    TRAIN_ANNOTATION_FILE = os.path.join(DATA_DIR, "captions", "text_flower")
    TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "images")

    VAL_ANNOTATION_FILE = os.path.join(DATA_DIR, "captions", "text_flower")
    VAL_IMAGE_DIR = os.path.join(DATA_DIR, "images")

    INSTANCE_TRAIN_ANNOTATION_FILE = os.path.join(DATA_DIR, "annotations", "instances_train2017.json")
    INSTANCE_VAL_ANNOTATION_FILE = os.path.join(DATA_DIR, "annotations", "instances_val2017.json")

    # 联邦学习配置
    NUM_CLIENTS = 2  # 参与方数量，可配置，默认为2
    NUM_ROUNDS = 3  # 联邦学习轮数
    PUBLIC_DATA_SIZE = 100  # 公共数据集大小

    # 数据分布配置
    ALIGNED_DATA_SIZE = 1000  # 对齐数据数量 1000
    NON_ALIGNED_DATA_SIZE_CLIENT = 1000
    NON_ALIGNED_DATA_SIZE = 3000 # 非对齐数据总数量 3000
    TEST_RATIO = 0.5  # 测试集比例

    #拼接维度
    CAT_DIM=8192

    # 图像缺失比例
    ALIGNED_IMAGE_MISSING_RATIO = 0.5  # 对齐数据中图像缺失比例p
    NON_ALIGNED_IMAGE_MISSING_RATIO = 0.5  # 非对齐数据中图像缺失比例gamma

    # 文本缺失比例
    ALIGNED_CAPTION_MISSING_RATIO = 0.5
    NON_ALIGNED_CAPTION_MISSING_RATIO = 0.5

    # 超原型个数
    NUM_PROTOTYPE = 100

    # 扩充超原型相似度指数
    SIM_INDEX = 0.15

    # 使用描述准确率
    use_caption_accuracy = 0.4

    # Image_generator训练轮数
    IMAGE_GENERATOR_EPOCHS = 20

    # 模型配置
    CLIENT_MODELS = [ModelType.CLIP_LARGE, ModelType.CLIP_BASE]  # 每个客户端的模型类型
    FREEZE_BACKBONE = False  # 是否冻结主干网络，设为False可以微调预训练模型
    CLIENT_MODELS_CAPTION = [CaptionModelType.BLIP_BASE] #缺文本的客户端模型类型 ？？？


    # 训练配置
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 5e-4  # 提高学习率
    WEIGHT_DECAY = 1e-4  # 添加权重衰减
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 服务端训练配置
    # LLaVA模型配置
    LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"  # 使用7B版本以降低资源需求
    LLAVA_PROMPT_TEMPLATE = "Analyze this image and caption: '{text}'. Extract the key features."
    SERVER_EPOCHS = 20  # 服务端训练轮数
    SERVER_LEARNING_RATE = 5e-5
    SERVER_VALIDATION_INTERVAL = 2 #验证集评估

    # 特征映射函数训练配置
    MAPPER_EPOCHS = 5  # 特征映射函数训练轮数
    MAPPER_LEARNING_RATE = 1e-3  # 特征映射函数学习率
    MAPPER_BATCH_SIZE = 16  # 映射器训练批次大小
    GRADIENT_ACCUMULATION_STEPS = 2  # 梯度累积步数

    # 随机种子
    SEED = 30

    # 验证配置
    VALIDATION_INTERVAL = 1  # 每隔多少个epoch验证一次

    # 数据质量配置
    MAX_CLASSES = 200  # 限制类别数量，减少稀有类别的影响
    MIN_SAMPLES_PER_CLASS = 5  # 每个类别的最小样本数

    @classmethod
    def get_client_model(cls, client_id,forcaption = False):
        """根据客户端ID获取对应的模型类型"""
        if forcaption == True:
            return cls.CLIENT_MODELS_CAPTION[client_id%len(cls.CLIENT_MODELS)] 
        if client_id < len(cls.CLIENT_MODELS):
            return cls.CLIENT_MODELS[client_id]
        else:
            # 如果客户端ID超出配置的模型列表长度，则循环使用
            return cls.CLIENT_MODELS[client_id % len(cls.CLIENT_MODELS)]

    @classmethod
    def update(cls, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)

        # 当NUM_CLIENTS更新时，确保CLIENT_MODELS有足够的元素
        if "NUM_CLIENTS" in kwargs:
            while len(cls.CLIENT_MODELS) < cls.NUM_CLIENTS:
                cls.CLIENT_MODELS.append(cls.CLIENT_MODELS[0])  # 不足时重复使用第一个模型

    #调度器配置
    net_p_q_epoch = 5
    net_p_q_lr = 1e-5
    ss_model = {
        'net_p_weights_path':"/root/autodl-tmp/46_FedMMDG/model_cache/net_p_weights",
        'net_q_weights_path':"/root/autodl-tmp/46_FedMMDG/model_cache/net_q_weights"
    }

    vocabulary_size = 32064
    hidden_dim = 4096

    max_iter = 4

    CUB_ROOT_DIR=   "/root/autodl-tmp/46_FedMMDG/CUB-200/CUB_200_2011"
    CUB_IMAGE_DIR = "/root/autodl-tmp/46_FedMMDG/CUB-200/CUB_200_2011/images"
    CUB_CAPTION_dir = "/root/autodl-tmp/46_FedMMDG/CUB-200/CUB_200_2011/captions/text_flower"