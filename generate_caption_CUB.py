import os
import argparse
import torch
from tqdm import tqdm
from config_cub import Config
from CUB_DataProcessor2 import DataProcessor
from text_generator_CUB import CaptionGenerator

### 为不同场景生成时要先更换 data_processor_scenario{number} 且清空outputs

def generate_missing_captions(client_data, client_id=None, seed=42, aggregation=False):
    """
    为缺失描述的样本生成文本描述(适用场景为scenario1、2)

    Args:
        client_id: 指定客户端ID，为None时处理所有客户端
        client_data: 已经分配好的客户端数据
    """
    # 初始化数据处理器
    processor = DataProcessor()

    # 初始化描述生成器
    print("初始化描述生成器...")
    generator = CaptionGenerator(device=Config.DEVICE)

    # 处理特定客户端或所有客户端
    client_ids = [client_id] if client_id is not None else range(Config.NUM_CLIENTS+1)

    for cid in client_ids:
        if cid not in client_data:
            print(f"警告: 客户端ID {cid} 不存在，已跳过")
            continue

        print(f"\n处理客户端 {cid} 的数据...")

        # 遍历训练和验证数据
        for split_name, split_key in [("训练", "train"), ("验证", "val")]:
            image_dir = Config.TRAIN_IMAGE_DIR if split_key == "train" else Config.VAL_IMAGE_DIR
            
            # 遍历对齐和非对齐数据
            for align_name, align_key in [("对齐", "aligned"), ("非对齐", "non_aligned")]:
                items = client_data[cid][split_key][align_key]

                # 筛选缺失描述的样本
                missing_items = []
                
                for item in items:
                    if ("caption" not in item or 
                        item["caption"] is None or 
                        item["caption"].strip() == ""):
                        missing_items.append(item)

                if missing_items:
                    print(f"为客户端 {cid} 的{split_name}集{align_name}数据生成 {len(missing_items)} 个描述...")
                    
                    # 生成描述并保存到文件
                    generator.generate_batch(
                        items=missing_items, 
                        client_id=cid, 
                        image_dir=image_dir,
                        aggregation=aggregation
                    )
                else:
                    print(f"客户端 {cid} 的{split_name}集{align_name}数据没有缺失描述。")

    print("\n描述生成完成!")



if __name__ == "__main__":
    """
    用于生成描述,其中aggregation为是否聚合
    如果用于baseline2或4, aggregation为True, 若用于baseline5, aggregation为False
    """
    parser = argparse.ArgumentParser(description="为缺失描述的样本生成文本描述")
    parser.add_argument("--client_id", type=int, help="指定客户端ID，不指定则处理所有客户端")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    aggregation = True
    # 设置torch随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    generate_missing_captions(client_id=args.client_id, seed=args.seed, aggregation=aggregation)