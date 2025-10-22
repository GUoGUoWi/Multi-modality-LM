import os
import argparse
import torch
from tqdm import tqdm
from config import Config
from data_processor import DataProcessor
from image_generator import ImageGenerator


def generate_missing_images(client_data, client_id=None, seed=42, aggregation=False, generator = None):
    """
    为缺失图像的样本生成图像(适用场景为scenario1、3)

    Args:
        client_id: 指定客户端ID,为None时处理所有客户端
        seed: 随机种子
    """
    # 初始化数据处理器
    processor = DataProcessor()

    if not generator:
        # 初始化图像生成器
        print("初始化图像生成器...")
        generator = ImageGenerator(device=Config.DEVICE)

    # 处理特定客户端或所有客户端
    client_ids = [client_id] if client_id is not None else range(Config.NUM_CLIENTS)

    for cid in client_ids:
        print(f"\n处理客户端 {cid} 的数据...")

        # 遍历训练和验证数据

        for split_name, split_key in [("训练", "train"), ("验证", "val")]:
            # 遍历对齐和非对齐数据

            for align_name, align_key in [("对齐", "aligned"), ("非对齐", "non_aligned")]:
                items = client_data[cid][split_key][align_key]

                # 筛选缺失图像的样本
                missing_items = [item for item in items if item.get("image_id") is None]

                if missing_items:
                    print(f"为客户端 {cid} 的{split_name}集{align_name}数据生成 {len(missing_items)} 张图像...")
                    generator.generate_batch(
                        missing_items, 
                        client_id=cid, 
                        seed=seed, 
                        aggregation=aggregation
                        )
                else:
                    print(f"客户端 {cid} 的{split_name}集{align_name}数据没有缺失图像。")

    print("\n图像生成完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为缺失图像的样本生成图像")
    parser.add_argument("--client_id", type=int, help="指定客户端ID，不指定则处理所有客户端")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # 设置torch随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    generate_missing_images(client_id=args.client_id, seed=args.seed)