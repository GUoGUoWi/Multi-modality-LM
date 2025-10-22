import torch
import torch.nn.functional as F
from models import get_model
from PIL import Image
import config as Config
import numpy as np

class Loss_function():
    def loss_prefer(self, g_p_logits=None, g_q_logits=None, 
                    ref_logits_preferred=None, ref_logits_non_preferred=None,
                    beta=1.0, modality="image"):
        """
        modality表示模态
        """
        reward_preferred = self.reward_fn(g_p_logits, ref_logits_preferred, modality)
        reward_non_preferred = self.reward_fn(g_q_logits, ref_logits_non_preferred, modality)
        
        
        reward_diff = beta * (reward_preferred - reward_non_preferred)
        reward_result = - F.logsigmoid(reward_diff)
        return reward_result.mean(dim=0), reward_diff

    def reward_fn(self, g, ref, modality = "image"):
        if modality == "image":
            logp_theta = F.log_softmax(g, dim=-1)
            logp_ref = F.log_softmax(ref, dim=-1)
            
            # 对目标 token 取值
            log_ratio = logp_theta - logp_ref
            return log_ratio
        else:
            raise ValueError("请至少选定一种模态来确定reward_fn")
    #加一个文本生成能力

    def loss_special(self, g_q_embedding=None, o_i_embedding=None, sim_score=None):
        """
        o_i 是 i样本里的最近的超原型
        g_i 是 i样本生成的模态信息
        sim 是 i样本tensor和超原型的相似度
        """
        return torch.norm(g_q_embedding-o_i_embedding, p=2)*(1 - sim_score)

#图像 文本   文本B 文本B 文本

    def info_nce_loss(self, x, y, z, temperature=0.07):
        """
        x: [1, dim] 生成向量
        y: [dim] 正样本
        z: list of [dim] 负样本
        """
        # 拼接正负样本
        vectors = [y] + z                 
        vectors = torch.stack(vectors, dim=0)  # [N+1, dim]

        # L2 normalize
        x_norm = F.normalize(x, dim=-1)          # [1, dim]
        vectors_norm = F.normalize(vectors, dim=-1)  # [N+1, dim]

        # 相似度 logits
        logits = torch.matmul(x_norm, vectors_norm.t()) / temperature  # [1, N+1]

        # 正样本在第0个位置
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=x.device)  # [1]

        # 交叉熵
        loss = F.cross_entropy(logits, labels)  
        return loss

