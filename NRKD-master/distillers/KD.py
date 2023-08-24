from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F


class KD(nn.Module):
    """
    Distilling the Knowledge in a Neural Network.
    """
    def __init__(self, T):
        super(KD, self).__init__()
        self.T = T


    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        # 计算KL散度，p_t指导p_s，后者指导前者的训练，p_t真实分布，p_s是预测分布取log值
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T ** 2) / y_s.shape[0]
        return loss


