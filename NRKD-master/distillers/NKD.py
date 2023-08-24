# coding=utf-8
import torch

import torch.nn as nn
import torch.nn.functional as F


class NKDLoss(nn.Module):
    def __init__(self, k=1, choose='angle', lambda1=10.0, lambda2=10.0):
        super(NKDLoss, self).__init__()
        self.k = k
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.choose = choose

    def angle(self, t):
        t = F.normalize(t, dim=1)
        cosine = torch.mm(t, t.T)
        sort, idx = torch.sort(cosine, descending=True)
        return sort, idx

    def nebor_loss(self, fea_s, fea_t, logits_s, logits_t):
        b = logits_t.size(0)
        sort, idx = self.angle(logits_t)
        nebor_idx = idx[:, 1:self.k+1]
        loss_nd = self.lambda2 * self.fea(fea_s, fea_t, b, nebor_idx) + self.lambda1 * self.res(logits_s, logits_t, b, nebor_idx)
        return loss_nd

    def fea(self, fea_s, fea_t, b, nebor_idx):
        loss_nebor_fea = torch.FloatTensor([0]).cuda()
        for f_s, f_t in zip(fea_s, fea_t):
            s_H, t_H = f_s.shape[2], f_t.shape[2]
            if s_H > t_H:
                f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
            elif s_H < t_H:
                f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
            else:
                pass
            c_s = f_s.size(1)
            c_t = f_t.size(1)
            f_s = f_s.view(b, int(c_s), -1)
            f_t = f_t.view(b, int(c_t), -1)
            idx = nebor_idx[:, 0]
            nebor_s_f = torch.index_select(f_s, 0, idx)
            nebor_t_f = torch.index_select(f_t, 0, idx)

            for i in range(1, self.k):
                idx = nebor_idx[:, i]
                n_s_f = torch.index_select(f_s, 0, idx)
                n_t_f = torch.index_select(f_t, 0, idx)
                nebor_s_f = torch.cat((nebor_s_f, n_s_f), 2)
                nebor_t_f = torch.cat((nebor_t_f, n_t_f), 2)

            nebor_s_f = nebor_s_f.view(b, self.k, c_s, -1)
            nebor_t_f = nebor_t_f.view(b, self.k, c_t, -1)
            f_s = torch.unsqueeze(f_s, 1)
            f_t = torch.unsqueeze(f_t, 1)
            fea_s_diff = (f_s - nebor_s_f)
            fea_t_diff = (f_t - nebor_t_f)

            fea_s_diff_spital = F.normalize(fea_s_diff.pow(2).mean(2), p=2, dim=2)  # b k hw
            fea_t_diff_spital = F.normalize(fea_t_diff.pow(2).mean(2), p=2, dim=2)  # b k hw

            loss_nebor_fea += torch.norm(fea_s_diff_spital - fea_t_diff_spital, p=2, dim=2).pow(2).mean()

        return loss_nebor_fea

    def res(self, logits_s, logits_t, b, nebor_idx):
        idx = nebor_idx[:, 0]
        nebor_s_p = torch.index_select(logits_s, 0, idx)
        nebor_t_p = torch.index_select(logits_t, 0, idx)
        for i in range(1, self.k):
            idx = nebor_idx[:, i]
            n_s_p = torch.index_select(logits_s, 0, idx)
            n_t_p = torch.index_select(logits_t, 0, idx)
            nebor_s_p = torch.cat((nebor_s_p, n_s_p), 1)
            nebor_t_p = torch.cat((nebor_t_p, n_t_p), 1)
        nebor_s_p = nebor_s_p.view(b, self.k, -1)
        nebor_t_p = nebor_t_p.view(b, self.k, -1)
        l_s = torch.unsqueeze(logits_s, 1)
        l_t = torch.unsqueeze(logits_t, 1)
        res_s_diff = l_s - nebor_s_p  # b k m
        res_t_diff = l_t - nebor_t_p

        loss_nebor_res = self.js_div(res_s_diff, res_t_diff) / (b * self.k)

        return loss_nebor_res

    def js_div(self, q, p):
        q = F.softmax(q, dim=2)
        p = F.softmax(p, dim=2)

        log_mean = ((q + p) / 2).log()
        loss = (F.kl_div(log_mean, q, reduction='sum') + F.kl_div(log_mean, p, reduction='sum')) / 2
        return loss

    def forward(self, fea_s, fea_t, logits_s, logits_t):
        loss = self.nebor_loss(fea_s, fea_t, logits_s, logits_t)
        # print(loss)
        return loss




