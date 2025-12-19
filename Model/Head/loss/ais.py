import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def binarize(T, nb_classes):
    return torch.zeros(T.size(0), nb_classes, device=T.device).scatter_(1, T.view(-1, 1), 1)

class Angular_Isotonic_Loss(nn.Module):
    def __init__(self, n_way, lamda=24, mrg=0.2, threshold=0.9, debug=True):
        super(Angular_Isotonic_Loss, self).__init__()
        self.n_way = n_way
        self.lamda = lamda
        self.mrg = mrg
        self.threshold = threshold
        self.epoch = 0
        self.debug = debug

    def forward(self, cos_sim, labels, epoch=None):
        if epoch is not None:
            self.epoch = epoch       

        cos_m = math.cos(self.mrg)
        sin_m = math.sin(self.mrg)

        P_one_hot = binarize(labels, self.n_way)
        N_one_hot = 1 - P_one_hot

        sin = torch.sqrt((1.0 - cos_sim ** 2).clamp(min=1e-8, max=1.0))
        pos_phi = cos_sim * cos_m - sin * sin_m
        pos_phi = torch.where(cos_sim > self.threshold, pos_phi, cos_sim)

        neg_phi = cos_sim * cos_m + sin * sin_m
        neg_phi = torch.where(cos_sim < self.threshold, neg_phi, cos_sim)

        pos_exp = torch.exp(-self.lamda * (pos_phi - self.threshold))
        neg_exp = torch.exp(self.lamda * (neg_phi - self.threshold))

        P_sim_sum = (pos_exp * P_one_hot).sum(dim=1)
        N_sim_sum = (neg_exp * N_one_hot).sum(dim=1)

        pos_term = torch.log1p(P_sim_sum)

        neg_term = torch.log1p(N_sim_sum)
        loss = (pos_term + neg_term).mean()

        if self.debug:
            print(f"Pos log1p: {torch.log1p(P_sim_sum).mean().item():.3f}  Neg log1p: {torch.log1p(N_sim_sum).mean().item():.3f}")

        return loss
