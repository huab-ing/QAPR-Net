import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss.ais import Angular_Isotonic_Loss


class DeltaGate(nn.Module):
    def __init__(self, ratios, feat_dim):
        super().__init__()
        self.ratios = ratios
        self.num_modes = len(ratios)
        self.logits = nn.Parameter(torch.zeros(self.num_modes))
        self.feat_dim = feat_dim

        
    def forward(self, fused_proto, base_proto):
        Q, N, D = fused_proto.shape
        device = fused_proto.device
        
        delta = torch.abs(fused_proto - base_proto)     # [Q, N, D]
        
        max_K = max(int(r * D) for r in self.ratios)
        _, topk_idx = torch.topk(delta, max_K, dim=-1)
        
        masks = []
        for r in self.ratios:
            K = max(1, int(r * D))
            mask = torch.zeros_like(delta, device=device)
            mask.scatter_(-1, topk_idx[..., :K], 1.0)
            masks.append(mask)
        
        # weight_fused
        stacked = torch.stack(masks, dim=0)  # [M, Q, N, D]
        mode_w = F.softmax(self.logits, dim=0).view(-1, 1, 1, 1)
        combined_mask = (stacked * mode_w).sum(dim=0)
        
        return fused_proto * combined_mask


class QSA(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.query_proj = nn.Linear(feat_dim, feat_dim//4)
        self.key_proj = nn.Linear(feat_dim, feat_dim//4)
        
    def forward(self, query, prototypes):

        q_proj = self.query_proj(query).unsqueeze(1)  # [Q, 1, D/4]
        k_proj = self.key_proj(prototypes)          # [Q, N, D/4]

        scale = self.query_proj.out_features ** 0.5
        
        attn_weights = torch.matmul(q_proj, k_proj.transpose(1,2)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)  # [Q, 1, N]
        
        return prototypes * attn_weights.transpose(1,2)


def qpa_loss(query_features, prototypes, labels, margin=0.3):
    cos_sim = F.cosine_similarity(
        query_features.unsqueeze(1),  # [Q, 1, D]
        prototypes, # [Q, N, D]
        dim=-1
    )
    
    pos_mask = F.one_hot(labels, num_classes=prototypes.size(1)).bool()
    neg_mask = ~pos_mask
    
    pos_sim = cos_sim[pos_mask]
    
    neg_sim = cos_sim[neg_mask].view(cos_sim.size(0), -1)
    hardest_neg = torch.topk(neg_sim, k=min(3, neg_sim.size(1)), dim=1)[0].mean(dim=1)
    
    # loss
    loss = F.relu(hardest_neg - pos_sim + margin).mean()
    
    return loss


class QAPR_Net(nn.Module):
    def __init__(self,n_way,k_shot,query,
                delta_ratios: list = [0.25, 0.5, 0.75],
                prj_num: int = 14,
                feat_dim: int = 64
                ):
        super().__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.query = query
        self.prj_num = prj_num
        self.feat_dim = feat_dim

        self.ais_loss = Angular_Isotonic_Loss(n_way=n_way, lamda=32, threshold=0.8)
        
        self.delta_gate = DeltaGate(delta_ratios, feat_dim)
        
        self.cross_attn = QSA(feat_dim)
        
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        self.epoch = 0
        self.ais_loss = Angular_Isotonic_Loss(n_way=n_way, lamda=32, threshold=0.8)
        
        self.delta_gate = DeltaGate(delta_ratios, feat_dim)
        self.cross_attn = QSA(feat_dim)

    def forward(self, feat: torch.Tensor, label):
        if isinstance(label, (list, tuple)):
            label = label[1]
        label = label.to(feat.device)
        total_support = self.n_way * self.k_shot * self.prj_num
        support = feat[:total_support]
        query = feat[total_support:]

        support = F.normalize(support, dim=1)
        query = F.normalize(query, dim=1)
        
        support = support.view(self.n_way, self.k_shot, self.prj_num, self.feat_dim, -1)
        view_proto = support.mean(dim=1)  # [N, V, D, H*W]
        
        # === Ada_Proto ===
        Q = query.size(0)
        batch_size = min(32, Q)
        
        proto_q_list = []
        p_w_list = []
        for i in range(0, Q, batch_size):
            batch_query = query[i:i+batch_size]
            batch_size_actual = batch_query.size(0)
            
            sim_batch = []
            for v in range(self.prj_num):
                view_feat = view_proto[:, v].flatten(1)  # [N, D*H*W]
                view_feat = F.normalize(view_feat, dim=1)
                
                sim = torch.mm(
                    batch_query.flatten(1), 
                    view_feat.transpose(0,1)
                ) / (self.temperature + 1e-6)
                sim_batch.append(sim.unsqueeze(-1))
            
            sim_all = torch.cat(sim_batch, dim=-1)  # [B, N, V]
            p_w = F.softmax(sim_all, dim=-1)        # [B, N, V]
            p_w_list.append(p_w.detach().cpu())
            
            batch_proto_q = torch.zeros(
                batch_size_actual, self.n_way, self.feat_dim, view_proto.size(-1), 
                device=feat.device
            )
            for v in range(self.prj_num):
                weight = p_w[..., v].unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
                view_feat = view_proto[:, v].unsqueeze(0)       # [1, N, D, H*W]
                batch_proto_q += weight * view_feat
            
            proto_q_list.append(batch_proto_q)
        
        proto_q = torch.cat(proto_q_list, dim=0)  # [Q, N, D, H*W]
        
        spatial_attn = torch.sigmoid(proto_q.mean(dim=2, keepdim=True))  # [Q, N, 1, H*W]
        proto_q = (proto_q * spatial_attn).sum(dim=-1)  # [Q, N, D]

        self.last_proto_q   = proto_q.detach().cpu()    # [Q, N, D]
        self.last_p_w       = torch.cat(p_w_list, dim=0)    # [Q, N, V]

        query_vec = query.mean(dim=[2,3])   # [Q, D]
        self.last_query_vec = query_vec.detach().cpu()

        align_loss = qpa_loss(query_vec, proto_q, label)
        fused_proto = self.cross_attn(query_vec, proto_q)


        # === Delta Gating ===
        base_proto = view_proto.mean(dim=[1,3])     # [N, D]
        self.last_base_proto = base_proto.detach().cpu()

        gated = self.delta_gate(fused_proto, base_proto.unsqueeze(0).expand_as(fused_proto))
        gated = F.normalize(gated, dim=-1)
        self.last_gated = gated.detach().cpu()
        
        # === class ===
        query_exp = query.mean(dim=[-2,-1]).unsqueeze(1)  # [Q, 1, D]
        cos_sim = F.cosine_similarity(query_exp, gated, dim=-1)  # [Q, N]
        
        # === loss ===
        
        cls_loss = self.ais_loss(cos_sim, label, self.epoch)
        total_loss = cls_loss + 0.3 * align_loss
        
        pred = F.softmax(cos_sim, dim=-1)

        return pred, total_loss
    
if __name__ == '__main__':
    n_way = 5
    k_shot = 1
    q = 10
    prj_num = 14
    feat_dim = 64
    H = W = 14

    # total images = support projections + query images
    feat = torch.randn((n_way*k_shot*prj_num+n_way*q, feat_dim, H, W))

    # query labels only (length = n_way*q)
    label = torch.arange(n_way).repeat_interleave(q)

    net = QAPR_Net(n_way=n_way, k_shot=k_shot, query=q, prj_num=prj_num, feat_dim=feat_dim)

    pred, loss = net(feat, [None, label])
