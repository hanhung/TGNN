import torch, torch.nn.functional as F
import numpy as np
from itertools import combinations
import sparseconvnet as scn


#   Discriminative Loss
def DiscriminativeLoss(fv, ins_lbl, sem_lbl, epsilon, delta):
    # fv: feature vector for each point(N, d)  
    # ins_lbl: instance label for each point (N, )
    # sem_lbl: semantic label for each point (N, ) NOT USED
    # epsilon: expected maximum radius within a cluster (instance) 
    # delta: expected minimum distance between two cluster center (instances)

    num, d = fv.shape
    uniq_ins = torch.unique(ins_lbl)
    grp_1hot = ins_lbl.unsqueeze(-1) == uniq_ins
    ins_lbl = grp_1hot.float().argmax(1)

    gather_func = scn.InputLayer(1, uniq_ins.shape[0], mode=4)
    # Obtain average features for each instance
    grp_mean = gather_func([ins_lbl.unsqueeze(-1), fv])
    # Get instance idx 
    grp_idx = grp_mean.get_spatial_locations()[:,0]
    grp_mean = grp_mean.features[grp_idx.argsort()]

    # Variance loss
    var_loss = torch.norm(fv - grp_mean[ins_lbl.long()], p=2, dim=1)
    var_loss = torch.max(var_loss-epsilon, torch.zeros_like(var_loss)) 
    
    var_loss = gather_func([ins_lbl.unsqueeze(-1), (var_loss**2).unsqueeze(-1)])
    var_loss = var_loss.features.mean()


    # Distance Loss
    pairwise = torch.Tensor(list(combinations(torch.unique(ins_lbl), 2))).long()
    dist_loss = torch.norm(grp_mean[pairwise[:,0]] - grp_mean[pairwise[:,1]], p=2, dim=1)
    dist_loss = torch.max(2*delta - dist_loss, torch.zeros_like(dist_loss))
    dist_loss = dist_loss.mean()
    

    reg_loss = torch.norm(grp_mean, p=2, dim=1).mean()

    loss = {'var':var_loss, 'dist':dist_loss, 'reg':reg_loss}
    return loss


def CenLoss(coord, pred_offset, gt_cen, ins_lbl):
    uniq_ins = torch.unique(ins_lbl)
    gather_func = scn.InputLayer(1, uniq_ins.shape[0], mode=4)

    # Regression Loss between predicted centers and ground truth centers
    reg_loss = F.smooth_l1_loss(coord+pred_offset, gt_cen, reduction='none')
    reg_loss = reg_loss.sum(-1)
    reg_loss = reg_loss.mean()

    # Direction Loss
    dir_loss = 1 - (F.normalize(pred_offset, dim=-1) * F.normalize(gt_cen - coord, dim=-1)).sum(-1)
    dir_loss = dir_loss.mean()  
    return reg_loss, dir_loss
