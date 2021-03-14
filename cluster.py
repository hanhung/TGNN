import torch
import numpy as np
from util import NMS, MS
import torch.nn.functional as F

INIT_COORD_RADIUS = 0.3
INIT_FEAT_RADIUS = 1

COORD_RADIUS = 1
FEAT_RADIUS = 1
SAMPLE_KERNEL = 0.2

ITER_F_KERNEL = 0.5
ITER_C_KERNEL = 0.5

MAX_ITER = 10
MAX_PROPOSALS = 200

def IterativeSample(feat, coord, offset, sem, use_NMS=True):
    d = (offset**2).sum(-1)**0.5  # Length of predicted offset
    pred_cen = coord + offset   # Shift each point to predicted center
    sort_d, sort_idx = d.sort()

    # Select points according to their predicted offset length        
    p = torch.exp(-sort_d**2/SAMPLE_KERNEL**2) # Probability for being selected
    sample_m = p > torch.rand(p.shape[0]).to(p.device)
    sort_idx = sort_idx[sample_m]

    assigned = torch.zeros_like(d)
    feat_cen = []
    coord_cen = []

    # Merge points that are too close to each other, and obtain initial proposals
    for idx in sort_idx:
        if assigned[idx]: continue
        m = ((pred_cen[idx] - pred_cen)**2).sum(-1)**0.5 <= INIT_COORD_RADIUS
        m1 = ((feat[idx] - feat)**2).sum(-1)**0.5 <= INIT_FEAT_RADIUS
        m = m&m1
        assigned[m] = 1
        feat_cen.append(feat[m].mean(0))
        coord_cen.append(pred_cen[m].mean(0))
        if len(feat_cen) >= MAX_PROPOSALS: break

    # Proposal features and coordinates
    feat_cen = torch.stack(feat_cen, 0)
    coord_cen = torch.stack(coord_cen, 0)


    masks = []
    new_feat_cen = []
    new_coord_cen = []
    new_sem = []
    score = []

    # Iteratively resampling to refine each proposal 
    # After refinement, each proposal predicts an instance mask
    for i in range(feat_cen.shape[0]):
        iteration = 0
        diff = float('inf')
        while(iteration < MAX_ITER  and diff > 0.01):
            d = ((feat - feat_cen[i])**2).sum(-1)/(2*ITER_F_KERNEL**2)
            d1 = ((pred_cen - coord_cen[i])**2).sum(-1)/(2*ITER_C_KERNEL**2)
            p = torch.exp(-d) * torch.exp(-d1)
            m = p > torch.rand(p.shape[0]).to(p.device)

            tmp_feat = feat[m].mean(0) # New proposal feature
            tmp_coord = pred_cen[m].mean(0) # New proposal coordinate
            diff = ((tmp_feat - feat_cen[i])**2).sum()**0.5 
            feat_cen[i] = tmp_feat
            coord_cen[i] = tmp_coord
            iteration += 1

        # Predict instance mask
        m = ((feat_cen[i] - feat)**2).sum(-1)**0.5 <= FEAT_RADIUS
        m1 = ((coord_cen[i] - pred_cen)**2).sum(-1)**0.5 <= COORD_RADIUS
        m = m*m1
        masks.append(m.long()) # Instance mask
        new_feat_cen.append(feat[m].mean(0)) # Instance feature
        new_coord_cen.append(coord[m].mean(0)) # Instance coordinate

        # Compute score (For NMS)
        dir_match = F.normalize(coord[m].mean(0)-coord[m], p=2, dim=-1) *\
                    F.normalize(offset[m], p=2, dim=-1)
        dir_match = dir_match.sum(-1)
        score.append(dir_match.mean())
        new_sem.append(sem[m].mean(0).argmax()) # Semantic class of the instance
    masks = torch.stack(masks, -1)
    new_feat_cen = torch.stack(new_feat_cen, 0)
    new_coord_cen = torch.stack(new_coord_cen, 0)
    score = torch.Tensor(score)
    new_sem = torch.LongTensor(new_sem)
    new_mask = masks

    # Use non maximum suppression to remove overlapped masks
    if use_NMS:
        #print ('Before NMS ', masks.shape)
        new_mask, idx = NMS(masks, score, 0.25)
        new_mask = new_mask.to(masks.device)
        #print ('After NMS ', new_mask.shape)
        new_feat_cen = new_feat_cen[idx]
        new_coord_cen = new_coord_cen[idx]
        score = score[idx]
        new_mask = new_mask.to(masks.device)
        new_sem = new_sem[idx]

    return new_mask, new_feat_cen, new_coord_cen, new_sem, score


def SampleMSCluster(feat, coord, max_num=3000, kernel_size=1):
    sample_num = min(max_num, feat.shape[0])
    rand_idx = np.random.choice(feat.shape[0], sample_num, replace=False)
    grp, feat_cen = MS(feat[rand_idx], kernel_size)

    d = (coord.cpu()**2).sum(-1).unsqueeze(-1)+(coord[rand_idx].cpu()**2).sum(-1)-2*torch.mm(coord.cpu(), coord[rand_idx].cpu().permute(1,0))
    result = grp[d.argmin(-1)]
    return result
