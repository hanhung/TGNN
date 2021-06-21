import numpy as np
import sparseconvnet as scn
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from util import *
from config import *

m = 32
residual_blocks= True
block_reps = 2

dimension = 3
full_scale = 4096

class SCN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(dimension,full_scale, mode=4)).add(
            scn.SubmanifoldConvolution(dimension, 3, m, 3, False)).add(
            scn.UNet(dimension, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks)).add(
            scn.BatchNormReLU(m)).add(
            #scn.SubmanifoldConvolution(data.dimension, m, 4, 1, False)).add(
            scn.OutputLayer(dimension))
        self.linear = nn.Linear(m, 20)
        self.linear1 = nn.Linear(m, m) 
        self.cen_pred = nn.Sequential(nn.Linear(m, m), nn.ReLU(), nn.Linear(m, 3))
    def forward(self,x):
        fv = self.sparseModel(x)
        y = self.linear(fv)
        fv = self.linear1(fv)
        offset = self.cen_pred(fv)
        #sigma=self.linear1(y)
        #fv = F.normalize(fv, p=2, dim=1)
        #return fv
        return y, fv, offset

def gather(feat, lbl):
    uniq_lbl = torch.unique(lbl)
    gather_func = scn.InputLayer(1, uniq_lbl.shape[0], mode=4)
    grp_f = gather_func([lbl.long().unsqueeze(-1), feat])
    grp_idx = grp_f.get_spatial_locations()[:,0]
    grp_idx, sorted_indice = grp_idx.sort()
    grp_f = grp_f.features[sorted_indice]
    return grp_f, grp_idx

def gather_1hot(feat, mask):
    # (obj, )
    obj_size = mask.sum(0)
    mean_f = torch.bmm(mask.unsqueeze(-1).float(), feat.unsqueeze(1))
    # (obj, d)
    mean_f = mean_f.sum(0) / obj_size.float().unsqueeze(-1)
    idx = torch.arange(mask.shape[1]).cuda()
    return mean_f, idx
