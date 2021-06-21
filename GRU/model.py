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

class SentenceEncoderGRU(nn.Module):
    def __init__(self, embedding_size=300, input_size=256, hidden_size=256, output_size=64, bidirectional=True):
        nn.Module.__init__(self)
        self.mlp = nn.Sequential(nn.Linear(embedding_size, input_size), nn.ReLU())
        self.biLSTM = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=False)
        self.lang_mlp = nn.Linear(hidden_size, output_size)
    def forward(self, lang_feat, lang_len):
        lang_feat = self.mlp(lang_feat)
        total_length = lang_feat.shape[1]
        lang_feat = pack_padded_sequence(lang_feat, lang_len, batch_first=True, enforce_sorted=False)
        output, hidden = self.biLSTM(lang_feat)
        # batch, sen_len, vec_size
        #lang_feat, _ = pad_packed_sequence(lang_feat, batch_first=True)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=total_length)
        output = self.lang_mlp(output)
        #weight = torch.zeros(hidden.shape[0], 2).cuda()
        return output, lang_len

class Attention(nn.Module):
    def __init__(self, input_dim):
        nn.Module.__init__(self)
        self.fc = nn.Linear(input_dim, 1)
    def forward(self, context, embedding, lang_len):
        #mask = (embedding != 0).any(-1).float()
        batch_size, max_len, _ = embedding.shape
        mask = torch.arange(max_len).repeat(batch_size, 1)
        mask = (mask < lang_len.unsqueeze(-1)).float().cuda()
        attn = self.fc(context).squeeze(-1) 
        #attn = F.softmax(attn)
        attn = torch.exp(attn)
        attn = attn * mask
        attn = attn / attn.sum(1).unsqueeze(-1)
        # batch, sen_len, vec_size
        weighted_embedd = embedding * attn.unsqueeze(-1)
        weighted_embedd = weighted_embedd.sum(1)
        return weighted_embedd, attn

class TARelationConv(nn.Module):
    def __init__(self, lang_id, lang_od, pc_id, pc_od, k):
        nn.Module.__init__(self)
        self.k = k
        self.rel_encoder = nn.Sequential(nn.Linear(10, lang_od), nn.ReLU(), nn.Linear(lang_od, lang_od))
        self.lang_encoder = nn.Sequential(nn.Linear(lang_id, lang_od), nn.ReLU(), nn.Linear(lang_od, lang_od))
        self.feat_encoder = nn.Sequential(nn.Linear(pc_id, pc_od), nn.ReLU(), nn.Linear(pc_od, pc_od))
        #self.merge = nn.Sequential(nn.Linear(pc_od+lang_od, pc_od), nn.ReLU(), nn.Linear(pc_od, pc_od))
    def forward(self, feat, coord, lang_feat, lang_len):
        num_sen, num_obj, _ = feat.shape
        k = min(self.k, num_obj-1)
        d = ((coord.unsqueeze(1) - coord.unsqueeze(2))**2).sum(-1)
        indice0 = torch.arange(coord.shape[0]).view(coord.shape[0],1,1).repeat(1, num_obj, k+1)
        _, indice1 = torch.topk(d, k+1, dim=-1, largest=False)

        coord_expand = coord[indice0, indice1]
        coord_expand1 = coord.unsqueeze(2).expand(coord.shape[0], coord.shape[1], k+1, coord.shape[-1])
        rel_coord = coord_expand - coord_expand1
        d = torch.norm(rel_coord, p=2, dim=-1).unsqueeze(-1)
        rel = torch.cat([coord_expand, coord_expand1, rel_coord, d], -1)
        # num_sen, num_obj, k+1, d
        rel = self.rel_encoder(rel)

        rel = rel.view(rel.shape[0], -1, rel.shape[-1])
        num_sen, max_len, _ = lang_feat.shape
        mask = torch.arange(max_len).expand(num_sen, max_len).to(feat.device)
        mask = (mask < lang_len.unsqueeze(-1)).float().unsqueeze(1)
        lang_feat = self.lang_encoder(lang_feat)
        feat = self.feat_encoder(feat)

        # num_sen, num_obj*(k+1), T
        attn = torch.bmm(feat[indice0, indice1].view(feat.shape[0],-1,feat.shape[-1]), lang_feat.permute(0,2,1))
        #mask: num_sen, 1, T
        attn = F.softmax(attn, -1) * mask
        attn = attn / (attn.sum(-1).unsqueeze(-1) + 1e-7)
        # num_sen, num_obj*(k+1), d
        ins_attn_lang_feat = torch.bmm(attn, lang_feat)

        # num_sen, num_obj*(k+1), d
        dim = rel.shape[-1]
        rel = rel.view(num_sen, num_obj, k+1, dim)

        ins_attn_lang_feat = ins_attn_lang_feat.view(num_sen, num_obj, k+1, dim)
        feat = ((feat[indice0, indice1] * ins_attn_lang_feat) * rel).sum(2) + feat

        score = feat.sum(-1)
        return feat, score

class TARelationConvBlock(nn.Module):
    def __init__(self, k):
        nn.Module.__init__(self)
        self.conv = TARelationConv(256, 128, 32, 128, k)
        self.conv1 = TARelationConv(256, 128, 128, 128, k)
        self.conv2 = TARelationConv(256, 128, 128, 128, k)
        self.ffn = nn.Linear(128, 1)
    def forward(self, feat, coord, lang_feat, lang_len):
        feat, _ = self.conv(feat, coord, lang_feat, lang_len)
        feat = F.relu(feat)
        feat, _ = self.conv1(feat, coord, lang_feat, lang_len)
        feat = F.relu(feat)
        feat, _ = self.conv2(feat, coord, lang_feat, lang_len)
        feat = F.relu(feat)
        score = self.ffn(feat).squeeze(-1)
        return feat, score

class MLP(nn.Module):
	def __init__(self, in_d, out_d):
		nn.Module.__init__(self)
		self.fc = nn.Sequential(nn.Linear(in_d,out_d),nn.ReLU(),nn.Linear(out_d,out_d))
	def forward(self, x):
		return self.fc(x)

class RefNetGRU(nn.Module):
    def __init__(self, k):
        nn.Module.__init__(self)
        self.lang_encoder = SentenceEncoderGRU(300,512,512,256)
        self.relconv = TARelationConvBlock(k)
    def forward(self, scene):
        #lang_feat: num_sentences, max_len, d   
        #lang_len: num_sentences, 
        num_obj, d = scene['obj_feat'].shape
        #num_sen, num_obj, d
        num_sen, max_len, _ = scene['lang_feat'].shape

        obj_feat = scene['obj_feat'].unsqueeze(0).expand(num_sen,num_obj,d)
        obj_coord = scene['obj_coord'].unsqueeze(0).expand(num_sen,num_obj,3)
   
        #num_sen, 30, d / num_sen 
        lang_feat, lang_len = self.lang_encoder(scene['lang_feat'], scene['lang_len'])
        feat, score = self.relconv(obj_feat, obj_coord, lang_feat, lang_len)
        return score

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
