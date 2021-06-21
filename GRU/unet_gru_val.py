import time
import math
import json
import argparse
import itertools
import numpy as np
from math import pi
import os, sys, glob
from tqdm import tqdm
import sparseconvnet as scn

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from util import *
from model import *
from cluster import *
import data_val as data

parser = argparse.ArgumentParser()
parser.add_argument('--restore_epoch', type=int, default=32, metavar='N', help='Epoch of model to restore')
parser.add_argument('--exp_name', type=str, default='gru', metavar='N', help='Name of the experiment')
args= parser.parse_args()

DIR = None
def _init_():
	DIR = './validation/' + args.exp_name + '/scenes/'
	print ('Save directory: ', DIR)

	if not os.path.exists('./validation'):
		os.mkdir('./validation')
	if not os.path.exists('./validation/' + args.exp_name):
		os.mkdir('./validation/' + args.exp_name)
	# if not os.path.exists(DIR):
	# 	os.mkdir(DIR)
_init_()
DIR = './validation/' + args.exp_name + '/scenes/'
io = IOStream('validation/' + args.exp_name + '/run.log')
    
def batch_val(batch, model, batch_size):
    backbone_model = model['backbone']
    ref_model = model['refer']

    sem, pc_feat, offset = backbone_model(batch['x'])
    sem = F.softmax(sem, -1)

    IOUs = []
    idx = 0
    for i, num_p in enumerate(batch['num_points']):
        name = batch['names'][i]
        io.cprint(name)

        scene_pcfeat = pc_feat[idx:idx+num_p]
        scene_sem = sem[idx:idx+num_p]
        scene_offset = offset[idx:idx+num_p]
        scene_coords = batch['x'][0][idx:idx+num_p, 0:3].float().cuda()
        scene_coords /= data.scale
        scene_coords = scene_coords - (scene_coords.max(0)[0]*0.5 + scene_coords.min(0)[0]*0.5)

        #pnt_grps = torch.zeros(scene_pcfeat.shape[0]).to(scene_pcfeat.device)

        # Mask out wall and floor
        m = scene_sem.argmax(-1) > 1
        pred_sem = scene_sem.argmax(-1)

        pred_cen = scene_coords + scene_offset

        scene_data = {}
        scene_data['lang_feat'] = batch['lang_feat'][i].float().cuda()
        scene_data['lang_len'] = batch['lang_len'][i]
 
        grps, grp_feat, grp_cen, _, _ = IterativeSample(scene_pcfeat[m], scene_coords[m], scene_offset[m], scene_sem[m])

        # Clustering for Wall & Floor, here we use mean shift clustering
        grps1 = SampleMSCluster(scene_pcfeat[~m], scene_coords[~m])
        grp_feat1, _ = gather(scene_pcfeat[~m], grps1)
        grp_cen1, _ = gather(scene_coords[~m], grps1)
        
        ins_mask = torch.zeros(num_p, grps.shape[-1]).to(grps.device).long()
        ins_mask[m] = grps

        obj_feat = torch.cat([grp_feat, grp_feat1], 0)
        obj_coord = torch.cat([grp_cen, grp_cen1], 0)

        obj_num, dim = obj_feat.shape
        scene_data['obj_feat'] = obj_feat
        scene_data['obj_coord'] = obj_coord

        possible_obj_num = grp_feat.shape[0]
        total_score = ref_model(scene_data)
        total_score = total_score[:, 0:possible_obj_num]
        total_score = F.softmax(total_score, -1)
        
        scores = [total_score.cpu().numpy()]

        pred = ins_mask[:, total_score.argmax(-1)]
        gt = batch['ref_lbl'][i].cuda()
        iou = (pred*gt).sum(0).float()/((pred|gt).sum(0).float()+1e-5)
        IOUs.append(iou.cpu().numpy())

        precision_half = (iou > 0.5).sum().float()/iou.shape[0]
        precision_quarter = (iou > 0.25).sum().float()/iou.shape[0]
        outstr = 'mean IOU {:.4f} | P@0.5 {:.4f} | P@0.25 {:.4f}'.format(iou.mean().item(), precision_half, precision_quarter)
        io.cprint(outstr)

        idx += num_p
    IOUs = np.concatenate(IOUs, 0)
    return IOUs

use_cuda = torch.cuda.is_available()

backbone_model = SCN().cuda()
backbone_model = nn.DataParallel(backbone_model)
ref_model=RefNetGRU(k=16).cuda()
ref_model.relconv = nn.DataParallel(ref_model.relconv)
models = {'backbone': backbone_model, 'refer': ref_model}

training_epoch = checkpoint_restore(models, 'checkpoints/'+args.exp_name+'/'+'models'+'/'+args.exp_name, io, use_cuda, args.restore_epoch)
for m in models:
    models[m].eval()

total_ious = []

for i,batch in enumerate(tqdm(data.val_data_loader)):
    if use_cuda:
        batch['x'][1]=batch['x'][1].cuda()
    with torch.no_grad():
        IOUs = batch_val(batch, models, data.batch_size)
        total_ious.append(IOUs)
    print ('({}/{}) Mean IOU so far {:.4f}'.format((i+1)*data.batch_size, len(data.loader_list), np.concatenate(total_ious, 0).mean()))
total_ious = np.concatenate(total_ious, 0)
IOU = total_ious.mean()

outstr = 'Mean IOU: {}'.format(IOU)
io.cprint(outstr)
Precision = (total_ious > 0.5).sum().astype(float)/total_ious.shape[0]
outstr = 'P@0.5: {}'.format(Precision)
io.cprint(outstr)
Precision = (total_ious > 0.25).sum().astype(float)/total_ious.shape[0]
outstr = 'P@0.25: {}'.format(Precision)
io.cprint(outstr)
