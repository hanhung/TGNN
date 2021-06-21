
"""
Modified from: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/unet.py

Script for training an instance segmentation backbone model

"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as scn
import time
import os, sys, glob
import math
import argparse
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
import itertools
from loss import *
from model import SCN, gather
from util import *
from tqdm import tqdm
import data

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
args= parser.parse_args()
    
def change_lr(optimizer,epoch):
    if epoch <= 100:
        lr = args.lr
    elif epoch > 100 & epoch <= 150:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def batch_train(batch, model, optimizer, batch_size):
    backbone_model = model['backbone']
    sem = batch['y'].cuda()
    sem = sem.unsqueeze(-1) == torch.arange(20).cuda()
    sem = sem.float()
    
    pc_sem, pc_feat, pc_offset = backbone_model(batch['x'])

    sem_loss = F.cross_entropy(pc_sem, batch['y'].cuda())
    train_loss = sem_loss 

    loss4print = {'var': 0,'dist': 0, 'cen': 0,'dir': 0} # For print
    idx = 0
    loss = 0
    for i, num_p in enumerate(batch['num_points']):
        scene_pcfeat = pc_feat[idx:idx+num_p]
        pc_ins = batch['y_ins'][idx:idx+num_p]
        pc_sem = batch['y'][idx:idx+num_p]
        m = pc_ins > -1 # Remove unlabeled points
        ins_loss = DiscriminativeLoss(scene_pcfeat[m], pc_ins[m].cuda(), pc_sem[m].cuda(), 0.1,1.5)
        loss += ins_loss['var']+ins_loss['dist']+0.1*ins_loss['reg']

        # Get instance mean features & coordinate using groundtruth labels
        #obj_feat, obj_id = gather(scene_pcfeat[m], pc_ins[m])  
        
        coord = batch['x'][0][idx:idx+num_p, 0:3].float().cuda()
        coord/=data.scale
        #shift
        coord = coord - (coord.max(0)[0]*0.5+coord.min(0)[0]*0.5)
        obj_coord, obj_id = gather(coord[m], pc_ins[m])

        # Predict Center-----------------------------------------
        # Filter out unlabel points and wall and floor
        m = m & (pc_sem != 0) & (pc_sem != 1)
        

        ins_1hot = pc_ins[m].unsqueeze(-1) == obj_id
        gt_cen = obj_coord[ins_1hot.float().argmax(-1)]
        
        pred_offset = pc_offset[idx:idx+num_p]
        cen_loss = CenLoss(coord[m], pred_offset[m], gt_cen, pc_ins[m])
        loss += cen_loss[0] + cen_loss[1]

        loss4print['var'] += ins_loss['var'].item()
        loss4print['dist'] += ins_loss['dist'].item()
        loss4print['cen'] += cen_loss[0].item()
        loss4print['dir'] += cen_loss[1].item()
        idx += num_p

    train_loss += loss/batch_size 
    train_loss.backward()
    optimizer.step()
    for t in loss4print:
        loss4print[t]/=batch_size
    loss4print['sem'] = sem_loss.item()
    return loss4print

def print_loss(loss, i):
    p = ''
    for k in loss.keys():
        p += k + ' {:.3f} '.format(loss[k]/i)
    return p

use_cuda = torch.cuda.is_available()

exp_name = 'model_insseg'

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
if not os.path.exists('checkpoints/'+exp_name):
    os.makedirs('checkpoints/'+exp_name)
if not os.path.exists('checkpoints/'+exp_name+'/'+'models'):
    os.makedirs('checkpoints/'+exp_name+'/'+'models')

backbone_model = SCN().cuda()
backbone_model = nn.DataParallel(backbone_model)
models = {}
models['backbone'] = backbone_model
training_epoch = checkpoint_restore(models, 'checkpoints/'+exp_name+'/'+'models'+'/'+exp_name, use_cuda)
training_epochs=512
print ('Starting with epoch:', training_epoch)

params = backbone_model.parameters()
optimizer = optim.Adam(params, lr=args.lr)

for m in models:
    models[m].train()

for epoch in range(training_epoch, training_epochs+1):
    change_lr(optimizer, epoch)
    start = time.time()
    total_loss = {}
    pbar = tqdm(data.train_data_loader)
    total = 0
    ttl_correct = 0
    obj_correct = 0
    for i,batch in enumerate(pbar):
        optimizer.zero_grad()
        if use_cuda:
            batch['x'][1]=batch['x'][1].cuda()
        loss = batch_train(batch, models, optimizer, data.batch_size)
        for k in loss.keys():
            if k not in total_loss:
                total_loss[k] = loss[k]
            else:
                total_loss[k] += loss[k]
    p = print_loss(total_loss, i)
    print(epoch, p)
    checkpoint_save(models, 'checkpoints/'+exp_name+'/'+'models'+'/'+exp_name, epoch, use_cuda)
