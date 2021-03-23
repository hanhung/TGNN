"""

Modified from: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/unet.py

Training for referring model with gru encoder.

"""

import math
import argparse
import itertools
import numpy as np
import os, sys, glob
from tqdm import tqdm
import sparseconvnet as scn

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.parallel import DistributedDataParallel as DDP

import data
from util import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--epochs', type=int, default=32, help='Number of epochs')
parser.add_argument('--exp_name', type=str, default='gru', metavar='N', help='Name of the experiment')
args= parser.parse_args()

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')

    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py')
    os.system('cp config.py checkpoints' + '/' + args.exp_name + '/' + 'config.py')
    os.system('cp unet_gru.py checkpoints' + '/' + args.exp_name + '/' + 'unet_gru.py')

_init_()
io = IOStream('checkpoints/' + args.exp_name + '/run.log')

def batch_train(batch, model, optimizer, batch_size):
    ref_model = model['refer']
    backbone_model = model['backbone']
    
    with torch.no_grad():
        pc_sem, pc_feat, pc_offset = backbone_model(batch['x'])

    idx = 0
    loss = 0
    train_loss = 0
    total_pred = 0
    total_ttl_correct = 0
    loss4print = {'ttl':0}
    for i, num_p in enumerate(batch['num_points']):
        scene_pcfeat = pc_feat[idx:idx+num_p]
        pc_ins = batch['y_ins'][idx:idx+num_p]
        pc_sem = batch['y'][idx:idx+num_p]
        m = pc_ins > -1 # Remove unlabeled points

        # Get instance mean features & coordinate using groundtruth labels
        obj_feat, obj_id = gather(scene_pcfeat[m], pc_ins[m])  
        
        coord = batch['x'][0][idx:idx+num_p, 0:3].float().cuda()
        coord/=data.scale
        #shift
        coord = coord - (coord.max(0)[0]*0.5+coord.min(0)[0]*0.5)
        obj_coord, _ = gather(coord[m], pc_ins[m])

        # Referring---------------------------------------------
        lang_len = batch['lang_len'][i]
        lang_feat = batch['lang_feat'][i].float().cuda()

        if lang_feat.shape[0] < 256:
            rand_idx = np.arange(lang_feat.shape[0])
        else:
            rand_idx = np.random.choice(lang_feat.shape[0], 256, replace=False)
            
        scene_data = {}
        scene_data['obj_feat'] = obj_feat
        scene_data['obj_coord'] = obj_coord
        scene_data['lang_len'] = lang_len[rand_idx]
        scene_data['lang_feat'] = lang_feat[rand_idx]
        
        ttl_score = ref_model(scene_data)

        obj_gt = batch['lang_objID'][i][rand_idx].unsqueeze(-1).cuda() == obj_id.cuda()
        obj_gt = obj_gt.float().argmax(-1)

        total_pred += ttl_score.shape[0]
        total_ttl_correct += (ttl_score.argmax(-1) == obj_gt).sum()

        if torch.isnan(ttl_score).any():
            print (ttl_score)
        ref_ttl_loss = F.cross_entropy(ttl_score, obj_gt)
        loss += ref_ttl_loss

        loss4print['ttl'] += ref_ttl_loss.item()
        idx += num_p
    train_loss += loss/batch_size 
    train_loss.backward()
    optimizer.step()
    for t in loss4print:
        loss4print[t]/=batch_size
    return loss4print['ttl'], total_pred, total_ttl_correct

use_cuda = torch.cuda.is_available()
io.cprint(args.exp_name)

# Initialize backbone Sparse 3D-Unet and Text-Guided GNN
backbone_model = SCN().cuda()
backbone_model = nn.DataParallel(backbone_model)
ref_model = RefNetGRU(k=16).cuda()
ref_model.relconv = nn.DataParallel(ref_model.relconv)

# Load pretrained instance segmentation model for backbone
models = {}
models['backbone'] = backbone_model
training_epoch = checkpoint_restore(models, 'checkpoints/model_insseg', io, use_cuda)
models['refer'] = ref_model

training_epoch = 1
training_epochs = args.epochs
io.cprint('Starting with epoch: ' + str(training_epoch))

params = ref_model.parameters()
optimizer = optim.Adam(params, lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

for m in models:
    models[m].train()

for epoch in range(training_epoch, training_epochs+1):
    total_loss = {}
    pbar = tqdm(data.train_data_loader)

    total = 0
    ttl_correct = 0
    obj_correct = 0
    for i, batch in enumerate(pbar):
        optimizer.zero_grad()
        if use_cuda:
            batch['x'][1] = batch['x'][1].cuda()

        with torch.autograd.set_detect_anomaly(True):
            loss, t, tc = batch_train(batch, models, optimizer, data.batch_size)
        total += t
        ttl_correct += tc
    scheduler.step()

    outstr = 'Epoch: {}, Loss: {:.4f}, '.format(epoch, loss) + 'Correct Objects: {:.4f}'.format(float(ttl_correct)/float(total))
    io.cprint(outstr)
    checkpoint_save(models, 'checkpoints/'+args.exp_name+'/'+'models'+'/'+args.exp_name, epoch, io, use_cuda)
