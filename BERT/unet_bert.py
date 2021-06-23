"""

Modified from: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/unet.py

Training for referring model with bert encoder.

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

import data_bert
from util import *
from model_bert import *

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--epochs', type=int, default=32, help='Number of epochs')
parser.add_argument('--exp_name', type=str, default='bert', metavar='N', help='Name of the experiment')
args= parser.parse_args()

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
        
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py')
    os.system('cp unet_bert.py checkpoints' + '/' + args.exp_name + '/' + 'unet_bert.py')
    os.system('cp data_bert.py checkpoints' + '/' + args.exp_name + '/' + 'data_bert.py')
    os.system('cp model_bert.py checkpoints' + '/' + args.exp_name + '/' + 'model_bert.py')
    os.system('cp config_bert.py checkpoints' + '/' + args.exp_name + '/' + 'config_bert.py')

_init_()
io = IOStream('checkpoints/' + args.exp_name + '/run.log')

def batch_train(batch, model, optimizer, optimizer1, batch_size):
    ref_model = model['refer']
    backbone_model = model['backbone']
    bert_model = model['bert']
    
    with torch.no_grad():
        pc_sem, pc_feat, pc_offset = backbone_model(batch['x'])

    idx = 0
    loss = 0
    train_loss = 0
    total_pred = 0
    total_ttl_correct = 0
    loss4print = {'ttl':0}
    for i, num_p in enumerate(batch['num_points']):
        optimizer.zero_grad()
        optimizer1.zero_grad()
        scene_pcfeat = pc_feat[idx:idx+num_p]
        pc_ins = batch['y_ins'][idx:idx+num_p]
        pc_sem = batch['y'][idx:idx+num_p]
        m = pc_ins > -1 # Remove unlabeled points

        # Get instance mean features & coordinate using groundtruth labels
        obj_feat, obj_id = gather(scene_pcfeat[m], pc_ins[m])  
        
        coord = batch['x'][0][idx:idx+num_p, 0:3].float().cuda()
        coord/=data_bert.scale
        #shift
        coord = coord - (coord.max(0)[0]*0.5+coord.min(0)[0]*0.5)
        obj_coord, _ = gather(coord[m], pc_ins[m])

        # Referring---------------------------------------------
        input_ids = batch['input_ids'][i].cuda()
        attention_mask = batch['attention_mask'][i].cuda()
        scene_data = {}
        scene_data['obj_feat'] = obj_feat
        scene_data['obj_coord'] = obj_coord
        obj_gt = batch['lang_objID'][i].unsqueeze(-1).cuda() == obj_id.cuda()
        obj_gt = obj_gt.float().argmax(-1)
        lang_feat = bert_model(input_ids, attention_mask=attention_mask)[0]
        scene_data['lang_feat'] = lang_feat
        scene_data['lang_mask'] = attention_mask
            
        ttl_score = ref_model(scene_data)
        obj_num = obj_coord.shape[0]

        total_pred += ttl_score.shape[0]
        total_ttl_correct += (ttl_score.argmax(-1) == obj_gt).sum()
        ref_ttl_loss = F.cross_entropy(ttl_score, obj_gt)        

        loss = ref_ttl_loss
        loss.backward()
        optimizer.step()
        optimizer1.step()
        loss4print['ttl'] += ref_ttl_loss.item()
        idx += num_p
    loss4print['ttl']/=batch_size 
    return loss4print['ttl'], total_pred, total_ttl_correct

use_cuda = torch.cuda.is_available()
io.cprint(args.exp_name)

# Initialize backbone Sparse 3D-Unet and Text-Guided GNN with Pretrained BERT
backbone_model = SCN().cuda()
backbone_model = nn.DataParallel(backbone_model)
ref_model = RefNetV2(k=16).cuda()
ref_model.relconv = nn.DataParallel(ref_model.relconv)
bert_model = BertModel.from_pretrained('bert-base-uncased').cuda()
bert_model = nn.DataParallel(bert_model)

# Load pretrained instance segmentation model for backbone
models = {}
models['backbone'] = backbone_model
training_epoch = checkpoint_restore(models, 'checkpoints/model_insseg', io, use_cuda)
models['refer'] = ref_model
models['bert'] = bert_model

training_epoch = 1
training_epochs = args.epochs
io.cprint('Starting with epoch: ' + str(training_epoch))

params = list(bert_model.parameters()) + list(ref_model.parameters())
optimizer = optim.Adam([{'params': ref_model.parameters(), 'initial_lr': args.lr}], lr=args.lr)
optimizer1 = optim.Adam([{'params': bert_model.parameters(), 'initial_lr': 2e-5}], lr=2e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5, last_epoch=training_epoch)
scheduler1 = optim.lr_scheduler.StepLR(optimizer1, 10, gamma=0.5, last_epoch=training_epoch)

for m in models:
    models[m].train()

for epoch in range(training_epoch, training_epochs+1):
    total_loss = {}
    pbar = tqdm(data_bert.train_data_loader)
    total = 0
    ttl_correct = 0
    obj_correct = 0
    for i,batch in enumerate(pbar):
        optimizer.zero_grad()
        optimizer1.zero_grad()
        if use_cuda:
            batch['x'][1]=batch['x'][1].cuda()

        with torch.autograd.set_detect_anomaly(True):
            loss, t, tc = batch_train(batch, models, optimizer, optimizer1, data_bert.batch_size)
        total += t
        ttl_correct += tc
    scheduler.step()
    scheduler1.step()
    
    outstr = 'Epoch: {}, Loss: {:.4f}, '.format(epoch, loss) + 'Correct Objects: {:.4f}'.format(float(ttl_correct)/float(total))
    io.cprint(outstr)
    checkpoint_save(models, 'checkpoints/'+args.exp_name+'/'+'models'+'/'+args.exp_name, epoch, io, use_cuda)
