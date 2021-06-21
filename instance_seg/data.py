"""
Modified from: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/data.py
"""

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp, time, json, pickle, random
from config import *



remap = {}

f = open('../labelids.txt', 'r')
NYU_CLASS_IDS = f.readlines()[2:]

for i, line in enumerate(NYU_CLASS_IDS):
    obj_name = line.strip().split('\t')[-1]
    remap[obj_name] = i

train_3d = {}
def load_data(name):
    idx = name[0].find('scene')
    scene_name = name[0][idx:idx+12]
    return torch.load(name[0]), scene_name

for x in torch.utils.data.DataLoader(
        glob.glob('../train/*.pth'),
        collate_fn=load_data, num_workers=mp.cpu_count()):
    train_3d[x[1]] = x[0]

print('Training examples:', len(train_3d))



loader_list = list(train_3d.keys())

#Elastic distortion
blur0=np.ones((3,1,1)).astype('float32')/3
blur1=np.ones((1,3,1)).astype('float32')/3
blur2=np.ones((1,1,3)).astype('float32')/3
def elastic(x,gran,mag):
    bb=np.abs(x).max(0).astype(np.int32)//gran+3
    noise=[np.random.randn(bb[0],bb[1],bb[2]).astype('float32') for _ in range(3)]
    noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
    ax=[np.linspace(-(b-1)*gran,(b-1)*gran,b) for b in bb]
    interp=[scipy.interpolate.RegularGridInterpolator(ax,n,bounds_error=0,fill_value=0) for n in noise]
    def g(x_):
        return np.hstack([i(x_)[:,None] for i in interp])
    return x+g(x)*mag

def trainMerge(tbl):
    locs=[]
    feats=[]
    labels=[]
    ins_labels=[]
    ref_labels=[]
    bboxs=[]
    coords=[]
    num_points=[]
    scene_names=[]

    for idx,scene_id in enumerate(tbl):
        
        
        a,b,c,d=train_3d[scene_id]
        m=np.eye(3)+np.random.randn(3,3)*0.1
        m[0][0]*=np.random.randint(0,2)*2-1
        m*=scale
        theta=np.random.rand()*2*math.pi
        rot = np.array([[math.cos(theta),math.sin(theta),0],[-math.sin(theta),math.cos(theta),0],[0,0,1]])
        m = np.matmul(m, rot)
        a=np.matmul(a,m)

        if elastic_deformation:
            a=elastic(a,6*scale//50,40*scale/50)
            a=elastic(a,20*scale//50,160*scale/50)
        m=a.min(0)
        M=a.max(0)
        q=M-m
        offset=-m+np.clip(full_scale-M+m-0.001,0,None)*np.random.rand(3)+np.clip(full_scale-M+m+0.001,None,0)*np.random.rand(3)
        a+=offset
        idxs=(a.min(1)>=0)*(a.max(1)<full_scale)
        a=a[idxs]
        b=b[idxs]
        c=c[idxs]
        d=d[idxs]

        a=torch.from_numpy(a).long()
        locs.append(torch.cat([a,torch.LongTensor(a.shape[0],1).fill_(idx)],1))
        feats.append(torch.from_numpy(b)+torch.randn(3)*0.1)
        labels.append(torch.from_numpy(c))
        ins_labels.append(torch.from_numpy(d.astype(int)-1))
        num_points.append(a.shape[0])
        scene_names.append(scene_id)
    locs=torch.cat(locs,0)
    feats=torch.cat(feats,0)
    labels=torch.cat(labels,0)
    ins_labels=torch.cat(ins_labels,0)

    batch_data = {'x': [locs,feats],
                  'y': labels.long(),
                  'id': tbl,
                  'y_ins': ins_labels.long(),
                  'num_points': num_points,
                  'names': scene_names}
    return batch_data

data_size = len(loader_list)
print (data_size)
total_iteration = data_size/batch_size
train_data_loader = torch.utils.data.DataLoader(
    #list(range(len(scanrefer))),
    loader_list, 
    batch_size=batch_size,
    collate_fn=trainMerge,
    num_workers=1, 
    shuffle=True,
    drop_last=True,
    worker_init_fn=lambda x: np.random.seed(x+int(time.time()))
)

    
