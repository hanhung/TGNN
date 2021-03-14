"""

Modified from: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/data.py,
               https://github.com/daveredrum/ScanRefer/blob/master/lib/dataset.py,
               https://github.com/daveredrum/ScanRefer/blob/master/lib/dataset.py

Dataloader for validation

"""

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp, time, json, pickle

from config import *

type2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
            'refrigerator':12, 'shower curtain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'others':17}  

nyu40ids = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40])

SCANNET_V2_TSV = '../scannet_data/meta_data/scannetv2-labels.combined.tsv' 
scanrefer = json.load(open('ScanRefer/ScanRefer_filtered_val.json'))

def get_raw2label():
	# mapping
	scannet_labels = type2class.keys()
	scannet2label = {label: i for i, label in enumerate(scannet_labels)}

	lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
	lines = lines[1:]
	raw2label = {}
	for i in range(len(lines)):
		label_classes_set = set(scannet_labels)
		elements = lines[i].split('\t')
		raw_name = elements[1]
		nyu40_name = elements[7]
		if nyu40_name not in label_classes_set:
			raw2label[raw_name] = scannet2label['others']
		else:
			raw2label[raw_name] = scannet2label[nyu40_name]

	return raw2label

def get_unique_multiple_lookup(raw2label):
    all_sem_labels = {}
    cache = {}
    for data in scanrefer:
        scene_id = data["scene_id"]
        object_id = data["object_id"]
        object_name = " ".join(data["object_name"].split("_"))
        ann_id = data["ann_id"]

        if scene_id not in all_sem_labels:
            all_sem_labels[scene_id] = []

        if scene_id not in cache:
            cache[scene_id] = {}

        if object_id not in cache[scene_id]:
            cache[scene_id][object_id] = {}
            try:
                all_sem_labels[scene_id].append(raw2label[object_name])
            except KeyError:
                all_sem_labels[scene_id].append(17)

    # convert to numpy array
    all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

    unique_multiple_lookup = {}
    for i, data in enumerate(scanrefer):
        scene_id = data["scene_id"]
        object_id = data["object_id"]
        object_name = " ".join(data["object_name"].split("_"))
        ann_id = data["ann_id"]

        try:
            sem_label = raw2label[object_name]
        except KeyError:
            sem_label = 17

        unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

        # store
        if scene_id not in unique_multiple_lookup:
            unique_multiple_lookup[scene_id] = {}

        if object_id not in unique_multiple_lookup[scene_id]:
            unique_multiple_lookup[scene_id][object_id] = {}

        if ann_id not in unique_multiple_lookup[scene_id][object_id]:
            unique_multiple_lookup[scene_id][object_id][ann_id] = None

        unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

    return unique_multiple_lookup

raw2label = get_raw2label()
unique_multiple_lookup = get_unique_multiple_lookup(raw2label)

GLOVE_PICKLE = 'glove.p'

# Load the preprocessed data
val_3d = {}
def load_data(name):
    idx = name[0].find('scene')
    scene_name = name[0][idx:idx+12]
    data = torch.load(name[0])
    return data, scene_name
for x in torch.utils.data.DataLoader(
        glob.glob('val/*.pth'),
        collate_fn=load_data, num_workers=mp.cpu_count()):
    val_3d[x[1]] = x[0]
print('Validating examples:', len(val_3d))

# Load Glove Embeddings
with open(GLOVE_PICKLE, 'rb') as f:
    glove = pickle.load(f)

# Load the ScanRefer dataset
scanrefer = json.load(open('ScanRefer/ScanRefer_filtered_val.json'))

lang = {}
for i, data in enumerate(scanrefer):
    scene_id = data['scene_id']
    object_id = data['object_id']
    ann_id = data['ann_id']

    if scene_id not in lang:
        lang[scene_id] = {'idx':[]}
    if object_id not in lang[scene_id]:
        lang[scene_id][object_id] = {}
    tokens = data['token']
    embeddings = np.zeros((MAX_DES_LEN, 300))
    for token_id in range(MAX_DES_LEN):
        if token_id < len(tokens):
            token = tokens[token_id]
            if token in glove:
                embeddings[token_id] = glove[token]
            else:
                embeddings[token_id] = glove['unk']
        lang[scene_id][object_id][ann_id] = [embeddings, len(tokens)]
    
    lang[scene_id]['idx'].append(i)
    
loader_list = list(val_3d.keys())

valOffsets=[0]
valLabels=[]

def trainMerge(tbl):
    locs=[]
    feats=[]
    labels=[]
    ins_labels=[]
    ref_labels=[]
    coords=[]
    num_points=[]
    point_ids=[]    
    scene_names=[]
    batch_ins_names=[]
    batch_lang_feat=[]
    batch_lang_len=[]
    batch_lang_objID=[]
    batch_lang_objname=[]
    batch_sentences=[]
    batch_tokens=[]
    batch_ref_lbl=[]
    batch_pred_data=[]
    for idx,scene_id in enumerate(tbl):
        scene_dict = lang[scene_id]
        refer_idxs = lang[scene_id]['idx']
        lang_feat=[]
        lang_len=[]
        lang_objID=[]
        lang_objname=[]
        sentences=[]
        tokens=[]
        pred_datas=[]
        for i in refer_idxs:
            scene_id = scanrefer[i]['scene_id']  
            object_id = scanrefer[i]['object_id']
            ann_id = scanrefer[i]['ann_id']
            object_name = ' '.join(scanrefer[i]['object_name'].split('_'))
            object_cat = raw2label[object_name] if object_name in raw2label else 17
            others = 1 if object_cat == 17 else 0
         
            lang_feat.append(torch.from_numpy(lang[scene_id][object_id][ann_id][0])) 
            lang_len.append(min(MAX_DES_LEN, lang[scene_id][object_id][ann_id][1]))
            lang_objID.append(int(object_id))
            lang_objname.append(object_name)
            sentences.append(scanrefer[i]['description'])
            tokens.append(scanrefer[i]['token'])
            
            # --------FOR EVALUATION----------
            pred_data = {
                'scene_id': scene_id,
                'object_id': object_id,
                'ann_id': ann_id,
                'unique_multiple': unique_multiple_lookup[scene_id][object_id][ann_id],
                'others': others
            }
            # ---------------------------------
            pred_datas.append(pred_data)

        # Obj_num, 30, 300
        lang_feat=torch.stack(lang_feat, 0)
        # Obj_num, 
        lang_len = torch.LongTensor(lang_len)
        # Obj_num, 
        lang_objID=torch.LongTensor(lang_objID)
        batch_lang_feat.append(lang_feat)
        batch_lang_len.append(lang_len)
        batch_lang_objID.append(lang_objID)
        batch_lang_objname.append(np.array(lang_objname))
        batch_sentences.append(sentences)
        batch_tokens.append(tokens)
        batch_pred_data.append(pred_datas)
        
        a,b,c,d=val_3d[scene_id]
        coord = a
        m=np.eye(3)
        m*=scale
        a=np.matmul(a,m)
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
        coord=coord[idxs]

        a=torch.from_numpy(a).long()
        locs.append(torch.cat([a,torch.LongTensor(a.shape[0],1).fill_(idx)],1))
        feats.append(torch.from_numpy(b)+torch.randn(3)*0.1)
        labels.append(torch.from_numpy(c))
        ins_labels.append(torch.from_numpy(d.astype(int)))
        coords.append(torch.from_numpy(coord))
        num_points.append(a.shape[0])
        scene_names.append(scene_id)

        # Label
        # Num_points, Obj_num
        ref_lbl = (ins_labels[-1].unsqueeze(-1)) == lang_objID
        batch_ref_lbl.append(ref_lbl.long())
    locs=torch.cat(locs,0)
    feats=torch.cat(feats,0)
    labels=torch.cat(labels,0)
    ins_labels=torch.cat(ins_labels,0)
    coords = torch.cat(coords,0)
    #point_ids = torch.cat(point_ids, 0)
    batch_data = {'x': [locs,feats], 
                  'y': labels.long(),
                  'id': tbl,
                  'y_ins': ins_labels.long(),
                  'coords': coords,
                  'num_points': num_points,
                  'names': scene_names,
                  'lang_feat': batch_lang_feat,
                  'lang_len': batch_lang_len,
                  'lang_objID': batch_lang_objID,
                  'lang_objname': batch_lang_objname,
                  'sentences': batch_sentences,
                  'tokens': batch_tokens,
                  'ref_lbl': batch_ref_lbl,
                  'pred_data': batch_pred_data} 
    return batch_data

print (len(loader_list))

val_data_loader = torch.utils.data.DataLoader(
    #list(range(len(scanrefer))),
    loader_list, 
    batch_size=batch_size,
    collate_fn=trainMerge,
    num_workers=0, 
    shuffle=True,
    worker_init_fn=lambda x: np.random.seed(x+int(time.time()))
)
