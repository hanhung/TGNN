"""

Modified from: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py

Prepare .pth file which stores [(xyz), (rgb), (semantic label), (instance label)]

"""
import glob, plyfile, numpy as np, multiprocessing as mp, torch, json, os

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper=np.ones(150)*(-100)
for i,x in enumerate([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]):
    remapper[x]=i

files=sorted(glob.glob('../scannet_data/scans/*/*_vh_clean_2.ply'))

#Destined data directory
DIR = './preprocessed_scannet/'
if not os.path.exists(DIR):
    os.makedirs(DIR)

def read_aggre(name):
    f = open(name, 'r')
    results = {}
    d = json.load(f)
    l = d['segGroups']
    for i in l:
        for s in i['segments']:
            results[s] = i['id']
    return results

def read_segs(name, aggregation):
    f = open(name, 'r')
    d = json.load(f)
    indices = np.array(d['segIndices'])
    results = np.zeros_like(indices) - 1
    for i in aggregation:
        m = indices == i
        results[m] = aggregation[i]
    return results


def f(fn):
    fn2 = fn[:-3]+'labels.ply'
    aggre_fn = fn[:-6]+'.aggregation.json'
    segs_fn = fn[:-3]+'0.010000.segs.json'
    a=plyfile.PlyData().read(fn)
    v=np.array([list(x) for x in a.elements[0]])
    coords=np.ascontiguousarray(v[:,:3]-v[:,:3].mean(0))
    colors=np.ascontiguousarray(v[:,3:6])/127.5-1
    a=plyfile.PlyData().read(fn2)
    w=remapper[np.array(a.elements[0]['label'])]
    ins = read_segs(segs_fn, read_aggre(aggre_fn))
    name=fn.split('/')[-1][0:12]		
    torch.save((coords,colors,w,ins),DIR+name+'.pth')
    print(name, 'done')

p = mp.Pool(processes=mp.cpu_count())
p.map(f,files)
p.close()
p.join()
