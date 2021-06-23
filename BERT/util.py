import sparseconvnet as scn
from sklearn.cluster import MeanShift

import torch, os, numpy as np, glob, multiprocessing as mp

NYU_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

class IOStream():
	def __init__(self, path):
		self.f = open(path, 'a')
	
	def cprint(self, text):
		print(text)
		self.f.write(text+'\n')
		self.f.flush()
	
	def close(self):
		self.f.close()

def MS(feat, thres):
    clustering = MeanShift(bandwidth=thres).fit(feat.cpu().numpy())
    grp = torch.from_numpy(clustering.labels_).long()
    center = torch.from_numpy(clustering.cluster_centers_).to(feat.device)
    return grp, center

def NMS(cluster, score, thres=0.5):
    #cluster N, C
    # C 
    results = []
    result_idxs = []
    #cluster_scores = torch.matmul(cluster.permute(1,0).float(), pred).max(1)[0]
    sort_idx = score.argsort(descending=True)  
    cluster = cluster[:, sort_idx]
    cluster = cluster.cpu()
    while(cluster.shape[-1] > 0):
        #print (cluster.shape) 
        if len(sort_idx) <= 1: break
        target = cluster[:,0]
        results.append(target) 
        result_idxs.append(sort_idx[0])
        target = target.unsqueeze(-1)
        if cluster.shape[1] == 1: break
        cluster = cluster[:, 1:]   
        sort_idx = sort_idx[1:] 
        ious = (target*cluster).float().sum(0)/(target|cluster).float().sum(0)
        m = ious <= thres
        cluster = cluster[:, m]
        sort_idx = sort_idx[m]
    results = torch.stack(results, 1)
    result_idxs = torch.LongTensor(result_idxs)
    return results, result_idxs

def checkpoint_save(models, exp_name, epoch, io, use_cuda=True):
	f=exp_name+'-%09d'%epoch+'.pth'
	save = {}
	for k in models.keys():
		model = models[k].cpu()
		save[k] = model.state_dict()
		if use_cuda:
			model.cuda()	
	torch.save(save,f)
	epoch=epoch-1
	f=exp_name+'-%09d'%epoch+'.pth'
	if os.path.isfile(f):
		if epoch % 8 != 0:
			os.remove(f)

def checkpoint_restore(models, exp_name, io, use_cuda=True, epoch=0):
	if use_cuda:
		for m in models:
			models[m].cpu()
	if epoch>0:
		f=exp_name+'-%09d'%epoch+'.pth'
		assert os.path.isfile(f)
		if io != None:
			io.cprint('Restore from ' + f)
		else:
			print('Restore from ' + f)
		checkpoint = torch.load(f)
		for m in models:
			models[m].load_state_dict(checkpoint[m])
	else:
		f=sorted(glob.glob(exp_name+'-*.pth'))
		if len(f)>0:
			f=f[-1]
			checkpoint = torch.load(f)
			if io != None:
				io.cprint('Restore from ' + f)
			else:
				print('Restore from ' + f)
			for m in models:
				models[m].load_state_dict(checkpoint[m])
			#model.load_state_dict(torch.load(f))
			epoch=int(f[len(exp_name)+1:-4])
		else:
			if exp_name.split('/')[1] == 'model_insseg':
				print('Instance Segmentation Model Not Found')
			else:
				print('Trained model Not Found')
			exit()
	if use_cuda:
		for m in models:
			models[m].cuda()
	return epoch+1
