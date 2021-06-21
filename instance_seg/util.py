import sparseconvnet as scn
from sklearn.cluster import MeanShift

import torch, os, numpy as np, glob, multiprocessing as mp

NYU_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

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
	if use_cuda:
		for m in models:
			models[m].cuda()
	return epoch+1
