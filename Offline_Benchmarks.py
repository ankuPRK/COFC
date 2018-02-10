import numpy as np
import os
import time
import cv2
import itertools
import os
import shutil
import GMM_updation_uni as GMMu
import pickle as pkl
import itertools
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy import misc
from scipy.optimize import linear_sum_assignment as lsaa
from collections import Counter
from sklearn.metrics import v_measure_score as vms
from sklearn.metrics import v_measure_score as vms
from sklearn.metrics import adjusted_rand_score as ars
from sklearn.metrics import fowlkes_mallows_score as fms
from sklearn.metrics import adjusted_mutual_info_score as amis
from sklearn.metrics import homogeneity_score as hs
from sklearn.metrics import completeness_score as cs

cntl = 0 

def overlap_in_percent(ref, new):
	lft = max(new[0]-new[2]/2, ref[0]-ref[2]/2)
	rt = min(new[0]+new[2]/2, ref[0]+ref[2]/2)
	top = max(new[1]-new[3]/2, ref[1]-ref[3]/2)
	bot = min(new[1]+new[3]/2, ref[1]+ref[3]/2)
	if rt-lft < 0 or bot-top < 0:
		return 0.0
	else:
		return 100.0 * (rt-lft)*(bot-top) / (0.0001 + max(new[2]*new[3], ref[2]*ref[3]))

def euc_dist_sq(x1,x2):
	return np.sum((x1-x2)**2)

def accuracy_score(y_act, y_pred):
	Cluster_Matrix = np.zeros((max(y_act)+1, max(y_pred)+1))
	print "Cluster_Matrix shape:"
	for i in range(len(y_act)):
		Cluster_Matrix[y_act[i], y_pred[i]]+=1.0	
	Inv_Cluster_Matrix = -1*Cluster_Matrix
	row, col = lsaa(Inv_Cluster_Matrix)
	accu = np.sum(Cluster_Matrix[row, col])/float(np.sum(np.sum(Cluster_Matrix)))
	print(Cluster_Matrix.astype(int))
	print(row)
	print(col)
	return accu

def clustering_score(ls_actual, ls_pred):

	dc = {0:0,1:1,2:2,6:3,9:4,11:5}
	for j in range(len(ls_actual)):
		ls_actual[j] = dc[ls_actual[j]]

	vm_score = vms(ls_actual, ls_pred)
	c_score = cs(ls_actual, ls_pred)
	h_score = hs(ls_actual, ls_pred)
	fm_score = fms(ls_actual, ls_pred)
	accu_score = accuracy_score(ls_actual, ls_pred)

	return vm_score, c_score, h_score, fm_score, accu_score

def update_labels(y_ls,np_t):
	ls_pred = []
	tu = np.unique(np_t)
	ls_tracks = []
	for i in range(len(tu)):
		ls_tracks.append([])

	dt = dict()
	for i in range(len(tu)):
		dt[tu[i]] = i

	print("y_ls: " + str(len(y_ls)))
	print("np_t: " + str(len(np_t)))

	for i in range(len(np_t)):
		tid = np_t[i]
		ls_tracks[dt[tid]].append(y_ls[i])

	dic_id = dict()

	for i in range(len(ls_tracks)):
		t = ls_tracks[i]
		c = Counter(t)
		value, count = c.most_common()[0]
		dic_id[tu[i]]=value

	for i in range(len(np_t)):
		tid = np_t[i]
		ls_pred.append(dic_id[tid]) 

	return ls_pred

def get_avg_feats(X_tr, np_lt, size=20):
	dt = dict()
	lb = dict()
	print("X_tr shape: "+str(X_tr.shape))

	for i in range(X_tr.shape[0]):
		tid = np_lt[i,0]
		if tid in dt:
			dt[tid].append(X_tr[i,:])
			lb[tid] = np_lt[i,1]
		else:
			dt[tid] = [X_tr[i,:]]
			lb[tid] = np_lt[i,1]
	ls_X = []
	ls_y = []
	ls_len = []
	for i in dt:
		fid = lb[i]
		X_t = dt[i]
		ls_ff = []
		for j in range(len(X_t)):
			ls_ff.append(X_t[j])
			if (j+1)%size==0:
				#print(len(ls_ff))
				#print(ls_ff)
				ls_X.append(np.mean(np.array(ls_ff),axis=0))
				ls_y.append(fid)
				ls_len.append(len(ls_ff))
				ls_ff = []
		if(len(ls_ff)>0):
			ls_X.append(np.mean(np.array(ls_ff),axis=0))
			ls_y.append(fid)
			ls_len.append(len(ls_ff))
			ls_ff = []
	np_X = np.array(ls_X)
	np_y = np.array(ls_y)
	np_l = np.array(ls_len)
	print(np_X.shape)
	print(np_y.shape)
	print(np_l.shape)
	print(sum(np_l))
	return np_X, np_y, np_l

def get_feats_and_cluster(np_lt, ls_feats_list, type="kmeans"):

	X_tr = np.array(ls_feats_list)
	print(X_tr.shape)
	y_tr = np_lt[:,1]
	tid = np_lt[:,0]
	print(y_tr.shape)
	print(tid.shape)
	# exit()
	n_comp = 6
	
	if(type=='gmm'):
		modelcurr = GaussianMixture(n_components=n_comp, covariance_type='tied', tol=0.001, 
			reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', 
			weights_init=None, means_init=None, precisions_init=None, 
			random_state=None, warm_start=False, verbose=0, verbose_interval=10)
		modelcurr.fit(X_tr)
		y = modelcurr.predict(X_tr)
		y_ls = []
		y_ = []
		for i in range(y.shape[0]):
			y_ls.append(y[i])
		for i in range(y_tr.shape[0]):
			y_.append(y_tr[i])
		y_pred = update_labels(y_ls, tid)
	elif (type=='kmeans'):
		model = KMeans(n_clusters=n_comp)
		y = model.fit_predict(X_tr)
		y_ls = []
		y_ = []
		for i in range(y.shape[0]):
			y_ls.append(y[i])
		for i in range(y_tr.shape[0]):
			y_.append(y_tr[i])
		y_pred = update_labels(y_ls, tid)
	elif (type=='sfc'):
		np_X, np_y, np_l = get_avg_feats(X_tr, np_lt, size=40)
		model = KMeans(n_clusters=n_comp)
		y = model.fit_predict(np_X)
		y_pred = []
		y_ = []
		for i in range(np_y.shape[0]):
			for j in range(np_l[i]):
				y_.append(np_y[i])
				y_pred.append(y[i])

	vm_score, c_score, h_score, fm_score, accu_score = clustering_score(y_, y_pred)
	return vm_score, c_score, h_score, fm_score, accu_score, n_comp

if __name__ == '__main__':

	with open("./resources/ls_feats_list_buffy.pkl", 'rb' ) as fp:
		ls_feats_list = pkl.load(fp)
	np_lt = np.load("./resources/buffy_tl.npy")

#######################K-Means##############################

	lsa = []
	lsv = []
	lsc = []
	lscm = []
	lsh = []
	lsf = []
	lsac = []
	for k in range(10):
		vm_score, c_score, h_score, fm_score, accu, ncl = get_feats_and_cluster(np_lt, ls_feats_list, "kmeans")
		print("KMeans Accu: " + str(accu))
		lsh.append(h_score)
		lscm.append(c_score)
		lsv.append(vm_score)
		lsc.append(ncl)
		lsf.append(fm_score)
		lsa.append(accu)
	np_h = np.array(lsh)
	np_cm = np.array(lscm)
	np_v = np.array(lsv)
	np_c = np.array(lsc)
	np_f = np.array(lsf)
	np_a = np.array(lsa)
	print "Homo: " + str(np.mean(np_h)) + "+-" + str(np.std(np_h))
	print "Cmpl: " + str(np.mean(np_cm)) + "+-" + str(np.std(np_cm))

	print "VM score: " + str(np.mean(np_v)) + "+-" + str(np.std(np_v))
	print "FM score: " + str(np.mean(np_f)) + "+-" + str(np.std(np_f))
	print "Accu: " + str(np.mean(np_a)) + "+-" + str(np.std(np_a))
	print "No of clusters: " + str(np.mean(np_c)) + "+-" + str(np.std(np_c))

#######################GMM##############################

	lsa = []
	lsv = []
	lsc = []
	lscm = []
	lsh = []
	lsf = []
	lsac = []
	for k in range(10):
		vm_score, c_score, h_score, fm_score, accu, ncl = get_feats_and_cluster(np_lt, ls_feats_list, "gmm")
		print("GMM Accu: " + str(accu))
		lsh.append(h_score)
		lscm.append(c_score)
		lsv.append(vm_score)
		lsc.append(ncl)
		lsf.append(fm_score)
		lsa.append(accu)
	np_h = np.array(lsh)
	np_cm = np.array(lscm)
	np_v = np.array(lsv)
	np_c = np.array(lsc)
	np_f = np.array(lsf)
	np_a = np.array(lsa)
	print "Homo: " + str(np.mean(np_h)) + "+-" + str(np.std(np_h))
	print "Cmpl: " + str(np.mean(np_cm)) + "+-" + str(np.std(np_cm))

	print "VM score: " + str(np.mean(np_v)) + "+-" + str(np.std(np_v))
	print "FM score: " + str(np.mean(np_f)) + "+-" + str(np.std(np_f))
	print "Accu: " + str(np.mean(np_a)) + "+-" + str(np.std(np_a))
	print "No of clusters: " + str(np.mean(np_c)) + "+-" + str(np.std(np_c))

#######################Track-based##############################

	lsa = []
	lsv = []
	lsc = []
	lscm = []
	lsh = []
	lsf = []
	lsac = []
	for k in range(10):
		vm_score, c_score, h_score, fm_score, accu, ncl = get_feats_and_cluster(np_lt, ls_feats_list, "sfc")
		print("Track-based Accu: " + str(accu))
		lsh.append(h_score)
		lscm.append(c_score)
		lsv.append(vm_score)
		lsc.append(ncl)
		lsf.append(fm_score)
		lsa.append(accu)
	np_h = np.array(lsh)
	np_cm = np.array(lscm)
	np_v = np.array(lsv)
	np_c = np.array(lsc)
	np_f = np.array(lsf)
	np_a = np.array(lsa)
	print "Homo: " + str(np.mean(np_h)) + "+-" + str(np.std(np_h))
	print "Cmpl: " + str(np.mean(np_cm)) + "+-" + str(np.std(np_cm))

	print "VM score: " + str(np.mean(np_v)) + "+-" + str(np.std(np_v))
	print "FM score: " + str(np.mean(np_f)) + "+-" + str(np.std(np_f))
	print "Accu: " + str(np.mean(np_a)) + "+-" + str(np.std(np_a))
	print "No of clusters: " + str(np.mean(np_c)) + "+-" + str(np.std(np_c))

	print("Done.")

