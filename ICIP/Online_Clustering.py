import numpy as np
import pickle as pkl
import os
import time
import cv2
import itertools
import dlib
import openface
from sklearn.metrics import v_measure_score as vms
from sklearn.metrics import adjusted_rand_score as ars
from sklearn.metrics import fowlkes_mallows_score as fms
from sklearn.metrics import adjusted_mutual_info_score as amis
from sklearn.metrics import homogeneity_score as hs
from sklearn.metrics import completeness_score as cs
from scipy import misc
from scipy.optimize import linear_sum_assignment as lsa
#import self made modules
import GMM_updation_uni as GMMu

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

def get_vid_list(feats_list):
	with open("./resources/file_Names_buffy.txt","r") as fp:
		lsX = pkl.load(fp)
	print("No. of faces = " + str(len(lsX)) )
	np_filedata = np.zeros((len(lsX), 3), dtype=int)
	for i in range(len(lsX)):
		strname = lsX[i]
		#print(strname)
		np_filedata[i,0] = int(strname[0:4]) #Shot IDs
		np_filedata[i,1] = int(strname[4:11]) #Frame IDs for each shot
		np_filedata[i,2] = int(strname[11:13]) #person IDs for each frame(different for each frame)
	np_cords = np.loadtxt('./resources/loc_info_buffy.txt', dtype=float)
	np_labels = np.loadtxt('./resources/labels_buffy.txt', dtype=int)
	# for every frame, this array has a row in form [xcentre ycentre width height] 
	j_c=0
	track_cnt=0

	ls_all = []
	ls_mats = []
	ls_all_labels = []
	ls_all_feats = []
	print("ss range: "+str(np_filedata[-1,:]))
	for ss in range(0,np_filedata[-1,0]+1):

		fno = 0 #frame no. of 'ss'th shot
		ls_tracks = []
		ls_pos = []
		ls_labels = []
		ls_feats = []
		np_tr = np.zeros((len(ls_tracks),len(ls_tracks)))
		print("shot: " + str(ss) + ", j_c: " + str(j_c))

		while (np_filedata[j_c,0] == ss):
			#ss is the shot number and j is the overall number
			data_file = []
			data_loc = []
			data_label = []
			data_feat = []
			jj = j_c #for getting faces in this frame
			while(np_filedata[j_c,1] == np_filedata[jj,1]):
				data_file.append(lsX[j_c])
				data_loc.append(np_cords[j_c,:])
				data_label.append(np_labels[j_c])
				data_feat.append(feats_list[j_c])
				j_c+=1
				if j_c >= len(lsX):
					break
			ll_matrix = np.zeros((len(ls_pos),len(data_loc)),dtype=float)
			dd_matrix = np.zeros((len(ls_pos),len(data_loc)),dtype=float)
				#fill ll_matrix
			for j in range(0,len(ls_pos)):		#ls no.
				for k in range(0,len(data_loc)): 	#data no
					ll_matrix[j,k] = overlap_in_percent(ls_pos[j], data_loc[k])
					dd_matrix[j,k] = 4.0-euc_dist_sq(ls_feats[j][-1], data_feat[k])

			Threshold = 85
			dd_Threshold = 3.0
			#print("ll_mat:\n"+str(ll_matrix))
			data_ind = np.arange(len(data_loc))
			tr_ind = np.arange(len(ls_pos))
			ls_diff = []

			while (data_ind.shape[0] > 0):
				maxval = -9999
				[argmj, argmk] = [-1, -1]
				for j in range(0,tr_ind.shape[0]):		#model no
					for k in range(0,data_ind.shape[0]): 	#data track no
						#print([j, k])
						if(ll_matrix[j,k] > maxval and dd_matrix[j,k]>=dd_Threshold):
							maxval = ll_matrix[j,k]
							[argmj, argmk] = [j,k]
				if argmj !=-1 and argmk!=-1:
					kmax = data_ind[argmk] 
					jmax = tr_ind[argmj]
					if maxval>Threshold:
						#we got a match
						ls_pos[jmax] = data_loc[kmax] 
						ls_tracks[jmax].append(data_file[kmax])
						ls_labels[jmax].append(data_label[kmax])
						ls_feats[jmax].append(data_feat[kmax])
						ls_diff.append(jmax)
						ll_matrix = np.delete(ll_matrix, argmj, axis=0)
						ll_matrix = np.delete(ll_matrix, argmk, axis=1)
						data_ind = np.delete(data_ind, argmk,axis=0)
						tr_ind = np.delete(tr_ind, argmj,axis=0)
					else:
						#no match i.e. new track
						ls_pos.append(data_loc[kmax])
						ls_tracks.append([data_file[kmax]])
						ls_labels.append([data_label[kmax]])
						ls_feats.append([data_feat[kmax]])
						ls_diff.append(len(ls_tracks)-1)
						ll_matrix = np.delete(ll_matrix, argmk, axis=1)
						data_ind = np.delete(data_ind, argmk,axis=0)
				else:
					#no match i.e. new track
					ls_pos.append(data_loc[data_ind[0]])
					ls_tracks.append([data_file[data_ind[0]]])
					ls_labels.append([data_label[data_ind[0]]])
					ls_feats.append([data_feat[data_ind[0]]])
					ls_diff.append(len(ls_tracks)-1)
					data_ind = np.delete(data_ind, 0,axis=0)
			#sif len(ls_diff) > 1:
			np_tr_new = np.zeros((len(ls_tracks),len(ls_tracks)))
			np_tr_new[0:np_tr.shape[0],0:np_tr.shape[1]] = np_tr
			for t1 in range(len(ls_diff)):
				for t2 in range(t1+1, len(ls_diff)):
					np_tr_new[ls_diff[t1],ls_diff[t2]] += 1
					np_tr_new[ls_diff[t2],ls_diff[t1]] += 1
			np_tr = np_tr_new
			if j_c>=len(lsX):
				break
		ls_all.append(ls_tracks)
		ls_all_labels.append(ls_labels)
		ls_all_feats.append(ls_feats)
		ls_mats.append(np_tr)
		if j_c>=len(lsX):
			break
	return ls_all, ls_mats, ls_all_labels, ls_all_feats

def iterate_progressively(ls_all, ls_all_labels, ls_mats, ls_data):

		print("Iterate progressively.............")
		lsModels = []
		nModels = []
		labels_Models = []
        	#Initialize
		shotNo = -1
		trackNo = -1

		for shot, s_labels, rel_mat,data in itertools.izip(ls_all, ls_all_labels, ls_mats,ls_data):
			shotNo+=1
			shot_mat = np.greater(rel_mat, 0).astype(int)
			print("\nShot " + str(shotNo) + ": ")
			data = [np.array(t) for t in data]
			ll_matrix = np.zeros((len(lsModels),len(data)),dtype=float)
			yn_matrix = np.ones((len(lsModels),len(data)),dtype=float)
			#fill ll_matrix
			for j in range(0,len(lsModels)):		#model no
				for k in range(0,len(data)): 	#data track no
					sumVal=0
					for l in range(0, len(data[k])):
						sumVal+=np.sum((lsModels[j].means_[0] - data[k][l])**2)
					ll = sumVal/(1.0*len(data[k]))
					ll_matrix[j,k] = 4.0-ll
			data_ind = np.arange(len(data))
			#####################################################################################
			Threshold = 4.0-1.2
			#####################################################################################
			
			#the checking
		
			trackNo = 0
			np_occurence = np.zeros((len(data))).astype(int)
			while (data_ind.shape[0] > 0):
				trackNo+=1
				maxval = -9999
				[argmj, argmk] = [-1, -1]
				for j in range(0,len(lsModels)):		#model no
					for k in range(0,data_ind.shape[0]): 	#data track no
						if(yn_matrix[j,k]==1 and ll_matrix[j,k] > maxval):
							maxval = ll_matrix[j,k]
							[argmj, argmk] = [j,k]
				if argmj !=-1 and argmk!=-1:
					kmax = data_ind[argmk] 
					jmax = argmj 
					if maxval>Threshold:
						print("Data " + str(kmax) + " goes into model " + str(jmax))
						nModels[jmax], lsModels[jmax] = GMMu.update_GMM(lsModels[jmax], nModels[jmax], data[kmax])
						labels_Models[jmax] = labels_Models[jmax] + s_labels[kmax]
						np_occurence[kmax] = jmax
						yn_matrix[argmj,:] = yn_matrix[argmj,:] - shot_mat[argmk,:]
						for kko in range(data_ind.shape[0]):
							sumVal=0
							kkk = data_ind[kko]
							for lll in range(0, len(data[kkk])):
								sumVal+=np.sum((lsModels[jmax].means_[0] - data[kkk][lll])**2)
							ll = sumVal/(1.0*len(data[kkk]))
							ll_matrix[jmax,kko] = 4.0-ll
						

						ll_matrix = np.delete(ll_matrix, argmk, axis=1)
						yn_matrix = np.delete(yn_matrix, argmk, axis=1)
						shot_mat = np.delete(shot_mat, argmk,axis=0)
						shot_mat = np.delete(shot_mat, argmk,axis=1)
						data_ind = np.delete(data_ind, argmk,axis=0)
					else:
						print("Data " + str(kmax) + " is sent into new cluster.")
						n, l = GMMu.initialize_GMM(data[kmax])
						lsModels.append(l)
						nModels.append(n)
						labels_Models.append(s_labels[kmax])
						np_occurence[kmax] = len(lsModels) - 1

						if data_ind.shape[0] > 1:
							jjj = len(nModels) - 1
							yn_matrix = np.concatenate([yn_matrix, np.ones((1,data_ind.shape[0]))],axis=0)
							ll_matrix = np.concatenate([ll_matrix, np.zeros((1,data_ind.shape[0]))],axis=0)
							for kko in range(data_ind.shape[0]):
								sumVal=0
								kkk = data_ind[kko]
								for lll in range(0, len(data[kkk])):
									sumVal+=np.sum((lsModels[jjj].means_[0] - data[kkk][lll])**2)
								ll = sumVal/(1.0*len(data[kkk]))
								ll_matrix[jjj,kko] = 4.0-ll
						ll_matrix = np.delete(ll_matrix, argmk, axis=1)
						yn_matrix = np.delete(yn_matrix, argmk, axis=1)
						data_ind = np.delete(data_ind, argmk,axis=0)
						shot_mat = np.delete(shot_mat, argmk, axis=0)
						shot_mat = np.delete(shot_mat, argmk, axis=1)
				else:
						print("Data " + str(data_ind[0]) + " is initialised as a new cluster.")
						n, l = GMMu.initialize_GMM(data[data_ind[0]])
						lsModels.append(l)
						nModels.append(n)
						np_occurence[data_ind[0]] = len(lsModels) - 1
						labels_Models.append(s_labels[data_ind[0]])
						#update the ll matrix
						if data_ind.shape[0] > 1:
							jjj = len(nModels) - 1
							yn_matrix = np.concatenate([yn_matrix, np.ones((1,data_ind.shape[0]))],axis=0)
							yn_matrix[jjj,:] = yn_matrix[jjj,:] - shot_mat[0,:]
							ll_matrix = np.concatenate([ll_matrix, np.zeros((1,data_ind.shape[0]))],axis=0)
							for kko in range(data_ind.shape[0]):
								sumVal=0
								kkk = data_ind[kko]
								for lll in range(0, len(data[kkk])):
									sumVal+=np.sum((lsModels[jjj].means_[0] - data[kkk][lll])**2)
								ll = sumVal/(1.0*len(data[kkk]))
								ll = np.sum((sumVal/(1.0*len(data[kkk])))**2)
								ll_matrix[jjj,kko] = 4.0-ll
						ll_matrix = np.delete(ll_matrix, 0, axis=1)
						yn_matrix = np.delete(yn_matrix, 0, axis=1)
						data_ind = np.delete(data_ind, 0, axis=0)
						shot_mat = np.delete(shot_mat, 0, axis=0)
						shot_mat = np.delete(shot_mat, 0, axis=1)
		return nModels, labels_Models

def accuracy_score(y_act, y_pred):
	Cluster_Matrix = np.zeros((max(y_act)+1, max(y_pred)+1))
	print "Cluster_Matrix shape:"
	for i in range(len(y_act)):
		Cluster_Matrix[y_act[i], y_pred[i]]+=1.0	
	Inv_Cluster_Matrix = -1*Cluster_Matrix
	row, col = lsa(Inv_Cluster_Matrix)
	accu = np.sum(Cluster_Matrix[row, col])/float(np.sum(np.sum(Cluster_Matrix)))
	for i,j in itertools.izip(row,col):
		print(i,j,np.sum(Cluster_Matrix[i,:])/np.sum(np.sum(Cluster_Matrix)),Cluster_Matrix[i,j]/float(np.sum(Cluster_Matrix[:,j])))
	print(Cluster_Matrix.astype(int))
	print(row)
	print(col)
	return accu


def clustering_score(labels_Models):
	for i in range(len(labels_Models)):
		j=0
	ls_pred = []
	for i in range(len(labels_Models)):
		for j in range(len(labels_Models[i])):
			ls_pred.append(i)
	ls_actual = []
	for item in labels_Models:
		ls_actual = ls_actual + item
	print (np.unique(np.array(ls_actual)))
	#Map uneven labels from 0 to 5 
	dc = {0:0,1:1,2:2,6:3,9:4,11:5}
	for j in range(len(ls_actual)):
		ls_actual[j] = dc[ls_actual[j]]
	# print ls_pred 
	vm_score = vms(ls_actual, ls_pred)
	c_score = cs(ls_actual, ls_pred)
	h_score = hs(ls_actual, ls_pred)
	fm_score = fms(ls_actual, ls_pred)
	accu_score = accuracy_score(ls_actual, ls_pred)

	return vm_score, c_score, h_score, fm_score, accu_score

def check_track_uniformity(ls_labels):
	curr = 0
	tot = 0
	sid=0
	tid=0
	for ss in ls_labels:
		for t in ss:
			tot+=len(t)
			if len(np.unique(np.array(t))) > 1:
				print(sid,tid)
				print(np.unique(np.array(t)))
				print(t)
				print()
			else:
				curr+=len(t)
			tid+=1
		sid+=1
	print("Tracks purity: " + str(curr) + " / " + str(tot))

if __name__ == '__main__':

	with open("./resources/ls_feats_list_buffy.pkl", 'rb' ) as fp:
		ls_feats_list = pkl.load(fp)
	print(len(ls_feats_list))
	print(ls_feats_list[0].shape)
	ls_all, ls_mats, ls_all_labels, ls_data = get_vid_list(ls_feats_list)
	del ls_feats_list
	check_track_uniformity(ls_all_labels)
	cnt_faces=0
	for s in ls_all:
		for pll in s:
			cnt_faces+=len(pll)
	print("Total faces available: " + str(cnt_faces))

	nModels, labels_Models = iterate_progressively(ls_all, ls_all_labels, ls_mats, ls_data)
	vm_score, c_score, h_score, fm_score, accu_score = clustering_score(labels_Models)
	print("Labels score: hom:" + str(h_score) + ", compl:" + str(c_score) + ", vm:" + str(vm_score) + ", fm:" + str(fm_score) + ", accu:" + str(accu_score))

	print("Done.")

