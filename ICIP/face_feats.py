import numpy as np
import time
import cv2
import itertools
import os
import dlib
import numpy as np
import openface

fileDir = os.path.dirname("/home/ankuprk/openface/")
modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
dlibFacePredictor = dlibModelDir + "/shape_predictor_68_face_landmarks.dat"
networkModel = openfaceModelDir + "/nn4.small2.v1.t7"
imgDim=96

def initialize_deep_models():
	start = time.time()
	align = openface.AlignDlib(dlibFacePredictor)
	net = openface.TorchNeuralNet(networkModel, imgDim)
	#if args.verbose:
	print("Loading the dlib and OpenFace models took {} seconds.".format(
	    time.time() - start))
	return align, net

def get_deep_features(lX, align, net):
	L = len(lX)
	lsfeats = []
	start = time.time()	
	for i in range(L):
		alignedFace = align.align(imgDim, lX[i], dlib.rectangle(0, 0,lX[i].shape[1], lX[i].shape[0]),
		                          landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
		if alignedFace is None:
		    print("Unable to align image...")
		rep = net.forward(alignedFace)
		lsfeats.append(rep)
		#print(i)
	npfeats = np.array(lsfeats)
	print("OpenFace feature extraction took "+str(time.time() - start) + " seconds for " + str(L) +" faces.")

	return npfeats

def get_deep_feature(X, align, net):
	alignedFace = align.align(imgDim, X, dlib.rectangle(0, 0,X.shape[1], X.shape[0]),
	                          landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
	if alignedFace is None:
	    print("Unable to align image...")
	rep = net.forward(alignedFace)
	return np.array(rep)


uniformPattern59=[    
	     1,   2,   3,   4,   5,   0,   6,   7,   8,   0,   0,   0,   9,   0,  10,  11,
		12,   0,   0,   0,   0,   0,   0,   0,  13,   0,   0,   0,  14,   0,  15,  16,
		17,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		18,   0,   0,   0,   0,   0,   0,   0,  19,   0,   0,   0,  20,   0,  21,  22,
		23,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		25,   0,   0,   0,   0,   0,   0,   0,  26,   0,   0,   0,  27,   0,  28,  29,
		30,  31,   0,  32,   0,   0,   0,  33,   0,   0,   0,   0,   0,   0,   0,  34,
		0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35,
		0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  36,
		37,  38,   0,  39,   0,   0,   0,  40,   0,   0,   0,   0,   0,   0,   0,  41,
		0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  42,
		43,  44,   0,  45,   0,   0,   0,  46,   0,   0,   0,   0,   0,   0,   0,  47,
		48,  49,   0,  50,   0,   0,   0,  51,  52,  53,   0,  54,  55,  56,  57,  58];

def get_LBP_word(X):
	N = X.shape[0]
	M = X.shape[1]
	Xcurr = np.zeros(59,dtype=float)
	#get the features
	for i in range(N):
		for j in range(M):
			Xcurr[uniformPattern59[X[i,j]]]+=1.0/(N*M);
	return Xcurr

def get_HOG_word(X):
	N = X.shape[0]
	M = X.shape[1]
	Xcurr = np.zeros(8,dtype=float)
	#get the features
#	print("shape::" + str(X.shape))
	X_grad_ls = np.gradient(X)
	X_cmplx = X_grad_ls[1] + 1j*X_grad_ls[0]
	X_mag = np.abs(X_cmplx)
	X_angle = np.angle(X_cmplx, deg=True) + 180.0
	ls = [0, 45, 90, 135, 180, 225, 270, 315, 360]

	tot = 0
	for i in range(N):
		for j in range(M):
			if(X_mag[i,j]!=0):
				ind1 = int(X_angle[i,j]/45) 
				ind2 = ind1 + 1
				alfa = abs(X_angle[i,j] - ls[ind1])/45.0
				Xcurr[ind1%8]+=X_mag[i,j]*alfa
				Xcurr[ind2%8]+=X_mag[i,j]*(1-alfa)
				tot+=X_mag[i,j]
	X_mag = Xcurr/tot

	return Xcurr


def get_LBP_features(lX, n_grid):
	L = len(lX)

	lsX2 = []
	print lX[0].shape

	for i in range(L):
		WIDTH = lX[0].shape[1]
		HEIGHT = lX[0].shape[0]

		ls = []
		for j in range(n_grid):
			for k in range(n_grid):
				j_i = int((WIDTH*j*1.0)/n_grid)
				j_f = int((WIDTH*(j+1)*1.0)/n_grid)
				k_i = int((HEIGHT*k*1.0)/n_grid)
				k_f = int((HEIGHT*(k+1)*1.0)/n_grid)
				Xcurr = get_LBP_word((lX[i])[j_i:j_f, k_i:k_f]);
				ls.append(Xcurr)

		lsX2.append(np.array(ls).flatten())
	X2 = np.array(lsX2)

	return X2

def get_HOG_features(lX, n_grid):
	L = len(lX)

	lsX2 = []
	print lX[0].shape

	for i in range(L):
		WIDTH = lX[0].shape[1]
		HEIGHT = lX[0].shape[0]
		ls = []
		for j in range(n_grid):
			for k in range(n_grid):
				j_i = int((WIDTH*j*1.0)/n_grid)
				j_f = int((WIDTH*(j+1)*1.0)/n_grid)
				k_i = int((HEIGHT*k*1.0)/n_grid)
				k_f = int((HEIGHT*(k+1)*1.0)/n_grid)
				Xcurr = get_HOG_word((lX[i])[j_i:j_f, k_i:k_f]);
				ls.append(Xcurr)

		lsX2.append(np.array(ls).flatten())
	X2 = np.array(lsX2)

	return X2


def get_LBP_features_color(lX, n_grid):
	L = len(lX)

	lsX2 = []
	print lX[0].shape

	for i in range(L):
		WIDTH = lX[0].shape[1]
		HEIGHT = lX[0].shape[0]

		ls = []
		for channelid in range(3):
			for j in range(n_grid):
				for k in range(n_grid):
					j_i = int((WIDTH*j*1.0)/n_grid)
					j_f = int((WIDTH*(j+1)*1.0)/n_grid)
					k_i = int((HEIGHT*k*1.0)/n_grid)
					k_f = int((HEIGHT*(k+1)*1.0)/n_grid)
					Xcurr = get_LBP_word((lX[i])[j_i:j_f, k_i:k_f, channelid]);
					ls.append(Xcurr)
		lsX2.append(np.array(ls).flatten())
	X2 = np.array(lsX2)

	return X2

def get_HOG_features_color(lX, n_grid):
	L = len(lX)

	lsX2 = []

	for i in range(L):
#		print lX[i].shape
		WIDTH = lX[i].shape[1]
		HEIGHT = lX[i].shape[0]
		ls = []
		print("HOG_feats_color_id: " + str(i) +"/" + str(L-1))
		for channelid in range(3):
			for j in range(n_grid):
				for k in range(n_grid):
					j_i = int((WIDTH*j*1.0)/n_grid)
					j_f = int((WIDTH*(j+1)*1.0)/n_grid)
					k_i = int((HEIGHT*k*1.0)/n_grid)
					k_f = int((HEIGHT*(k+1)*1.0)/n_grid)
					# print("(" + str(j_i) + ", " + str(j_f) + ", " + str(k_i) + ", " + str(k_f) + ")")
					#print("\t("+ str(j) + "," + str(k) +") size: " + str(j_f-j_i) + ", " + str(k_f-k_i) + ", " + str(channelid))
					# print(lX[i].shape)
					# XXX = (lX[i])[j_i:j_f, k_i:k_f, channelid]
					# print("arr: " + str(XXX.shape))
					Xcurr = get_HOG_word((lX[i])[j_i:j_f, k_i:k_f, channelid]);
					ls.append(Xcurr)
		lsX2.append(np.array(ls).flatten())
	X2 = np.array(lsX2)

	return X2

