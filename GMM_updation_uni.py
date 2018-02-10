import numpy as np
from math import exp, sqrt
from sklearn.mixture import GaussianMixture as GMM

def EM_using_mono(chunk):
        model = GMM(n_components=1, covariance_type='spherical')
        model.fit(chunk)
        GMM_new = model
        n_argmin=1
        return GMM_new

def merge( g_x, N, a_x, M):

        u = (N*g_x.means_ + float(M)*a_x.means_)/float(N + M)
        d = g_x.means_.shape[1]
        print("d::"+str(d))         
        wt = g_x.weights_ 
        cov_g = g_x.covariances_
        cov_a = a_x.covariances_

        #For Spherical Cov Matrix
        u_g = g_x.means_
        u_a = a_x.means_
        sigMat = (N*cov_g + M*cov_a)/float(N+M) + (N*np.sum(u_g**2) + M*np.sum(u_a**2))/float(N+M) - np.sum(u**2) 
        C = [u, sigMat, wt]
        return C

def modify_gaussian(g_x2, C):
    g_x2.weights_ = C[2]
    g_x2.means_ = C[0]
    g_x2.covariances_ = C[1]
    g_x2.precisions_ = 1/C[1]
    g_x2.precisions_cholesky_ = g_x2.precisions_**0.5
    g_x2.converged_ = True
    return g_x2

def initialize_GMM(chunk):
        a_x = EM_using_mono(chunk)
#        print("comp during init: " + str(n))
        N_samples = chunk.shape[0]
        return N_samples, a_x

def update_GMM(g_x, N_samples, chunk):
        
        M_samples = chunk.shape[0]
#3
        a_x = EM_using_mono(chunk)
        m = g_x.weights_.shape[0] # no of components in original model
        C = merge(g_x, N_samples, a_x, M_samples)
        g_x2 = GMM(n_components=1, covariance_type='spherical')
        g_x2.fit(chunk)
        modify_gaussian(g_x2, C)
        g_x2.converged_ = True
        return N_samples+M_samples, g_x2