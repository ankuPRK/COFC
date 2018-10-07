from BaseClusters import BaseClusters
import numpy as np
import os, cv2
from cofc_utils import overlap_in_percent, euc_dist_sq

class ClustersShots(BaseClusters):
    #Here the final clusters will be the actual characterwise clusters

    def __init__(self, simThresh, savePath="./clusters"):
        super(ClustersShots,self).__init__(simThresh)
        if savePath[-1] !="/":
            savePath+="/"        
        self.savePath = savePath
        if os.path.exists(self.savePath):
            print("Directory already exists! : "+self.savePath)
        else:
            os.mkdir(self.savePath)
            print("Creating directory: "+self.savePath)

    def update_cluster(self, ls_data, jhat, kstar):
        track = ls_data[kstar]
        if jhat < len(self.clusters):
            M = self.clusters[jhat][1]
            mean_feat = self.clusters[jhat][0] * M 
            for f in track:
                mean_feat+=f.feat
            N = len(track)
            mean_feat/=float(M+N)
            self.clusters[jhat] = (mean_feat, N+M)
        else:
            mean_feat = np.zeros((128))
            for f in track:
                mean_feat+=f.feat
            N = len(track)
            mean_feat/=float(N)
            self.clusters.append((mean_feat, N))
        if not os.path.exists(self.savePath+str(jhat)+"/"):
            print("Making new cluster: " + self.savePath + str(jhat)+"/")
            os.mkdir(self.savePath+str(jhat)+"/")
        for j, f in enumerate(track):
            cv2.imwrite(self.savePath+str(jhat)+"/"+str(f.fno)+"_"+str(j)+".png", f.img)

    def build_matrices(self, ls_data, ls_inds):
        
        similarity_mat = np.zeros((len(self.clusters), len(ls_inds)))
        w_mat = np.ones((len(self.clusters), len(ls_inds)))
        
        for j in range(0,len(self.clusters)):		#ls no.
            for k in range(0,len(ls_inds)): 	#ls_data no
                avgd = 0.0
                for l in ls_data[k]:
                    avgd += euc_dist_sq(self.clusters[j][0], l.feat)
                assert len(ls_data[k]) > 0
                avgd /= len(ls_data[k])
                similarity_mat[j,k] = 4.0 - avgd 
        return similarity_mat, w_mat, self.qt_mat
    
    def update_matrices(self, similarity_mat, w_mat, ls_data, ls_inds, jhat):
        similarity_mat2 = np.zeros((len(self.clusters), similarity_mat.shape[1]))
        similarity_mat2[:similarity_mat.shape[0], :similarity_mat.shape[1]] = similarity_mat

        w_mat2 = np.ones((len(self.clusters), w_mat.shape[1]))
        w_mat2[:w_mat.shape[0], :w_mat.shape[1]] = w_mat

        for kk in range(0,len(ls_inds)):
            avgd = 0.0
            kstar = ls_inds[kk]
            for l in ls_data[kstar]:
                avgd += euc_dist_sq(self.clusters[jhat][0], l.feat)
            assert len(ls_data[kstar]) > 0
            avgd /= len(ls_data[kstar])
            similarity_mat2[jhat,kk] = 4.0 - avgd 
        similarity_mat = similarity_mat2
        w_mat = w_mat2
        return similarity_mat, w_mat


class ClustersTracks(BaseClusters):
    #Here the final clusters will be the facetracks
    
    def __init__(self, simThresh, featThresh):
        super(ClustersTracks,self).__init__(simThresh)
        self.featThresh = featThresh
        
    def update_cluster(self, ls_data, jhat, kstar):
        if jhat < len(self.clusters):
            self.clusters[jhat].append(ls_data[kstar])
        else:
            self.clusters.append([ls_data[kstar]])

    def build_matrices(self, ls_data, ls_inds):
        
        similarity_mat = np.zeros((len(self.clusters), len(ls_inds)))
        w_mat = np.ones((len(self.clusters), len(ls_inds)))
        qt_mat = np.eye(len(ls_inds))
        for j in range(0,len(self.clusters)):		#ls no.
            for k in range(0,len(ls_inds)): 	#ls_data no
                similarity_mat[j,k] = overlap_in_percent(self.clusters[j][-1].bbox, ls_data[ls_inds[k]].bbox)
                dfeat = euc_dist_sq(self.clusters[j][-1].feat, ls_data[ls_inds[k]].feat)
                if (dfeat > self.featThresh):
                    w_mat[j,k] = 0.0
        return similarity_mat, w_mat, qt_mat
    
    def update_matrices(self, similarity_mat, w_mat, ls_data, ls_inds, jhat):
        similarity_mat2 = np.zeros((len(self.clusters), similarity_mat.shape[1]))
        similarity_mat2[:similarity_mat.shape[0], :similarity_mat.shape[1]] = similarity_mat

        w_mat2 = np.ones((len(self.clusters), w_mat.shape[1]))
        w_mat2[:w_mat.shape[0], :w_mat.shape[1]] = w_mat

        for kk in range(0,len(ls_inds)):
            similarity_mat2[jhat, kk] = overlap_in_percent(self.clusters[jhat][-1].bbox, ls_data[ls_inds[kk]].bbox)
            dfeat = euc_dist_sq(self.clusters[jhat][-1].feat, ls_data[ls_inds[kk]].feat)
            if (dfeat > self.featThresh):
                w_mat2[jhat,kk] = 0.0
        similarity_mat = similarity_mat2
        w_mat = w_mat2
        return similarity_mat, w_mat
