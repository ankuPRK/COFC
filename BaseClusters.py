import numpy as np

class BaseClusters(object):
    
    def __init__(self, simThresh):
        self.clusters = []
        self.ls_data = []
        self.simThresh = simThresh
    
    def get_clusters(self):
        return self.clusters
    
    def get_data(self):
        return self.ls_data
    
    def update_cluster(self, ls_data, jhat, khat):
        assert True, "implement method update_cluster in the new class" 
        return

    def build_similarity_matrix():
        assert True, "implement method build_similarity_matrix in the new class" 
        return
    
    def cluster_online(self, ls_data, qt_mat=np.zeros((0,0))):
        ls_cl = []
        ls_inds = list(range(len(ls_data)))
        self.qt_mat = qt_mat
        similarity_mat, w_mat, qt_mat = self.build_matrices(ls_data, ls_inds)
        
        while(len(ls_inds)>0):
            newTrack = True
            jhat, khat = -1, -1
            if(len(self.clusters) > 0):
                jhat, khat = np.unravel_index((w_mat*similarity_mat).argmax(), (w_mat*similarity_mat).shape)
                if w_mat[jhat, khat]*similarity_mat[jhat, khat]>=self.simThresh:
                    kstar = ls_inds[khat]
                    newTrack = False   
            if newTrack:
                jhat, khat = len(self.clusters), 0
                kstar = ls_inds[khat]
            self.update_cluster(ls_data, jhat, kstar)

            #Recompute D and W for jhat
            if jhat not in ls_cl:
                ls_cl.append(jhat)

            similarity_mat, w_mat = self.update_matrices(similarity_mat, w_mat, ls_data, ls_inds, jhat)
            
            w_mat[jhat,:] = qt_mat[khat,:]

            similarity_mat = np.delete(similarity_mat, khat, axis=1)
            w_mat = np.delete(w_mat, khat, axis=1)
            qt_mat = np.delete(qt_mat, khat, axis=0)
            qt_mat = np.delete(qt_mat, khat, axis=1)
            del ls_inds[khat]
        return ls_cl    
