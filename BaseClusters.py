import numpy as np

class BaseClusters:
    
    clusters = []
    simThresh = -1
    ls_data = []
    def __init__(self, simThresh):
        self.simThresh = simThresh
    
    def update_cluster(self, ls_data, jhat, khat):
        assert True, "implement method update_cluster in the new class" 
        return

    def build_similarity_matrix():
        assert True, "implement method build_similarity_matrix in the new class" 
        return
    
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
    
    def cluster_online(self, ls_data, qt_mat=np.zeros((0,0))):
        ls_cl = []
        ls_inds = np.arange(0, len(ls_data))
        self.qt_mat = qt_mat
        overlap_mat, w_mat, qt_mat = self.build_matrices(ls_data, ls_inds)
        
        while(len(ls_inds)>0):
            newTrack = True
            jhat, khat = -1, -1
            if(len(self.clusters) > 0):
                jhat, khat = np.unravel_index((w_mat*overlap_mat).argmax(), (w_mat*overlap_mat).shape)
                if w_mat[jhat, khat]*overlap_mat[jhat, khat]>=th_overlap:
                    kstar = ls_inds[khat]
                    newTrack = False   
            if newTrack:
                jhat, khat = len(self.clusters), 0
                kstar = ls_inds[khat]
            self.update_cluster(ls_data, jhat, kstar)

            #Recompute D and W for jhat
            if jhat not in ls_cl:
                ls_cl.append(jhat)

            overlap_mat, w_mat = self.update_matrices(overlap_mat, w_mat, ls_data, ls_inds, jhat)
            
            w_mat[jhat,:] = qt_mat[khat,:]

            overlap_mat = np.delete(overlap_mat, khat, axis=1)
            featdist_mat = np.delete(featdist_mat, khat, axis=1)
            w_mat = np.delete(w_mat, khat, axis=1)
            qt_mat = np.delete(qt_mat, khat, axis=0)
            qt_mat = np.delete(qt_mat, khat, axis=1)
            del ls_inds[khat]
        return ls_cl    
