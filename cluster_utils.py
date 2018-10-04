from BaseClusters import BaseClusters

class ClustersShots(BaseClusters):
    #Here the final clusters will be the actual characterwise clusters
    
    def update_cluster(self, ls_data, jhat, kstar):
        if jhat < len(self.clusters):
            self.clusters[jhat].append(ls_data[kstar])
        else:
            self.clusters.append([ls_data[kstar]])

    def build_matrices(self, ls_data, ls_inds):
        
        overlap_mat = np.zeros((len(self.clusters), len(ls_inds)))
        w_mat = np.ones((len(self.clusters), len(ls_inds)))
        
        for j in range(0,len(self.clusters)):		#ls no.
            for k in range(0,len(ls_inds)): 	#ls_data no
                overlap_mat[j,k] = overlap_in_percent(self.clusters[j][-1].bbox, ls_data[ls_inds[k]].bbox)
                dfeat = euc_dist_sq(self.clusters[j][-1].feat, ls_data[ls_inds[k]].feat)
                if (dfeat > self.featThresh):
                    w_mat[j,k] = 0.0
        return overlap_mat, w_mat, self.qt_mat
    
    def update_matrices(self, overlap_mat, w_mat, ls_data, ls_inds, jhat):
        overlap_mat2 = np.zeros((len(self.clusters), overlap_mat.shape[1]))
        overlap_mat2[:overlap_mat.shape[0], :overlap_mat.shape[1]] = overlap_mat

        w_mat2 = np.ones((len(self.clusters), w_mat.shape[1]))
        w_mat2[:w_mat.shape[0], :w_mat.shape[1]] = w_mat

        for kk in range(0,len(ls_inds)):
            overlap_mat2[jhat, kk] = overlap_in_percent(self.clusters[jhat][-1].bbox, ls_data[ls_inds[kk]].bbox)
            dfeat = euc_dist_sq(self.clusters[jhat][-1].feat, ls_data[ls_inds[kk]].feat)
            if (dfeat > self.featThresh):
                w_mat2[jhat,kk] = 0.0
        overlap_mat = overlap_mat2
        featdist_mat = featdist_mat2
        w_mat = w_mat2
        return overlap_mat, w_mat


class ClustersTracks(BaseClusters):
    #Here the final clusters will be the facetracks
    
    featThresh = -1
    def __init__(self, simThresh, featThresh):
        self.simThresh = simThresh
        self.featThresh = featThresh
        
    def update_cluster(self, ls_data, jhat, kstar):
        if jhat < len(self.clusters):
            self.clusters[jhat].append(ls_data[kstar])
        else:
            self.clusters.append([ls_data[kstar]])

    def build_matrices(self, ls_data, ls_inds):
        
        overlap_mat = np.zeros((len(self.clusters), len(ls_inds)))
        w_mat = np.ones((len(self.clusters), len(ls_inds)))
        
        for j in range(0,len(self.clusters)):		#ls no.
            for k in range(0,len(ls_inds)): 	#ls_data no
                overlap_mat[j,k] = overlap_in_percent(self.clusters[j][-1].bbox, ls_data[ls_inds[k]].bbox)
                dfeat = euc_dist_sq(self.clusters[j][-1].feat, ls_data[ls_inds[k]].feat)
                if (dfeat > self.featThresh):
                    w_mat[j,k] = 0.0
        return overlap_mat, w_mat, qt_mat
    
    def update_matrices(self, overlap_mat, w_mat, ls_data, ls_inds, jhat):
        overlap_mat2 = np.zeros((len(self.clusters), overlap_mat.shape[1]))
        overlap_mat2[:overlap_mat.shape[0], :overlap_mat.shape[1]] = overlap_mat

        w_mat2 = np.ones((len(self.clusters), w_mat.shape[1]))
        w_mat2[:w_mat.shape[0], :w_mat.shape[1]] = w_mat

        for kk in range(0,len(ls_inds)):
            overlap_mat2[jhat, kk] = overlap_in_percent(self.clusters[jhat][-1].bbox, ls_data[ls_inds[kk]].bbox)
            dfeat = euc_dist_sq(self.clusters[jhat][-1].feat, ls_data[ls_inds[kk]].feat)
            if (dfeat > self.featThresh):
                w_mat2[jhat,kk] = 0.0
        overlap_mat = overlap_mat2
        featdist_mat = featdist_mat2
        w_mat = w_mat2
        return overlap_mat, w_mat