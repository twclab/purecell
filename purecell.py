import numpy as np
import pandas as pd
import scipy
import scanpy as sp
import math
import heapq
from sklearn.metrics.pairwise import euclidean_distances

class FacilityLocation:

    def __init__(self, D, V, alpha=1.):
        self.D = D
        self.curVal = 0
        self.curMax = np.zeros(len(D))
        self.gains = []
        self.alpha = alpha
        self.f_norm = self.alpha / self.f_norm(V)
        self.norm = 1. / self.inc(V, [])

    def f_norm(self, sset):
        return self.D[:, sset].max(axis=1).sum()

    def inc(self, sset, ndx):
        if len(sset + [ndx]) > 1:
            if not ndx:
                return math.log(1 + self.alpha * 1)
            return self.norm * math.log(1 + self.f_norm * np.maximum(self.curMax, self.D[:, ndx]).sum()) - self.curVal
        else:
            return self.norm * math.log(1 + self.f_norm * self.D[:, ndx].sum()) - self.curVal

    def add(self, sset, ndx):
        cur_old = self.curVal
        if len(sset + [ndx]) > 1:
            self.curMax = np.maximum(self.curMax, self.D[:, ndx])
        else:
            self.curMax = self.D[:, ndx]
        self.curVal = self.norm * math.log(1 + self.f_norm * self.curMax.sum())
        self.gains.extend([self.curVal - cur_old])
        return self.curVal

class PureCell:
    
    def __init__(self,D,batch_id,label_id,n_neighbors,n_pcs):
        self.D = D.copy()
        self.n_pcs = n_pcs
        self.n_neighbors = n_neighbors
        self.indices = np.array(D.obs.index)
        self.D.obs.reset_index(drop=True,inplace=True)
        self.batch_id = batch_id
        self.label_id = label_id

    def heappush_max(self,heap, item):
        heap.append(item)
        heapq._siftdown_max(heap, 0, len(heap)-1)

    def heappop_max(self,heap):
        lastelt = heap.pop()
        if heap:
            returnitem = heap[0]
            heap[0] = lastelt
            heapq._siftup_max(heap, 0)
            return returnitem
        return lastelt

    def lazy_greedy_heap(self,F, V, B):
        curVal = 0
        sset = []
        vals = []
        order = []
        heapq._heapify_max(order)
        [self.heappush_max(order, (F.inc(sset, index), index)) for index in V]

        while order and len(sset) < B:
            el = self.heappop_max(order)
            improv = F.inc(sset, el[1])
            if improv >= 0:
                if not order:
                    curVal = F.add(sset, el[1])
                    sset.append(el[1])
                    vals.append(curVal)
                else:
                    top = self.heappop_max(order)
                    if improv >= top[0]:
                        curVal = F.add(sset, el[1])
                        sset.append(el[1])
                        vals.append(curVal)
                    else:
                        self.heappush_max(order, (improv, el[1]))
                    self.heappush_max(order, top)
        return np.array(sset), np.array(vals)
        
    def embed(self):
        sp.tl.pca(self.D, svd_solver='arpack',n_comps=self.n_pcs)
        sp.pp.neighbors(self.D, n_neighbors=self.n_neighbors)
    
    def get_candidates(self):
        nodes = self.D.obs.index
        graph = self.D.uns['neighbors']['connectivities']
        self.candidates = {}
        for node in nodes:
            label = self.D.obs[self.label_id][node]
            batch = self.D.obs[self.batch_id][node]
            this = 0; other = 0
            if label not in self.candidates:
                self.candidates[label] = []
            neighbors = list(graph.getrow(int(node)).nonzero()[1])
            for neighbor in neighbors:
                batch_ =  self.D.obs[self.batch_id][neighbor]
                label_ =  self.D.obs[self.label_id][neighbor]
                if batch_ != batch: other+=1
                else: this+=1
            if this>0 and other>0:
                self.candidates[label].append(node)
        
    def get_coreset(self,th):
        ordered_indices = []; ordered_nodes = []; similarities=[];node_cuts=[]
        for label in self.candidates:
            nodes = np.array(self.candidates[label])   
            x = self.D.obsm['X_pca'][nodes,:]
            S = self.get_similarities(x)
            indices = list(range(len(S)))
            F = FacilityLocation(S,indices)
            order,sim = self.lazy_greedy_heap(F, indices, len(indices))
            ordered_nodes.append(nodes[order])
            ordered_indices.append(self.indices[ordered_nodes[-1]])
            similarities.append(sim)
            for sim in similarities:
                  for ii in range(len(sim)):
                        if sim[ii]>th:
                            node_cuts.append(ii)
                            break
        return ordered_indices,ordered_nodes,similarities,node_cuts        
    
    def get_similarities(self,x):
        st = np.std(x)
        S = euclidean_distances(x)
        S = -1*(S*S)/2/st/st
        S = np.exp(S)/np.power(np.sqrt(2*np.pi)*st,x.shape[1])
        return S
        
    def run(self,th):
        self.embed()
        self.get_candidates()
        return self.get_coreset(th)