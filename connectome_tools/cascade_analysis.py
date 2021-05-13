# object for analysing hit_histograms from cascades run using TraverseDispatcher
import numpy as np
import pandas as pd
import connectome_tools.process_matrix as pm

import sys
sys.path.append('/Users/mwinding/repos/maggot_models')
from src.traverse import Cascade, to_transmission_matrix
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot
from joblib import Parallel, delayed
from tqdm import tqdm

class Cascade_Analyzer:
    def __init__(self, name, hit_hist, skids_in_hit_hist=True, adj_index=None): # changed mg to adj_index for custom/modified adj matrices
        self.hit_hist = hit_hist
        self.name = name
        if(skids_in_hit_hist):
            self.adj_index = hit_hist.index
            self.skid_hit_hist = hit_hist
        if(skids_in_hit_hist==False):
            self.adj_index = adj_index
            self.skid_hit_hist = pd.DataFrame(hit_hist, index = self.adj_index) # convert indices to skids

        self.pairs = pm.Promat.get_pairs()

    def get_hit_hist(self):
        return(self.hit_hist)

    def get_skid_hit_hist(self):
        return(self.skid_hit_hist)

    def get_name(self):
        return(self.name)

    def set_pairs(self, pairs):
        self.pairs = pairs

    def index_to_skid(self, index):
        return(self.adj_index[index].name)

    def skid_to_index(self, skid):
        index_match = np.where(self.adj_index == skid)[0]
        if(len(index_match)==1):
            return(int(index_match[0]))
        if(len(index_match)!=1):
            print(f'Not one match for skid {skid}!')
            return(False)

    def pairwise_threshold_detail(self, threshold, hops, excluded_skids=False):

        neurons = np.where((self.skid_hit_hist.iloc[:, 1:(hops+1)]).sum(axis=1)>threshold)[0]
        neurons = self.skid_hit_hist.index[neurons]

        # remove particular skids if included
        if(excluded_skids!=False): 
            neurons = np.delete(neurons, excluded_skids)

        neurons_pairs, neurons_unpaired, neurons_nonpaired = pm.Promat.extract_pairs_from_list(neurons, self.pairs)
        return(neurons_pairs, neurons_unpaired, neurons_nonpaired)

    def pairwise_threshold(self, threshold, hops, excluded_skids=False):
        neurons_pairs, neurons_unpaired, neurons_nonpaired = Cascade_Analyzer.pairwise_threshold_detail(self, threshold, hops, excluded_skids)
        skids = np.concatenate([neurons_pairs.leftid, neurons_pairs.rightid, neurons_nonpaired.nonpaired])
        return(skids)

    @staticmethod
    def run_cascade(i, cdispatch):
        return(cdispatch.multistart(start_nodes = i))
        
    @staticmethod
    def run_cascades_parallel(source_skids_list, source_names, stop_skids, adj, p, max_hops, n_init, simultaneous):
        # adj format must be pd.DataFrame with skids for index/columns

        source_indices_list = []
        for skids in source_skids_list:
            indices = np.where([x in skids for x in adj.index])[0]
            source_indices_list.append(indices)

        stop_indices = np.where([x in stop_skids for x in adj.index])[0]

        transition_probs = to_transmission_matrix(adj.values, p)

        cdispatch = TraverseDispatcher(
            Cascade,
            transition_probs,
            stop_nodes = stop_indices,
            max_hops=max_hops,
            allow_loops = False,
            n_init=n_init,
            simultaneous=simultaneous,
        )

        job = Parallel(n_jobs=-1)(delayed(Cascade_Analyzer.run_cascade)(i, cdispatch) for i in source_indices_list)
        data = [Cascade_Analyzer(name=source_names[i], hit_hist=hit_hist, skids_in_hit_hist=False, adj_index=adj.index) for i, hit_hist in enumerate(job)]
        return(data)
