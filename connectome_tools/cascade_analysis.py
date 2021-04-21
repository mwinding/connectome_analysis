# object for analysing hit_histograms from cascades run using TraverseDispatcher
import numpy as np
import pandas as pd
import sys
sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
from connectome_tools.process_matrix import Promat

class Cascade_Analyzer:
    def __init__(self, hit_hist, adj_index, pairs): # changed mg to adj_index for custom/modified adj matrices
        self.hit_hist = hit_hist
        self.adj_index = adj_index
        self.pairs = pairs
        self.skid_hit_hist = pd.DataFrame(hit_hist, index = self.adj_index) # convert indices to skids

    def get_hit_hist(self):
        return(self.hit_hist)

    def get_skid_hit_hist(self):
        return(self.skid_hit_hist)

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

        neurons_pairs, neurons_unpaired, neurons_nonpaired = Promat.extract_pairs_from_list(neurons, self.pairs)
        return(neurons_pairs, neurons_unpaired, neurons_nonpaired)

    def pairwise_threshold(self, threshold, hops, excluded_skids=False):
        neurons_pairs, neurons_unpaired, neurons_nonpaired = Cascade_Analyzer.pairwise_threshold_detail(self, threshold, hops, excluded_skids)
        skids = np.concatenate([neurons_pairs.leftid, neurons_pairs.rightid, neurons_nonpaired.nonpaired])
        return(skids)
