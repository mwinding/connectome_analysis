# object for analysing hit_histograms from cascades run using TraverseDispatcher
import numpy as np
import pandas as pd
import sys
sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
from connectome_tools.process_matrix import Promat

class Cascade_Analyzer:
    def __init__(self, hit_hist, mg, pairs):
        self.hit_hist = hit_hist
        self.mg = mg
        self.pairs = pairs
        self.skid_hit_hist = pd.DataFrame(hit_hist, index = mg.meta.index) # convert indices to skids

    def get_hit_hist(self):
        return(self.hit_hist)

    def set_pairs(self, pairs):
        self.pairs = pairs

    def index_to_skid(self, index):
        return(self.mg.meta.iloc[index, :].name)

    def skid_to_index(self, skid):
        index_match = np.where(self.mg.meta.index == skid)[0]
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

class Celltype:
    def __init__(self, name, skids):
        self.name = name
        self.skids = skids

    def get_name(self):
        return(self.name)

    def get_skids(self):
        return(self.skids)

class Celltype_Analyzer:
    def __init__(self, list_Celltypes):
        self.Celltypes = list_Celltypes
        self.num = len(list_Celltypes) # how many cell types
        self.known_types = [] 

    def add_celltype(self, Celltype):
        self.Celltypes = self.Celltypes + Celltype
        self.num += 1

    def add_known_celltypes(self, list_Celltypes):
        self.known_types = list_Celltypes

    # determine membership similarity (intersection over union) between all pair-wise combinations of celltypes
    def compare_membership(self):
        iou_matrix = np.zeros((len(self.Celltypes), len(self.Celltypes)))

        for i in range(len(self.Celltypes)):
            for j in range(len(self.Celltypes)):
                if(len(np.union1d(self.Celltypes[i].skids, self.Celltypes[j].skids)) > 0):
                    intersection = len(np.intersect1d(self.Celltypes[i].get_skids(), self.Celltypes[j].get_skids()))
                    union = len(np.union1d(self.Celltypes[i].get_skids(), self.Celltypes[j].get_skids()))
                    iou_matrix[i, j] = intersection/union

        iou_matrix = pd.DataFrame(iou_matrix, index = [f'{x.get_name()} ({len(x.get_skids())})' for x in self.Celltypes], 
                                            columns = [f'{x.get_name()}' for x in self.Celltypes])

        return(iou_matrix)
        
                    


    


