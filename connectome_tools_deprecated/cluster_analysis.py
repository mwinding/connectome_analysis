# object for analysing hit_histograms from cascades run using TraverseDispatcher

import numpy as np
import pandas as pd
import sys
import pymaid as pymaid
from pymaid_creds import url, name, password, token
rm = pymaid.CatmaidInstance(url, token, name, password)

import connectome_tools.process_matrix as pm
import connectome_tools.cascade_analysis as casc
import connectome_tools.celltype as ct

class Analyze_Cluster():

    # cluster_lvl should be integer between 0 and max levels of cluster hierarchy
    # meta_data_path is the path to a meta_data file in 'data/graphs/'; contains cluster and sort information
    def __init__(self, cluster_lvl, meta_data_path = 'data/graphs/meta_data.csv', skids = pymaid.get_skids_by_annotation('mw brain paper clustered neurons')):

        self.meta_data = pd.read_csv(meta_data_path, index_col = 0, header = 0) # load meta_data file
        self.skids = skids

        # determine where neurons are in the signal from sensory -> descending neurons
        # determined using iterative random walks
        self.cluster_order, self.cluster_df = self.cluster_order(cluster_lvl = cluster_lvl)
        self.cluster_cta = ct.Celltype_Analyzer([ct.Celltype(self.cluster_order[i], skids) for i, skids in enumerate(list(self.cluster_df.skids))])

    def cluster_order(self, cluster_lvl):

        brain_clustered = self.skids

        meta_data_df = self.meta_data.copy()
        meta_data_df['skid']=meta_data_df.index

        cluster_df = pd.DataFrame(list(meta_data_df.groupby(f'dc_level_{cluster_lvl}_n_components=10_min_split=32')['skid']), columns=['cluster', 'skids'])
        cluster_df['skids'] = [x.values for x in cluster_df.skids]
        cluster_df['sum_walk_sort'] = [np.nanmean(x[1].values) for x in list(meta_data_df.groupby(f'dc_level_{cluster_lvl}_n_components=10_min_split=32')['sum_walk_sort'])]
        cluster_df.sort_values(by='sum_walk_sort', inplace=True)
        cluster_df.reset_index(inplace=True, drop=True)

        # returns cluster order and clusters dataframe (with order, skids, walk_sort values)
        return(list(cluster_df.cluster), cluster_df) 

    def ff_fb_cascades(self, adj, p, max_hops, n_init):

        skids_list = list(self.cluster_df.skids)
        source_names = list(self.cluster_df.cluster)
        stop_skids = []
        simultaneous = True
        hit_hists_list = casc.Cascade_Analyzer.run_cascades_parallel(source_skids_list = skids_list, source_names = source_names, stop_skids=stop_skids,
                                                                    adj=adj, p=p, max_hops=max_hops, n_init=n_init, simultaneous=simultaneous)
        return(hit_hists_list)
        
    def all_ff_fb_df(self, cascs_list, normalize='visits'):

        rows = []
        for i, casc_analyzer in enumerate(cascs_list):
            precounts = len(self.cluster_df.skids[i])
            casc_row = casc_analyzer.cascades_in_celltypes(cta=self.cluster_cta, hops=4, start_hop=0, normalize=normalize, pre_counts=precounts)
            rows.append(casc_row)

        ff_fb_df = pd.concat(rows, axis=1)
        ff_fb_df.drop(columns='neuropil', inplace=True)
        ff_fb_df.columns = self.cluster_order
        ff_fb_df.index = self.cluster_order
        return(ff_fb_df)            
        



    
