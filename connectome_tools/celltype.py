# object for analysing and grouping celltypes

import numpy as np
import pandas as pd
import sys
import pymaid
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import itertools
import connectome_tools.process_matrix as pm

from upsetplot import plot
from upsetplot import from_contents
from upsetplot import from_memberships
import connectome_tools.cluster_analysis as clust

class Celltype:
    def __init__(self, name, skids, color=[]):
        self.name = name
        self.skids = list(np.unique(skids))
        if(color!=[]):
            self.color = color

    def get_name(self):
        return(self.name)

    def get_skids(self):
        return(self.skids)

    def get_color(self):
        return(self.color)

    # plots memberships of celltype in a list of other celltypes
    def plot_cell_type_memberships(self, celltypes): # list of Celltype objects
        celltype_colors = [x.get_color() for x in celltypes] + ['tab:gray']

        ct_analyzer = Celltype_Analyzer([self])
        ct_analyzer.set_known_types(celltypes)
        memberships = ct_analyzer.memberships()

        # plot memberships
        ind = np.arange(0, len(ct_analyzer.Celltypes))
        plt.bar(ind, memberships.iloc[0], color=celltype_colors[0])
        bottom = memberships.iloc[0]
        for i in range(1, len(memberships.index)):
            plt.bar(ind, memberships.iloc[i], bottom = bottom, color=celltype_colors[i])
            bottom = bottom + memberships.iloc[i]

    def identify_LNs(self, threshold, summed_adj, adj_aa, input_skids, outputs, exclude, pairs = pm.Promat.get_pairs(), sort = True):
        mat = summed_adj.loc[np.intersect1d(summed_adj.index, self.skids), np.intersect1d(summed_adj.index, self.skids)]
        mat = mat.sum(axis=1)

        mat_axon = adj_aa.loc[np.intersect1d(adj_aa.index, self.skids), np.intersect1d(adj_aa.index, input_skids)]
        mat_axon = mat_axon.sum(axis=1)

        # convert to % outputs
        skid_percent_output = []
        for skid in self.skids:
            skid_output = 0
            output = sum(outputs.loc[skid, :])
            if(output != 0):
                if(skid in mat.index):
                    skid_output = skid_output + mat.loc[skid]/output
                if(skid in mat_axon.index):
                    skid_output = skid_output + mat_axon.loc[skid]/output

            skid_percent_output.append([skid, skid_output])

        skid_percent_output = pm.Promat.convert_df_to_pairwise(pd.DataFrame(skid_percent_output, columns=['skid', 'percent_output_intragroup']).set_index('skid'))

        # identify neurons with >=50% output within group (or axoaxonic onto input neurons to group)
        LNs = skid_percent_output.groupby('pair_id').sum()      
        LNs = LNs[np.array([x for sublist in (LNs>=threshold*2).values for x in sublist])]
        LNs = list(LNs.index) # identify pair_ids of all neurons pairs/nonpaired over threshold
        LNs = [list(skid_percent_output.loc[(slice(None), skid), :].index) for skid in LNs] # pull all left/right pairs or just nonpaired neurons
        LNs = [x[2] for sublist in LNs for x in sublist]
        LNs = list(np.setdiff1d(LNs, exclude)) # don't count neurons flagged as excludes: for example, MBONs/MBINs/RGNs probably shouldn't be LNs
        return(LNs, skid_percent_output)


class Celltype_Analyzer:
    def __init__(self, list_Celltypes, adj=[], skids=[]):
        self.Celltypes = list_Celltypes
        self.celltype_names = [celltype.get_name() for celltype in self.Celltypes]
        self.num = len(list_Celltypes) # how many cell types
        self.known_types = []
        self.known_types_names = []
        self.adj = adj

        if(len(skids)>0):
            self.skids = skids
        if(len(skids)==0):
            self.skids = list(np.unique([x for sublist in self.Celltypes for x in sublist.get_skids()]))

        self.adj_df = []

    def get_celltype_names(self):
        return self.celltype_names

    def generate_adj(self):
        # adjacency matrix only between assigned cell types
        adj_df = pd.DataFrame(self.adj, index = self.skids, columns = self.skids)
        skids = [skid for celltype in self.Celltypes for skid in celltype.get_skids()]
        adj_df = adj_df.loc[skids, skids]

        # generating multiindex for adjacency matrix df
        index_df = pd.DataFrame([[celltype.get_name(), skid] for celltype in self.Celltypes for skid in celltype.get_skids()], 
                                columns = ['celltype', 'skid'])
        index = pd.MultiIndex.from_frame(index_df)

        # add multiindex to both rows and columns
        adj_df.index = index
        adj_df.columns = index

        self.adj_df = adj_df

    def add_celltype(self, Celltype):
        self.Celltypes = self.Celltypes + Celltype
        self.num += 1
        self.generate_adj()

    def set_known_types(self, list_Celltypes, unknown=True):
        if(unknown==True):
            unknown_skids = np.setdiff1d(self.skids, np.unique([skid for celltype in list_Celltypes for skid in celltype.get_skids()]))
            unknown_type = [Celltype('unknown', unknown_skids, 'tab:gray')]
            list_Celltypes = list_Celltypes + unknown_type
            
        self.known_types = list_Celltypes
        self.known_types_names = [celltype.get_name() for celltype in list_Celltypes]

    def get_known_types(self):
        return(self.known_types)

    # determine membership similarity (intersection over union) between all pair-wise combinations of celltypes
    def compare_membership(self, sim_type):
        iou_matrix = np.zeros((len(self.Celltypes), len(self.Celltypes)))

        for i in range(len(self.Celltypes)):
            for j in range(len(self.Celltypes)):
                if(len(np.union1d(self.Celltypes[i].skids, self.Celltypes[j].skids)) > 0):
                    if(sim_type=='iou'):
                        intersection = len(np.intersect1d(self.Celltypes[i].get_skids(), self.Celltypes[j].get_skids()))
                        union = len(np.union1d(self.Celltypes[i].get_skids(), self.Celltypes[j].get_skids()))
                        calculation = intersection/union
                    
                    if(sim_type=='dice'):
                        intersection = len(np.intersect1d(self.Celltypes[i].get_skids(), self.Celltypes[j].get_skids()))
                        diff1 = len(np.setdiff1d(self.Celltypes[i].get_skids(), self.Celltypes[j].get_skids()))
                        diff2 = len(np.setdiff1d(self.Celltypes[j].get_skids(), self.Celltypes[i].get_skids()))
                        calculation = intersection*2/(intersection*2 + diff1 + diff2)
                    
                    if(sim_type=='cosine'):
                            unique_skids = list(np.unique(list(self.Celltypes[i].get_skids()) + list(self.Celltypes[j].get_skids())))
                            data = pd.DataFrame(np.zeros(shape=(2, len(unique_skids))), columns = unique_skids, index = [i,j])
                            
                            for k in range(len(data.columns)):
                                if(data.columns[k] in self.Celltypes[i].get_skids()):
                                    data.iloc[0,k] = 1
                                if(data.columns[k] in self.Celltypes[j].get_skids()):
                                    data.iloc[1,k] = 1

                            a = list(data.iloc[0, :])
                            b = list(data.iloc[1, :])

                            dot = np.dot(a, b)
                            norma = np.linalg.norm(a)
                            normb = np.linalg.norm(b)
                            calculation = dot / (norma * normb)

                    iou_matrix[i, j] = calculation

        iou_matrix = pd.DataFrame(iou_matrix, index = [f'{x.get_name()} ({len(x.get_skids())})' for x in self.Celltypes], 
                                            columns = [f'{x.get_name()}' for x in self.Celltypes])

        return(iou_matrix)

    # calculate fraction of neurons in each cell type that have previously known cell type annotations
    def memberships(self, by_celltype=True, raw_num=False): # raw_num=True outputs number of neurons in each category instead of fraction
        fraction_type = np.zeros((len(self.known_types), len(self.Celltypes)))
        for i, knowntype in enumerate(self.known_types):
            for j, celltype in enumerate(self.Celltypes):
                if(by_celltype): # fraction of new cell type in each known category
                    if(raw_num==False):
                        if(len(celltype.get_skids())==0):
                            fraction = 0
                        if(len(celltype.get_skids())>0):
                            fraction = len(np.intersect1d(celltype.get_skids(), knowntype.get_skids()))/len(celltype.get_skids())
                    if(raw_num==True):
                        fraction = len(np.intersect1d(celltype.get_skids(), knowntype.get_skids()))
                    fraction_type[i, j] = fraction
                if(by_celltype==False): # fraction of each known category that is in new cell type
                    fraction = len(np.intersect1d(celltype.get_skids(), knowntype.get_skids()))/len(knowntype.get_skids())
                    fraction_type[i, j] = fraction

        fraction_type = pd.DataFrame(fraction_type, index = self.known_types_names, 
                                    columns = [f'{celltype.get_name()} ({len(celltype.get_skids())})' for celltype in self.Celltypes])
        return(fraction_type)

    def plot_memberships(self, path, figsize):
        memberships = self.memberships()
        celltype_colors = [x.get_color() for x in self.get_known_types()]

        # plot memberships
        ind = [cell_type.get_name() for cell_type in self.Celltypes]
        f, ax = plt.subplots(figsize=figsize)
        plt.bar(ind, memberships.iloc[0, :], color=celltype_colors[0])
        bottom = memberships.iloc[0, :]
        for i in range(1, len(memberships.index)):
            plt.bar(ind, memberships.iloc[i, :], bottom = bottom, color=celltype_colors[i])
            bottom = bottom + memberships.iloc[i, :]
        plt.savefig(path, format='pdf', bbox_inches='tight')

    def connectivtiy(self, celltypes, normalize_pre_num = False, normalize_post_num = False):

        #level0_keys = np.unique(self.adj_df.index.get_level_values(0))
        mat = np.zeros((len(celltypes), len(celltypes)))
        for i, key_i in enumerate(celltypes):
            for j, key_j in enumerate(celltypes):
                if(normalize_pre_num==False & normalize_post_num==False):
                    mat[i, j] = self.adj_df.loc[key_i, key_j].values.sum()
                if(normalize_pre_num==True):
                    mat[i, j] = self.adj_df.loc[key_i, key_j].values.sum()/len(self.adj_df.loc[key_i].index)
                if(normalize_post_num==True):
                    mat[i, j] = self.adj_df.loc[key_i, key_j].values.sum()/len(self.adj_df.loc[key_j].index)
        mat = pd.DataFrame(mat, index = celltypes, columns = celltypes)
        return(mat)

    def upset_members(self, path=None, plot_upset=False):

        celltypes = self.Celltypes

        contents = {} # empty dictionary
        for celltype in celltypes:
            name = celltype.get_name()
            contents[name] = celltype.get_skids()

        data = from_contents(contents)

        if(plot_upset):
            fg = plot(data)
            plt.savefig(f'{path}.pdf', bbox_inches='tight')

        unique_indices = np.unique(data.index)
        cat_types = [Celltype(' + '.join([data.index.names[i] for i, value in enumerate(index) if value==True]), 
                    list(data.loc[index].id)) for index in unique_indices]

        return (cat_types)

    @staticmethod
    def get_skids_from_meta_meta_annotation(meta_meta, split=False):
        meta_annots = pymaid.get_annotated(meta_meta).name
        annot_list = [list(pymaid.get_annotated(meta).name) for meta in meta_annots]
        skids = [list(pymaid.get_skids_by_annotation(annots)) for annots in annot_list]
        if(split==False):
            skids = [x for sublist in skids for x in sublist]
            return(skids)
        if(split==True):
            return(skids, meta_annots)
    
    @staticmethod
    def default_celltypes():
        priority_list = pymaid.get_annotated('mw brain simple priorities').name
        priority_skids = [Celltype_Analyzer.get_skids_from_meta_meta_annotation(priority) for priority in priority_list]

        # made the priority groups exclusive by removing neurons from lists that also in higher priority
        override = priority_skids[0]
        priority_skids_unique = [priority_skids[0]]
        for i in range(1, len(priority_skids)):
            skids_temp = list(np.setdiff1d(priority_skids[i], override))
            priority_skids_unique.append(skids_temp)
            override = override + skids_temp

        # take all 'mw brain simple groups' skids (under 'mw brain simple priorities' meta-annotation)
        #   and remove skids that aren't in the appropriate priority_skids_unique level
        priority_skid_groups = [list(pymaid.get_annotated(meta).name) for meta in priority_list]

        skid_groups = []
        for i in range(0, len(priority_skid_groups)):
            group = []
            for j in range(0, len(priority_skid_groups[i])):
                skids_temp = pymaid.get_skids_by_annotation(pymaid.get_annotated(priority_skid_groups[i][j]).name)
                skids_temp = list(np.intersect1d(skids_temp, priority_skids_unique[i])) # make sure skid in subgroup is set in correct priority list
                group.append(skids_temp)
            skid_groups.append(group)

        # test skid counts for each group
        #[len(x) for sublist in skid_groups for x in sublist]

        # make list of lists of skids + their associated names
        skid_groups = [x for sublist in skid_groups for x in sublist]
        names = [list(pymaid.get_annotated(x).name) for x in priority_list]
        names = [x for sublist in names for x in sublist]
        #names = [x.replace('mw brain ', '') for x in names]
        
        # identify colors
        colors = list(pymaid.get_annotated('mw brain simple colors').name)
        colors_order = [x.name.values[0] for x in list(map(pymaid.get_annotated, colors))] # use order of colors annotation for now
        
        # ordered properly and linked to colors
        groups_sort = [np.where(x==np.array(colors_order))[0][0] for x in names]
        names = [element for _, element in sorted(zip(groups_sort, names))]
        skid_groups = [element for _, element in sorted(zip(groups_sort, skid_groups))]

        names = [x.replace('mw brain ', '') for x in names] #format names
        data = pd.DataFrame(zip(names, skid_groups, colors), columns = ['name', 'skids', 'color'])
        celltype_objs = list(map(lambda x: Celltype(*x), zip(names, skid_groups, colors)))
        return(data, celltype_objs)


def plot_cell_types_cluster(lvl_labels, path):

    _, all_celltypes = Celltype_Analyzer.default_celltypes()
    lvl = clust.Analyze_Cluster('cascades/data/meta-method=color_iso-d=8-bic_ratio=0.95-min_split=32.csv', 'data/meta_data_w_order.csv', lvl_labels)

    all_clusters = [Celltype(lvl.clusters.cluster[i], lvl.clusters.skids[i]) for i in range(0, len(lvl.clusters))]
    cluster_analyze = Celltype_Analyzer(all_clusters)

    cluster_analyze.set_known_types(all_celltypes)
    celltype_colors = [x.get_color() for x in cluster_analyze.get_known_types()]
    memberships = cluster_analyze.memberships()
    memberships = memberships.iloc[[0,1,2,3,4,5,6,7,8,9,10,11,15,12,13,14], :]
    celltype_colors = [celltype_colors[i] for i in [0,1,2,3,4,5,6,7,8,9,10,11,15,12,13,14]]

    ind = np.arange(0, len(cluster_analyze.Celltypes))
    plt.bar(ind, memberships.iloc[0, :], color=celltype_colors[0])
    bottom = memberships.iloc[0, :]
    for i in range(1, len(memberships.index)):
        plt.bar(ind, memberships.iloc[i, :], bottom = bottom, color=celltype_colors[i])
        bottom = bottom + memberships.iloc[i, :]
    plt.savefig(path, format='pdf', bbox_inches='tight')

def plot_marginal_cell_type_cluster(size, particular_cell_type, particular_color, lvl_labels, path):

    # all cell types plot data
    _, all_celltypes = Celltype_Analyzer.default_celltypes()
    lvl = clust.Analyze_Cluster('cascades/data/meta-method=color_iso-d=8-bic_ratio=0.95-min_split=32.csv', 'data/meta_data_w_order.csv', lvl_labels)

    all_clusters = [Celltype(lvl.clusters.cluster[i], lvl.clusters.skids[i]) for i in range(0, len(lvl.clusters))]
    cluster_analyze = Celltype_Analyzer(all_clusters)

    cluster_analyze.set_known_types(all_celltypes)
    celltype_colors = [x.get_color() for x in cluster_analyze.get_known_types()]
    all_memberships = cluster_analyze.memberships()
    all_memberships = all_memberships.iloc[[0,1,2,3,4,5,6,7,8,9,10,11,15,16,12,13,14], :]
    celltype_colors = [celltype_colors[i] for i in [0,1,2,3,4,5,6,7,8,9,10,11,15,16,12,13,14]]
    
    # particular cell type data
    cluster_analyze.set_known_types([particular_cell_type])
    membership = cluster_analyze.memberships()

    # plot
    fig = plt.figure(figsize=size) 
    fig.subplots_adjust(hspace=0.1)
    gs = GridSpec(4, 1)

    ax = fig.add_subplot(gs[0:3, 0])
    ind = np.arange(0, len(cluster_analyze.Celltypes))
    ax.bar(ind, membership.iloc[0, :], color=particular_color)
    ax.set(xlim = (-1, len(ind)), ylim=(0,1), xticks=([]), yticks=([]), title=particular_cell_type.get_name())

    ax = fig.add_subplot(gs[3, 0])
    ind = np.arange(0, len(cluster_analyze.Celltypes))
    ax.bar(ind, all_memberships.iloc[0, :], color=celltype_colors[0])
    bottom = all_memberships.iloc[0, :]
    for i in range(1, len(all_memberships.index)):
        plt.bar(ind, all_memberships.iloc[i, :], bottom = bottom, color=celltype_colors[i])
        bottom = bottom + all_memberships.iloc[i, :]
    ax.set(xlim = (-1, len(ind)), ylim=(0,1), xticks=([]), yticks=([]))
    ax.axis('off')
    ax.axis('off')

    plt.savefig(path, format='pdf', bbox_inches='tight')
