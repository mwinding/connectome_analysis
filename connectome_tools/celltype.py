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
    def __init__(self, name, skids, color=None):
        self.name = name
        self.skids = list(np.unique(skids))
        if(color!=None):
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
    
    def identify_in_out_LNs(self, threshold, summed_adj, outputs, inputs, exclude, pairs = pm.Promat.get_pairs(), sort = True):

        mat_output = summed_adj.loc[:, np.intersect1d(summed_adj.index, self.skids)]
        mat_output = mat_output.sum(axis=1)
        mat_input = summed_adj.loc[np.intersect1d(summed_adj.index, self.skids), :]
        mat_input = mat_input.sum(axis=0)
        
        # convert to % outputs
        skid_percent_in_out = []
        for skid in summed_adj.index:
            skid_output = 0
            output = sum(outputs.loc[skid, :])
            if(output != 0):
                if(skid in mat_output.index):
                    skid_output = skid_output + mat_output.loc[skid]/output

            skid_input = 0
            input_ = sum(inputs.loc[skid, :])
            if(input_ != 0):
                if(skid in mat_input.index):
                    skid_input = skid_input + mat_input.loc[skid]/input_
            
            skid_percent_in_out.append([skid, skid_input, skid_output])

        skid_percent_in_out = pm.Promat.convert_df_to_pairwise(pd.DataFrame(skid_percent_in_out, columns=['skid', 'percent_input_from_group', 'percent_output_to_group']).set_index('skid'))

        # identify neurons with >=50% output within group (or axoaxonic onto input neurons to group)
        LNs = skid_percent_in_out.groupby('pair_id').sum()      
        LNs = LNs[((LNs>=threshold*2).sum(axis=1)==2).values]
        LNs = list(LNs.index) # identify pair_ids of all neurons pairs/nonpaired over threshold
        LNs = [list(skid_percent_in_out.loc[(slice(None), skid), :].index) for skid in LNs] # pull all left/right pairs or just nonpaired neurons
        LNs = [x[2] for sublist in LNs for x in sublist]
        LNs = list(np.setdiff1d(LNs, exclude)) # don't count neurons flagged as excludes: for example, MBONs/MBINs/RGNs probably shouldn't be LNs
        return(LNs, skid_percent_in_out)


class Celltype_Analyzer:
    def __init__(self, list_Celltypes, adj=[]):
        self.Celltypes = list_Celltypes
        self.celltype_names = [celltype.get_name() for celltype in self.Celltypes]
        self.num = len(list_Celltypes) # how many cell types
        self.known_types = []
        self.known_types_names = []
        self.adj = adj
        self.skids = [x for sublist in [celltype.get_skids() for celltype in list_Celltypes] for x in sublist]

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
        
        if(raw_num==True):
            fraction_type = fraction_type.astype(int)
        
        return(fraction_type)

    def plot_memberships(self, path, figsize, rotated_labels = True, raw_num = False, memberships=None, ylim=None):
        if(type(memberships)!=pd.DataFrame):
            memberships = self.memberships(raw_num=raw_num)
        celltype_colors = [x.get_color() for x in self.get_known_types()]

        # plot memberships
        ind = [cell_type.get_name() for cell_type in self.Celltypes]
        f, ax = plt.subplots(figsize=figsize)
        plt.bar(ind, memberships.iloc[0, :], color=celltype_colors[0])
        bottom = memberships.iloc[0, :]
        for i in range(1, len(memberships.index)):
            plt.bar(ind, memberships.iloc[i, :], bottom = bottom, color=celltype_colors[i])
            bottom = bottom + memberships.iloc[i, :]

        if(rotated_labels):
            plt.xticks(rotation=45, ha='right')
        if(ylim!=None):
            plt.ylim(ylim[0], ylim[1])
        plt.savefig(path, format='pdf', bbox_inches='tight')

    def connectivity(self, adj, use_stored_adj=None, normalize_pre_num = False, normalize_post_num = False):

        if(use_stored_adj==True):
            adj_df = self.adj
        else:
            adj_df = adj

        celltypes = [x.get_skids() for x in self.Celltypes]
        celltype_names = [x.get_name() for x in self.Celltypes]
        mat = np.zeros((len(celltypes), len(celltypes)))
        for i, key_i in enumerate(celltypes):
            for j, key_j in enumerate(celltypes):
                if(normalize_pre_num==False & normalize_post_num==False):
                    mat[i, j] = adj_df.loc[key_i, key_j].values.sum()
                if(normalize_pre_num==True):
                    mat[i, j] = adj_df.loc[key_i, key_j].values.sum()/len(key_i)
                if(normalize_post_num==True):
                    mat[i, j] = adj_df.loc[key_i, key_j].values.sum()/len(key_j)
        mat = pd.DataFrame(mat, index = celltype_names, columns = celltype_names)
        return(mat)

    def upset_members(self, threshold=0, path=None, plot_upset=False, show_counts_bool=True, exclude_singletons_from_threshold=False, threshold_dual_cats=None, exclude_skids=None):

        celltypes = self.Celltypes

        contents = {} # empty dictionary
        for celltype in celltypes:
            name = celltype.get_name()
            contents[name] = celltype.get_skids()

        data = from_contents(contents)

        # identify indices of set intersection between all data and exclude_skids
        if(exclude_skids!=None):
            ind_dict = dict((k,i) for i,k in enumerate(data.id.values))
            inter = set(ind_dict).intersection(exclude_skids)
            indices = [ind_dict[x] for x in inter]
            data = data.iloc[np.setdiff1d(range(0, len(data)), indices)]

        unique_indices = np.unique(data.index)
        cat_types = [Celltype(' + '.join([data.index.names[i] for i, value in enumerate(index) if value==True]), 
                    list(data.loc[index].id)) for index in unique_indices]

        # apply threshold to all category types
        if(exclude_singletons_from_threshold==False):
            cat_bool = [len(x.get_skids())>=threshold for x in cat_types]
        
        # allows categories with no intersection ('singletons') to dodge the threshold
        if((exclude_singletons_from_threshold==True) & (threshold_dual_cats==None)): 
            cat_bool = [(((len(x.get_skids())>=threshold) | ('+' not in x.get_name()))) for x in cat_types]

        # allows categories with no intersection ('singletons') to dodge the threshold and additional threshold for dual combos
        if((exclude_singletons_from_threshold==True) & (threshold_dual_cats!=None)): 
            cat_bool = [(((len(x.get_skids())>=threshold) | ('+' not in x.get_name())) | (len(x.get_skids())>=threshold_dual_cats) & (x.get_name().count('+')<2)) for x in cat_types]

        cats_selected = list(np.array(cat_types)[cat_bool])
        skids_selected = [x for sublist in [cat.get_skids() for cat in cats_selected] for x in sublist]

        # identify indices of set intersection between all data and skids_selected
        ind_dict = dict((k,i) for i,k in enumerate(data.id.values))
        inter = set(ind_dict).intersection(skids_selected)
        indices = [ind_dict[x] for x in inter]

        data = data.iloc[indices]

        # identify skids that weren't plotting in upset plot (based on plotting threshold)
        all_skids = [x for sublist in [cat.get_skids() for cat in cat_types] for x in sublist]
        skids_excluded = list(np.setdiff1d(all_skids, skids_selected))

        if(plot_upset):
            if(show_counts_bool):
                fg = plot(data, sort_categories_by = None, show_counts='%d')
            else: 
                fg = plot(data, sort_categories_by = None)

            if(threshold_dual_cats==None):
                plt.savefig(f'{path}_excluded{len(skids_excluded)}_threshold{threshold}.pdf', bbox_inches='tight')
            if(threshold_dual_cats!=None):
                plt.savefig(f'{path}_excluded{len(skids_excluded)}_threshold{threshold}_dual-threshold{threshold_dual_cats}.pdf', bbox_inches='tight')

        return (cat_types, cats_selected, skids_excluded)

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
    def get_skids_from_meta_annotation(meta, split=False):
        annot_list = pymaid.get_annotated(meta).name
        skids = [list(pymaid.get_skids_by_annotation(annots)) for annots in annot_list]
        if(split==False):
            skids = [x for sublist in skids for x in sublist]
            return(skids)
        if(split==True):
            return(skids, annot_list)
    
    @staticmethod
    def default_celltypes(exclude = []):
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
                
                # remove neurons in optional "exclude" list
                if(len(exclude)>0):
                    skids_temp = list(np.setdiff1d(skids_temp, exclude))

                group.append(skids_temp)
            skid_groups.append(group)

        # test skid counts for each group
        #[len(x) for sublist in skid_groups for x in sublist]

        # make list of lists of skids + their associated names
        skid_groups = [x for sublist in skid_groups for x in sublist]
        names = [list(pymaid.get_annotated(x).name) for x in priority_list]
        names = [x for sublist in names for x in sublist]
        names = [x.replace('mw brain ', '') for x in names]
        
        # identify colors
        colors = list(pymaid.get_annotated('mw brain simple colors').name)

        # official order; note that it will have to change if any new groups are added
        official_order = ['sensories', 'PNs', 'LNs', 'LHNs', 'FFNs', 'MBINs', 'KCs', 'MBONs', 'MB-FBNs', 'CNs', 'ascendings', 'pre-dSEZs', 'pre-dVNCs', 'RGNs', 'dSEZs', 'dVNCs']
        colors_names = [x.name.values[0] for x in list(map(pymaid.get_annotated, colors))] # use order of colors annotation for now
        if(len(official_order)!=len(colors_names)):
            print('warning: issue with annotations! Check "official_order" in Celltype_Analyzer.default_celltypes()')
            
        # ordered properly and linked to colors
        groups_sort = [np.where(x==np.array(official_order))[0][0] for x in names]
        names = [element for _, element in sorted(zip(groups_sort, names))]
        skid_groups = [element for _, element in sorted(zip(groups_sort, skid_groups))]

        color_sort = [np.where(x.replace('mw brain ', '')==np.array(official_order))[0][0] for x in colors_names]
        colors = [element for _, element in sorted(zip(color_sort, colors))]

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
