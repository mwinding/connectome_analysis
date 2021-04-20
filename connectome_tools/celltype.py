# object for analysing and grouping celltypes

import numpy as np
import pandas as pd
import sys
import pymaid
sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
from connectome_tools.process_matrix import Promat


class Celltype:
    def __init__(self, name, skids):
        self.name = name
        self.skids = list(np.unique(skids))

    def get_name(self):
        return(self.name)

    def get_skids(self):
        return(self.skids)


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
            unknown_type = [Celltype('unknown', unknown_skids)]
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

    def get_skids_from_meta_meta_annotation(self, meta_meta, split=False):
        meta_annots = pymaid.get_annotated(meta_meta).name
        annot_list = [list(pymaid.get_annotated(meta).name) for meta in meta_annots]
        skids = [list(pymaid.get_skids_by_annotation(annots)) for annots in annot_list]
        if(split==False):
            skids = [x for sublist in skids for x in sublist]
            return(skids)
        if(split==True):
            return(skids, meta_annots)
        
    def default_celltypes(self):
        priority_list = pymaid.get_annotated('mw brain simple priorities').name
        priority_skids = [self.get_skids_from_meta_meta_annotation(priority) for priority in priority_list]

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
        data = pd.DataFrame(zip(names, skid_groups, colors), columns = ['celltype', 'skids', 'color'])
        return(data)

    # set of known celltypes, returned as skid lists
    @staticmethod
    def celltypes(more_celltypes=[], more_names=[]):
        A1 = pymaid.get_skids_by_annotation('mw A1 neurons paired')
        br = pymaid.get_skids_by_annotation('mw brain neurons')
        MBON = pymaid.get_skids_by_annotation('mw MBON')
        MBIN = pymaid.get_skids_by_annotation('mw MBIN')
        LHN = pymaid.get_skids_by_annotation('mw LHN')
        LN = pymaid.get_skids_by_annotation('mw LNs')
        CN = pymaid.get_skids_by_annotation('mw CN')
        KC = pymaid.get_skids_by_annotation('mw KC')
        RGN = pymaid.get_skids_by_annotation('mw RGN')
        dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')
        pre_dVNC = pymaid.get_skids_by_annotation('mw pre-dVNC 1%')
        pre_dSEZ = pymaid.get_skids_by_annotation('mw pre-dSEZ 1%')
        pre_RGN = pymaid.get_skids_by_annotation('mw pre-RGN 1%')
        dVNC = pymaid.get_skids_by_annotation('mw dVNC')
        uPN = pymaid.get_skids_by_annotation('mw uPN')
        tPN = pymaid.get_skids_by_annotation('mw tPN')
        vPN = pymaid.get_skids_by_annotation('mw vPN')
        mPN = pymaid.get_skids_by_annotation('mw mPN')
        PN = uPN + tPN + vPN + mPN
        FBN = pymaid.get_skids_by_annotation('mw FBN')
        FB2N = pymaid.get_skids_by_annotation('mw FB2N')
        FBN_all = FBN + FB2N

        input_names = pymaid.get_annotated('mw brain sensories').name
        input_skids_list = list(map(pymaid.get_skids_by_annotation, input_names))
        sens_all = [x for sublist in input_skids_list for x in sublist]

        asc_noci = pymaid.get_skids_by_annotation('mw A1 ascending noci')
        asc_mechano = pymaid.get_skids_by_annotation('mw A1 ascending mechano')
        asc_proprio = pymaid.get_skids_by_annotation('mw A1 ascending proprio')
        asc_classII_III = pymaid.get_skids_by_annotation('mw A1 ascending class II_III')
        asc_all = [pymaid.get_skids_by_annotation(x) for x in pymaid.get_annotated('mw A1 ascending').name]
        asc_all = [x for sublist in asc_all for x in sublist]

        LHN = list(np.setdiff1d(LHN, asc_all + PN + FBN_all + dVNC)) # 'LHN' means exclusive LHNs that are not FBN or dVNC
        CN = list(np.setdiff1d(CN, asc_all + PN + MBON + LHN + FBN_all + dVNC)) # 'CN' means exclusive CNs that are not FBN or LHN or dVNC
        pre_dVNC = list(np.setdiff1d(pre_dVNC, asc_all + sens_all + LN + MBON + MBIN + LHN + CN + KC + RGN + dSEZ + dVNC + PN + FBN_all + asc_all)) # 'pre_dVNC' must have no other category assignment
        pre_dSEZ = list(np.setdiff1d(pre_dSEZ, asc_all + sens_all + LN + MBON + MBIN + LHN + CN + KC + RGN + dSEZ + dVNC + PN + FBN_all + asc_all + pre_dVNC)) # 'pre_dSEZ' must have no other category assignment
        pre_RGN = list(np.setdiff1d(pre_RGN, asc_all + sens_all + LN + MBON + MBIN + LHN + CN + KC + RGN + dSEZ + dVNC + PN + FBN_all + asc_all + pre_dVNC + pre_RGN)) # 'pre_RGN' must have no other category assignment
        dSEZ = list(np.setdiff1d(dSEZ, asc_all + sens_all + MBON + LN + MBIN + LHN + CN + KC + dVNC + PN + FBN_all + dVNC))
        RGN = list(np.setdiff1d(RGN, asc_all + dSEZ + dVNC))

        immature = pymaid.get_skids_by_annotation('mw partially differentiated')
        A1_local = list(np.setdiff1d(A1, asc_all)) # all A1 without A1_ascending

        celltypes = [sens_all, PN, LN, LHN, MBIN, list(np.setdiff1d(KC, immature)), MBON, FBN_all, CN, asc_all, pre_RGN, pre_dSEZ, pre_dVNC, RGN, dSEZ, dVNC]
        celltype_names = ['Sens', 'PN', 'LN', 'LHN', 'MBIN', 'KC', 'MBON', 'MB-FBN', 'CN', 'ascending', 'pre-RGN', 'pre-dSEZ','pre-dVNC', 'RGN', 'dSEZ', 'dVNC']
        colors = ['#00753F', '#1D79B7', '#5D8C90', '#D4E29E', '#FF8734', '#E55560', '#F9EB4D', '#C144BC', '#8C7700', '#77CDFC', '#FFDAC7', '#E0B1AD', '#9467BD','#D88052', '#A52A2A']

        if(len(more_celltypes)>0):
            celltypes = celltypes + more_celltypes
            celltype_names = celltype_names + more_names

        # only use celltypes that have non-zero number of members
        exists = [len(x)!=0 for x in celltypes]
        celltypes = [x for i,x in enumerate(celltypes) if exists[i]==True]
        celltype_names = [x for i,x in enumerate(celltype_names) if exists[i]==True]

        return(celltypes, celltype_names, colors)
