#%%
import os
import sys
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory

from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import connectome_tools.process_matrix as pm
import connectome_tools.process_graph as pg
import connectome_tools.cascade_analysis as casc
import connectome_tools.celltype as ct
import connectome_tools.cluster_analysis as clust

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

adj = pm.Promat.pull_adj('ad', subgraph='brain and accessory')
ad_edges = pd.read_csv('data/edges_threshold/ad_all-paired-edges.csv', index_col=0)
graph = pg.Analyze_Nx_G(ad_edges, split_pairs=False)

pairs = pm.Promat.get_pairs()
dVNCs = pymaid.get_skids_by_annotation('mw dVNC')
dVNCs = [x if x!=21790197 else 15672263 for x in dVNCs] # a single descending neuron was incorrectly merged and split, so skid is different...

# %%
# connection probability between ipsi/bilateral/contra
dVNC_pairs = pm.Promat.load_pairs_from_annotation('dVNCs', pairs, return_type='all_pair_ids_bothsides', skids=dVNCs, use_skids=True)
dSEZ_pairs = pm.Promat.load_pairs_from_annotation('mw dSEZ', pairs, return_type='all_pair_ids_bothsides')
RGN_pairs = pm.Promat.load_pairs_from_annotation('mw RGN', pairs, return_type='all_pair_ids_bothsides')

ascendings_all = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw A1 ascending')
asc_pairs = pm.Promat.load_pairs_from_annotation('ascendings', pairs, return_type='all_pair_ids_bothsides', skids=ascendings_all, use_skids=True)

non_outputs_brain = np.intersect1d(pymaid.get_skids_by_annotation('mw brain paper clustered neurons'), pymaid.get_skids_by_annotation('mw brain neurons'))
non_outputs_brain = np.setdiff1d(non_outputs_brain, ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain outputs'))
non_outputs_brain_pairs = pm.Promat.load_pairs_from_annotation('non-outputs', pairs, return_type='all_pair_ids_bothsides', skids=non_outputs_brain, use_skids=True)

# ascnedings/dVNCs to outputs and ascendings
data_adj = ad_edges.set_index(['upstream_pair_id', 'downstream_pair_id'])
celltypes_pre = [list(asc_pairs.leftid), list(dVNC_pairs.leftid)]
celltypes_post = [list(dVNC_pairs.leftid), list(dSEZ_pairs.leftid), list(RGN_pairs.leftid), list(non_outputs_brain_pairs.leftid), list(asc_pairs.leftid)]

mat = np.zeros(shape=(len(celltypes_pre), len(celltypes_post)))
for i, pair_type1 in enumerate(celltypes_pre):
    for j, pair_type2 in enumerate(celltypes_post):
        connection = []
        for skid1 in pair_type1:
            for skid2 in pair_type2:
                if((skid1, skid2) in graph.G.edges): connection.append(1)
                if((skid1, skid2) not in graph.G.edges): connection.append(0)

            mat[i, j] = sum(connection)/len(connection)

df = pd.DataFrame(mat, columns = ['dVNC', 'dSEZ', 'RGN', 'brain-non-outputs', 'A1-ascending'],
                        index = ['A1-ascending', 'dVNC'])

fig, ax = plt.subplots(1,1, figsize=(2,2))
sns.heatmap(df, square=True, cmap='Blues', vmax=0.007)
plt.savefig(f'VNC_interaction/plots/connection-probability_brain-outputs_ascendings.pdf', format='pdf', bbox_inches='tight')

# ascendings to brain

_, celltypes = ct.Celltype_Analyzer.default_celltypes(exclude=pymaid.get_skids_by_annotation('mw dVNC to A1'))
celltypes = celltypes + [ct.Celltype(name='dVNCs-A1', skids=pymaid.get_skids_by_annotation('mw dVNC to A1'))]

celltypes_pairs = [pm.Promat.load_pairs_from_annotation('', pairs, return_type='all_pair_ids_bothsides', skids=celltype.get_skids(), use_skids=True) for celltype in celltypes]

celltypes_pre = [list(asc_pairs.leftid)]
celltypes_post = [list(pairs_from_list.leftid) for pairs_from_list in celltypes_pairs]

mat = np.zeros(shape=(len(celltypes_pre), len(celltypes_post)))
for i, pair_type1 in enumerate(celltypes_pre):
    for j, pair_type2 in enumerate(celltypes_post):
        connection = []
        for skid1 in pair_type1:
            for skid2 in pair_type2:
                if((skid1, skid2) in graph.G.edges): connection.append(1)
                if((skid1, skid2) not in graph.G.edges): connection.append(0)

            mat[i, j] = sum(connection)/len(connection)

df = pd.DataFrame(mat, columns = [celltype.get_name() for celltype in celltypes],
                        index = ['A1-ascending'])

# modify 'Blues' cmap to have a white background
cmap = plt.cm.get_cmap('Blues')
blue_cmap = cmap(np.linspace(0, 1, 20))
blue_cmap[0] = np.array([1, 1, 1, 1])
blue_cmap = mpl.colors.LinearSegmentedColormap.from_list(name='New_Blues', colors=blue_cmap)

cmap = blue_cmap
vmax = 0.02

fig, ax = plt.subplots(1,1, figsize=(2.5,2))
sns.heatmap(df, square=True, cmap=cmap, vmax=vmax)
plt.savefig(f'VNC_interaction/plots/connection-probability_ascendings_all-brain-celltypes.pdf', format='pdf', bbox_inches='tight')


# dVNCs to A1
motorneuron_pairs = pm.Promat.load_pairs_from_annotation('mw A1 MN', pairs)
A1_cells = np.setdiff1d(pymaid.get_skids_by_annotation('mw A1 neurons paired'), pymaid.get_skids_by_annotation('mw A1 MN') + ascendings_all)
A1_pairs = pm.Promat.load_pairs_from_annotation('A1', pairs, return_type='all_pair_ids_bothsides', skids=A1_cells, use_skids=True)
dVNC_A1_pairs = pm.Promat.load_pairs_from_annotation('mw dVNC to A1', pairs, return_type='all_pair_ids_bothsides')

dVNC_nonA1 = np.setdiff1d(pymaid.get_skids_by_annotation('mw dVNC'), pymaid.get_skids_by_annotation('mw dVNC to A1'))
dVNC_nonA1_pairs = pm.Promat.load_pairs_from_annotation('mw dVNC not to A1', pairs, return_type='all_pair_ids_bothsides', skids=dVNC_nonA1, use_skids=True)
celltypes_pre = [list(dVNC_A1_pairs.leftid), list(dVNC_nonA1_pairs.leftid)]
celltypes_post = [list(asc_pairs.leftid), list(motorneuron_pairs.leftid), list(A1_pairs.leftid)]

mat = np.zeros(shape=(len(celltypes_pre), len(celltypes_post)))
for i, pair_type1 in enumerate(celltypes_pre):
    for j, pair_type2 in enumerate(celltypes_post):
        connection = []
        for skid1 in pair_type1:
            for skid2 in pair_type2:
                if((skid1, skid2) in graph.G.edges): connection.append(1)
                if((skid1, skid2) not in graph.G.edges): connection.append(0)

            mat[i, j] = sum(connection)/len(connection)

df = pd.DataFrame(mat, columns = ['A1-ascending', 'A1-motorneuron', 'A1-interneuron'],
                        index = ['dVNC to A1', 'dVNC not to A1'])

cmap = blue_cmap
vmax = 0.02

fig, ax = plt.subplots(1,1, figsize=(2,2))
sns.heatmap(df, square=True, cmap=cmap, vmax=vmax)
plt.savefig(f'VNC_interaction/plots/connection-probability_dVNCs_A1cells.pdf', format='pdf', bbox_inches='tight')

# summary connectivity probability plot
celltypes_pre = [list(dVNC_A1_pairs.leftid), list(dVNC_nonA1_pairs.leftid), list(dSEZ_pairs.leftid), list(RGN_pairs.leftid), list(asc_pairs.leftid), list(A1_pairs.leftid), list(motorneuron_pairs.leftid)]
celltypes_post = [list(dVNC_A1_pairs.leftid), list(dVNC_nonA1_pairs.leftid), list(dSEZ_pairs.leftid), list(RGN_pairs.leftid), list(asc_pairs.leftid), list(A1_pairs.leftid), list(motorneuron_pairs.leftid)]


mat = np.zeros(shape=(len(celltypes_pre), len(celltypes_post)))
for i, pair_type1 in enumerate(celltypes_pre):
    for j, pair_type2 in enumerate(celltypes_post):
        connection = []
        for skid1 in pair_type1:
            for skid2 in pair_type2:
                if((skid1, skid2) in graph.G.edges): connection.append(1)
                if((skid1, skid2) not in graph.G.edges): connection.append(0)

            mat[i, j] = sum(connection)/len(connection)

df = pd.DataFrame(mat, columns = ['dVNC to A1', 'dVNC not to A1', 'dSEZ', 'RGN', 'A1-ascending', 'A1-interneuron', 'A1-motorneuron'],
                        index = ['dVNC to A1', 'dVNC not to A1', 'dSEZ', 'RGN', 'A1-ascending', 'A1-interneuron', 'A1-motorneuron'])

cmap = blue_cmap
vmax = 0.02

fig, ax = plt.subplots(1,1, figsize=(2,2))
sns.heatmap(df, square=True, cmap=cmap, vmax=vmax)
plt.savefig(f'VNC_interaction/plots/connection-probability_brain-A1_summary.pdf', format='pdf', bbox_inches='tight')

# %%
# connection probability of self loop (dVNC<->ascending-A1) vs zigzag motif (dVNC1->ascending-A1->dVNC2)


def loop_zigzag_probability(graph, pairs, length):
    # requires Analyze_Nx_G(..., split_pairs=True)

    ascending = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw A1 ascending')
    
    if(length<2):
        print('length must be 2 or greater!')
        return

    partner_loop = []
    nonpartner_loop = []
    zigzag = []
    all_paths = []
    for i in pairs.index:
        leftid = pairs.loc[i].leftid
        rightid = pairs.loc[i].rightid
        paths = graph.all_simple_self_loop_paths(source = leftid, cutoff=length)
        paths = [path for path in paths if len(path)==(length+1)]
        all_paths.append(paths)

        # when loops exist
        if(len(paths)>0):
            loop_partners = [path[1:length] for path in paths] # collect all partners that mediate loops
            if(type(loop_partners[0])==list): loop_partners = [x for sublist in loop_partners for x in sublist]
            loop_partners = list(np.unique(loop_partners))

            if((rightid in loop_partners) & (sum([x in loop_partners for x in ascending])>0)): partner_loop.append(1)
            if(rightid not in loop_partners): partner_loop.append(0)

            for skid in pairs.rightid.values:
                if(skid in loop_partners): nonpartner_loop.append(1)
                if(skid not in loop_partners): nonpartner_loop.append(0)
                
        # when loops don't exist
        if(len(paths)==0):
            partner_loop.append(0)
            for skid in pairs.rightid:
                nonpartner_loop.append(0)

    prob_partner_loop = sum(partner_loop)/len(partner_loop)
    prob_nonpartner_loop = sum(nonpartner_loop)/len(nonpartner_loop)

    return(prob_partner_loop, prob_nonpartner_loop, all_paths)


