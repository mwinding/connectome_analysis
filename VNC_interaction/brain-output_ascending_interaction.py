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
import networkx as nx

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

# ascendings/dVNCs to outputs and ascendings
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

fig, ax = plt.subplots(1,1, figsize=(5,2))
sns.heatmap(df, square=True, cmap=cmap, vmax=vmax, annot=True, fmt='.3f')
plt.savefig(f'VNC_interaction/plots/connection-probability_ascendings_all-brain-celltypes.pdf', format='pdf', bbox_inches='tight')

# dVNCs to A1
motorneuron_pairs = pm.Promat.load_pairs_from_annotation('mw A1 MN', pairs)
pre_motorneuron_pairids = ad_edges.set_index('downstream_pair_id').loc[np.intersect1d(motorneuron_pairs.leftid, ad_edges.downstream_pair_id), 'upstream_pair_id']
pre_motorneuron_pairids = list(np.unique(pre_motorneuron_pairids))
pre_motorneuron_pairids = list(np.intersect1d(pre_motorneuron_pairids, pymaid.get_skids_by_annotation('mw A1 neurons paired')))
pre_motorneurons = pre_motorneuron_pairids + list(pairs.set_index('leftid').loc[pre_motorneuron_pairids, 'rightid'])

A1_cells = np.setdiff1d(pymaid.get_skids_by_annotation('mw A1 neurons paired'), pymaid.get_skids_by_annotation('mw A1 MN') + pre_motorneurons + ascendings_all)
A1_pairs = pm.Promat.load_pairs_from_annotation('A1', pairs, return_type='all_pair_ids_bothsides', skids=A1_cells, use_skids=True)
dVNC_A1_pairs = pm.Promat.load_pairs_from_annotation('mw dVNC to A1', pairs, return_type='all_pair_ids_bothsides')

dVNC_nonA1 = np.setdiff1d(pymaid.get_skids_by_annotation('mw dVNC'), pymaid.get_skids_by_annotation('mw dVNC to A1'))
dVNC_nonA1_pairs = pm.Promat.load_pairs_from_annotation('mw dVNC not to A1', pairs, return_type='all_pair_ids_bothsides', skids=dVNC_nonA1, use_skids=True)
celltypes_pre = [list(dVNC_A1_pairs.leftid), list(dVNC_nonA1_pairs.leftid)]
celltypes_post = [list(asc_pairs.leftid), list(A1_pairs.leftid), pre_motorneuron_pairids, list(motorneuron_pairs.leftid)]

mat = np.zeros(shape=(len(celltypes_pre), len(celltypes_post)))
for i, pair_type1 in enumerate(celltypes_pre):
    for j, pair_type2 in enumerate(celltypes_post):
        connection = []
        for skid1 in pair_type1:
            for skid2 in pair_type2:
                if((skid1, skid2) in graph.G.edges): connection.append(1)
                if((skid1, skid2) not in graph.G.edges): connection.append(0)

            mat[i, j] = sum(connection)/len(connection)

df = pd.DataFrame(mat, columns = ['A1-ascending', 'A1-interneuron', 'A1-pre-motorneuron', 'A1-motorneuron'],
                        index = ['dVNC to A1', 'dVNC not to A1'])

cmap = blue_cmap
vmax = 0.02

fig, ax = plt.subplots(1,1, figsize=(2,2))
sns.heatmap(df, square=True, cmap=cmap, vmax=vmax)
plt.savefig(f'VNC_interaction/plots/connection-probability_dVNCs_A1cells.pdf', format='pdf', bbox_inches='tight')

# summary connectivity probability plot
celltypes_pre = [list(dVNC_A1_pairs.leftid), list(dVNC_nonA1_pairs.leftid), list(dSEZ_pairs.leftid), list(RGN_pairs.leftid), list(asc_pairs.leftid), list(A1_pairs.leftid), pre_motorneuron_pairids, list(motorneuron_pairs.leftid)]
celltypes_post = [list(dVNC_A1_pairs.leftid), list(dVNC_nonA1_pairs.leftid), list(dSEZ_pairs.leftid), list(RGN_pairs.leftid), list(asc_pairs.leftid), list(A1_pairs.leftid), pre_motorneuron_pairids, list(motorneuron_pairs.leftid)]

mat = np.zeros(shape=(len(celltypes_pre), len(celltypes_post)))
for i, pair_type1 in enumerate(celltypes_pre):
    for j, pair_type2 in enumerate(celltypes_post):
        connection = []
        for skid1 in pair_type1:
            for skid2 in pair_type2:
                if((skid1, skid2) in graph.G.edges): connection.append(1)
                if((skid1, skid2) not in graph.G.edges): connection.append(0)

            mat[i, j] = sum(connection)/len(connection)

df = pd.DataFrame(mat, columns = ['dVNC to A1', 'dVNC not to A1', 'dSEZ', 'RGN', 'A1-ascending', 'A1-interneuron', 'A1-pre-motorneuron', 'A1-motorneuron'],
                        index = ['dVNC to A1', 'dVNC not to A1', 'dSEZ', 'RGN', 'A1-ascending', 'A1-interneuron', 'A1-pre-motorneuron', 'A1-motorneuron'])

cmap = blue_cmap
vmax = 0.04

fig, ax = plt.subplots(1,1, figsize=(3,3))
sns.heatmap(df, square=True, cmap=cmap, vmax=vmax, annot=True, fmt='.3f')
plt.savefig(f'VNC_interaction/plots/connection-probability_brain-A1_summary.pdf', format='pdf', bbox_inches='tight')

# %%
# connection probability of self loop (dVNC<->ascending-A1) vs zigzag motif (dVNC1->ascending-A1->dVNC2)

# generate graph for dVNCs and A1

def dVNC_asc_loop_probability(graph, pairs, length, pre=[], use_pre=False):
    # requires Analyze_Nx_G(..., split_pairs=True)

    ascs = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw A1 ascending')
    
    if(length<2):
        print('length must be 2 or greater!')
        return

    dVNC_ascending_loop = []
    all_paths = []
    loop_paths = []
    for i in pairs.index:
        leftid = pairs.loc[i].leftid
        if(leftid in graph.G.nodes):
            paths = graph.all_simple_self_loop_paths(source = leftid, cutoff=length)
            paths = [path for path in paths if len(path)==(length+1)]
            all_paths.append(paths)

            # when loops exist
            if(len(paths)>0):
                loop_partners = [path[1:length] for path in paths] # collect all partners that mediate loops
                if(type(loop_partners[0])==list): loop_partners = [x for sublist in loop_partners for x in sublist]
                loop_partners = list(np.unique(loop_partners))

                if(use_pre):
                    asc_present = sum([1 for x in loop_partners if x in ascs])>0
                    pre_not_in_middle = sum([1 for x in loop_partners[0:(len(loop_partners)-1)] if x in pre])==0

                    if(asc_present & pre_not_in_middle): 
                        dVNC_ascending_loop.append(1)
                        loop_paths.append(path)
                    if((asc_present==False) | (pre_not_in_middle==False)): dVNC_ascending_loop.append(0)

                if(use_pre==False):
                    asc_present = sum([1 for x in loop_partners if x in ascs])>0
                    if(asc_present): dVNC_ascending_loop.append(1)
                    if(asc_present==False): dVNC_ascending_loop.append(0)

            # when loops don't exist
            if(len(paths)==0): dVNC_ascending_loop.append(0)

        if(leftid not in graph.G.nodes):
            dVNC_ascending_loop.append(0)

    prob_dVNC_ascending_loop = sum(dVNC_ascending_loop)/len(dVNC_ascending_loop)
    return(prob_dVNC_ascending_loop, all_paths, loop_paths)

def dVNC_asc_zigzag_probability(graph, pairs, targets, length, exclude_from_path=[], pre=[], use_pre=False):
    # requires Analyze_Nx_G(..., split_pairs=True)
    ascending = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw A1 ascending')
    brain_neurons = pymaid.get_skids_by_annotation('mw brain neurons')

    if(length<2):
        print('length must be 2 or greater!')
        return

    # generate appropriate paths
    all_paths = []
    for i in pairs.index:
        leftid = pairs.loc[i].leftid
        if(leftid in graph.G.nodes):
            if(leftid in targets): targets.remove(leftid) # remove the current neuron from the list of targets (not looking for self loops)
            paths = nx.all_simple_paths(G=graph.G, source = leftid, target = targets, cutoff=length)
            paths = [path for path in paths if len(path)==(length+1)]
            all_paths.append(paths)

    # check how many paths exist with ascending and dVNCs present
    zigzag_paths = []
    dVNC_ascending_zigzag = []
    for paths in all_paths:
        path_exists = []
        for path in paths:

            # must talk to A1 neuron in first hop
            if(path[1] in brain_neurons):
                continue

            # celltypes to compare against
            ascs = ascending
            
            # are there ascending and dVNCs present in each path?
            asc_present = sum([1 for x in path[1:len(path)] if x in ascs])>0
            target_present = sum([1 for x in path[1:(len(path)-1)] if x in (targets+exclude_from_path)])==0 # should only be the target type at the end of the path
            if(use_pre):
                if(length>=4):
                    pre_not_in_middle = sum([1 for x in path[1:(len(path)-2)] if x in pre])==0
                    if((asc_present) & (target_present) & (pre_not_in_middle)):
                        path_exists.append(1)
                        zigzag_paths.append(path)
                if(length<4):
                    if((asc_present) & (target_present)): 
                        path_exists.append(1)
                        zigzag_paths.append(path)
            if(use_pre==False):
                if((asc_present) & (target_present)): 
                    path_exists.append(1)
                    zigzag_paths.append(path)        
        
        if(sum(path_exists)>0): dVNC_ascending_zigzag.append(1)

    dVNC_ascending_zigzag_prob = sum(dVNC_ascending_zigzag)/len(pairs.leftid)

    return(dVNC_ascending_zigzag_prob, all_paths, zigzag_paths)

A1 = pymaid.get_skids_by_annotation('mw A1 neurons paired')
pre_dVNC = pymaid.get_skids_by_annotation('mw pre-dVNC')
graph_dVNC_A1 = pg.Analyze_Nx_G(ad_edges, split_pairs=False, select_neurons = list(dVNCs) + A1 + pre_dVNC) # dVNCs includes the one skid that needs to be manually added (in chunk #1)

from joblib import Parallel, delayed
from tqdm import tqdm

sources = dVNC_A1_pairs
targets = dVNC_pairs.leftid.to_list()

lengths = [2,3,4,5]
loops = Parallel(n_jobs=-1)(delayed(dVNC_asc_loop_probability)(graph_dVNC_A1, sources, length=lengths[i], pre=pre_dVNC, use_pre=True) for i in tqdm(range(len(lengths))))
loop_probs = [x[0] for x in loops]
loop_all_paths = [x[1] for x in loops]
loop_paths = [x[2] for x in loops]

zigzag_dVNC = Parallel(n_jobs=-1)(delayed(dVNC_asc_zigzag_probability)(graph_dVNC_A1, sources, targets, length=lengths[i], pre=pre_dVNC, use_pre=True) for i in tqdm(range(len(lengths))))
zigzag_dVNC_probs = [x[0] for x in zigzag_dVNC]
zigzag_all_dVNC_paths = [x[1] for x in zigzag_dVNC]
zigzag_dVNC_paths = [x[2] for x in zigzag_dVNC[0:3]]

# zigzags to other output types
dSEZs = pymaid.get_skids_by_annotation('mw dSEZ')
RGNs = pymaid.get_skids_by_annotation('mw RGN')
pre_dSEZ = pymaid.get_skids_by_annotation('mw pre-dSEZ')
graph_outputs_A1 = pg.Analyze_Nx_G(ad_edges, split_pairs=False, select_neurons = np.unique(list(dVNCs) + dSEZs + pre_dSEZ + A1)) # dVNCs includes the one skid that needs to be manually added (in chunk #1)
targets = dSEZ_pairs.leftid.to_list()
zigzag_dSEZ = Parallel(n_jobs=-1)(delayed(dVNC_asc_zigzag_probability)(graph_outputs_A1, sources, targets, length=lengths[i], pre=pre_dSEZ, use_pre=True) for i in tqdm(range(len(lengths))))
zigzag_dSEZ_probs = [x[0] for x in zigzag_dSEZ]
zigzag_all_dSEZ_paths = [x[1] for x in zigzag_dSEZ]
zigzag_dSEZ_paths = [x[2] for x in zigzag_dSEZ[0:3]]

pre_RGN = pymaid.get_skids_by_annotation('mw pre-RGN')
graph_outputs_A1 = pg.Analyze_Nx_G(ad_edges, split_pairs=False, select_neurons = np.unique(list(dVNCs) + RGNs + pre_RGN + A1)) # dVNCs includes the one skid that needs to be manually added (in chunk #1)
targets = RGN_pairs.leftid.to_list()
zigzag_RGN = Parallel(n_jobs=-1)(delayed(dVNC_asc_zigzag_probability)(graph_outputs_A1, sources, targets, length=lengths[i], pre=pre_RGN, use_pre=True) for i in tqdm(range(len(lengths))))
zigzag_RGN_probs = [x[0] for x in zigzag_RGN]
zigzag_all_RGN_paths = [x[1] for x in zigzag_RGN]
zigzag_RGN_paths = [x[2] for x in zigzag_RGN[0:3]]

graph_outputs_A1 = pg.Analyze_Nx_G(ad_edges, split_pairs=False, select_neurons = np.unique(list(dVNCs) + dSEZs + RGNs + pre_RGN + pre_dSEZ + pre_RGN + A1)) # dVNCs includes the one skid that needs to be manually added (in chunk #1)
targets = list(np.unique(RGN_pairs.leftid.to_list() + dSEZ_pairs.leftid.to_list() + dVNC_pairs.leftid.to_list()))
zigzag_outputs = Parallel(n_jobs=-1)(delayed(dVNC_asc_zigzag_probability)(graph_outputs_A1, sources, targets, length=lengths[i], pre=pre_RGN + pre_dSEZ + pre_dVNC, use_pre=True) for i in tqdm(range(len(lengths))))
zigzag_outputs_probs = [x[0] for x in zigzag_outputs]
zigzag_all_outputs_paths = [x[1] for x in zigzag_outputs]
zigzag_outputs_paths = [x[2] for x in zigzag_outputs]

# %%
# plot fraction of dVNCs displaying zigzags vs loops

df = pd.DataFrame([loop_probs, zigzag_dVNC_probs], index = ['loops', 'zigzags'], columns = lengths)

fig,ax = plt.subplots(1,1,figsize=(2,2))
sns.heatmap(df, ax=ax, cmap='Greens', annot=True, vmax=0.2)
plt.savefig(f'VNC_interaction/plots/loops-vs-zigzag_brain-A1.pdf', format='pdf', bbox_inches='tight')

# plot fraction of dVNC-A1s displaying zigzags to different output types
df = pd.DataFrame([zigzag_dVNC_probs, zigzag_dSEZ_probs, zigzag_RGN_probs], index = ['zigzag-dVNC', 'zigzag-dSEZ', 'zigzag-RGN'], columns = lengths)

fig,ax = plt.subplots(1,1,figsize=(2,2))
sns.heatmap(df, ax=ax, cmap='Greens', annot=True, vmax=0.2)
plt.savefig(f'VNC_interaction/plots/zigzag_different-outputs_brain-A1.pdf', format='pdf', bbox_inches='tight')

# number of motifs/paths observed
num_loop_paths = [len(x) for x in loop_paths]
num_zigzag_dVNC_paths = [len(x) for x in zigzag_dVNC_paths]
df = pd.DataFrame([num_loop_paths, num_zigzag_dVNC_paths], index = ['loops', 'zigzags'], columns = lengths)

fig,ax = plt.subplots(1,1,figsize=(1,0.5))
sns.heatmap(df, ax=ax, cmap='Greens', annot=True, vmax=25, square=True)
plt.savefig(f'VNC_interaction/plots/num_loops-vs-zigzag_brain-A1.pdf', format='pdf', bbox_inches='tight')

# number of zigzags observed for different output types
num_zigzag_dVNC_paths = [len(x) for x in zigzag_dVNC_paths]
num_zigzag_dSEZ_paths = [len(x) for x in zigzag_dSEZ_paths]
num_zigzag_RGN_paths = [len(x) for x in zigzag_RGN_paths]
num_output_paths = list(map(sum, zip(num_zigzag_dVNC_paths, num_zigzag_dSEZ_paths, num_zigzag_RGN_paths)))
df = pd.DataFrame([num_output_paths, num_zigzag_dVNC_paths, num_zigzag_dSEZ_paths, num_zigzag_RGN_paths], index = ['zigzag-all', 'zigzag-dVNC', 'zigzag-dSEZ', 'zigzag-RGN'], columns = [3,4,5])

fig,ax = plt.subplots(1,1,figsize=(1,0.5))
sns.heatmap(df, ax=ax, cmap='Greens', annot=True, vmax=35, square=True)
plt.savefig(f'VNC_interaction/plots/num_zigzag_different-outputs_brain-A1.pdf', format='pdf', bbox_inches='tight')


# %%
# plot of paths

# path that is just barely under threshold due to reconstruction issue that has been since fixed (but not in currently used data)
zigzag_dVNC_paths[1].append([5462159, 4206755, 13743125, 10728333])

def pull_zigzag_points(all_paths, middle_skid_type, convert_to_position=False, converter=None):
    source_zigzag = []
    middle_zigzag = []
    target_zigzag = []
    hops = []
    for i, paths in enumerate(all_paths):
        for path in paths:
            source = path[0]
            target = path[len(path)-1]
            middle = path[1:(len(path)-1)]
            middle = np.intersect1d(middle, middle_skid_type)
            if(len(middle)!=1): print(f'check {path}')
            if(len(middle)==1): middle = middle[0]

            if(convert_to_position==False):
                source_zigzag.append(source)
                middle_zigzag.append(middle)
                target_zigzag.append(target)
                hops.append(i+2)

            if(convert_to_position):
                converter_df = converter.set_index('leftid').copy()
                source = converter_df.loc[source, 'value']
                middle = converter_df.loc[middle, 'value']
                target = converter_df.loc[target, 'value']
                
                source_zigzag.append(source)
                middle_zigzag.append(middle)
                target_zigzag.append(target)
                hops.append(i+2)         

    return(source_zigzag, middle_zigzag, target_zigzag, hops)

source_zigzag_dVNC, middle_zigzag_dVNC, target_zigzag_dVNC, hops_dVNC = pull_zigzag_points(zigzag_dVNC_paths, ascendings_all)
source_zigzag_dSEZ, middle_zigzag_dSEZ, target_zigzag_dSEZ, hops_dSEZ = pull_zigzag_points(zigzag_dSEZ_paths, ascendings_all)

from pandas.plotting import parallel_coordinates

member_order = pd.concat([dVNC_A1_pairs, dVNC_nonA1_pairs, dSEZ_pairs])
member_order['value'] = np.flip(np.arange(0, len(member_order.index)))

addition = asc_pairs.copy()
addition['value'] = -np.arange(1, len(addition.index)+1)

member_order = pd.concat([member_order, addition])
member_order.reset_index(inplace=True, drop=True)

from pandas.plotting import parallel_coordinates


source_zigzag_dVNC, middle_zigzag_dVNC, target_zigzag_dVNC, hops_dVNC = pull_zigzag_points(zigzag_dVNC_paths, ascendings_all, convert_to_position=True, converter=member_order)
source_zigzag_dSEZ, middle_zigzag_dSEZ, target_zigzag_dSEZ, hops_dSEZ = pull_zigzag_points(zigzag_dSEZ_paths, ascendings_all, convert_to_position=True, converter=member_order)

df_dVNC = pd.DataFrame(zip(source_zigzag_dVNC, middle_zigzag_dVNC, target_zigzag_dVNC, hops_dVNC), columns = ['source', 'mid', 'target', 'nodes'])
df_dSEZ = pd.DataFrame(zip(source_zigzag_dSEZ, middle_zigzag_dSEZ, target_zigzag_dSEZ, hops_dSEZ), columns = ['source', 'mid', 'target', 'nodes'])

color=['black']
alpha=0.25
#color=['black', 'gray', 'tab:gray']
fig, ax = plt.subplots(1,1, figsize=(4,4))
parallel_coordinates(frame=df_dVNC, class_column='nodes', linewidth=0.5, alpha=alpha, color=color)
ax.set(ylim=(-22, 179))
plt.axhline(y=179-len(dVNC_A1_pairs), color='r', linestyle='-')
plt.axhline(y=179-len(dVNC_A1_pairs)-len(dVNC_nonA1_pairs), color='r', linestyle='-')
plt.axhline(y=0, color='r', linestyle='-')
plt.savefig(f'VNC_interaction/plots/parallel-coordinates_dVNC_zigzags.pdf', format='pdf', bbox_inches='tight')

#color = ['tab:gray','#D0D2D3']
fig, ax = plt.subplots(1,1, figsize=(4,4))
parallel_coordinates(frame=df_dSEZ, class_column='nodes', linewidth=0.5, alpha=alpha, color=color)
ax.set(ylim=(-22, 179))
plt.axhline(y=179-len(dVNC_A1_pairs), color='r', linestyle='-')
plt.axhline(y=179-len(dVNC_A1_pairs)-len(dVNC_nonA1_pairs), color='r', linestyle='-')
plt.axhline(y=0, color='r', linestyle='-')
plt.savefig(f'VNC_interaction/plots/parallel-coordinates_dSEZ_zigzags.pdf', format='pdf', bbox_inches='tight')

df_dVNC['type'] = ['dVNC']*len(df_dVNC.index)
df_dSEZ['type'] = ['dSEZ']*len(df_dSEZ.index)

df = pd.concat([df_dVNC, df_dSEZ])
color = ['black', 'tab:gray']
fig, ax = plt.subplots(1,1, figsize=(4,4))
parallel_coordinates(frame=df, class_column='type', linewidth=0.5, color=color)
ax.set(ylim=(-22, 179))
plt.axhline(y=179-len(dVNC_A1_pairs), color='r', linestyle='-')
plt.axhline(y=179-len(dVNC_A1_pairs)-len(dVNC_nonA1_pairs), color='r', linestyle='-')
plt.axhline(y=0, color='r', linestyle='-')
plt.savefig(f'VNC_interaction/plots/parallel-coordinates_all_zigzags.pdf', format='pdf', bbox_inches='tight')

# %%
# plot number of zigzag paths