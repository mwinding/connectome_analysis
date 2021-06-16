#%%
import os
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import connectome_tools.celltype as ct
import connectome_tools.process_graph as pg
import connectome_tools.process_matrix as pm
import connectome_tools.process_skeletons as skel
import navis

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

adj = pm.Promat.pull_adj(type_adj='ad', subgraph='brain')
inputs = pd.read_csv('data/graphs/inputs.csv', index_col=0)
adj_mat = pm.Adjacency_matrix(adj, inputs, 'ad')
pairs = pm.Promat.get_pairs()

# %%
# remove LNs and outputs from each sensory 2nd-order neuropil to identify PNs

order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-warm', 'thermo-cold', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']
order2 = [pymaid.get_skids_by_annotation(f'mw {celltype} 2nd_order') for celltype in order]

LNs = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain LNs')
outputs = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain outputs')

order2 = [list(np.setdiff1d(skids, LNs + outputs)) for skids in order2]

order2_names = [f'mw {celltype} 2nd_order PN' for celltype in order]
#[pymaid.add_annotations(skids, order2_names[i]) for i, skids in enumerate(order2)]
#[pymaid.add_meta_annotations(name, 'mw brain inputs 2nd_order PN') for name in order2_names]
'''
# %%
# distance between centroid of axon/dendrite in LNs and PNs
uPNs = pymaid.get_skids_by_annotation('mw uPN')
LNs = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain LNs')
order2 = [uPNs] + [LNs] + order2
order2_names = ['uPNs', 'LNs'] + order2_names

order2_split_axon_dendrites = [skel.get_connectors_group(skids, output_separate=True) for skids in order2]
# list of 2nd_order groups; list of axon outputs, dendrite inputs; list of individual skids

axon_dendrite_centroid_order2 = []
for name_index, group in enumerate(order2_split_axon_dendrites):

    axon_outputs = group[0]
    dendrite_inputs = group[1]

    # all will be in microns
    axon_dendrite_centroids = []
    for i in range(0, len(axon_outputs)):
        axon_x = axon_outputs[i].mean(axis=0).x/1000
        axon_y = axon_outputs[i].mean(axis=0).y/1000
        axon_z = axon_outputs[i].mean(axis=0).z/1000
    
        dendrite_x = dendrite_inputs[i].mean(axis=0).x/1000
        dendrite_y = dendrite_inputs[i].mean(axis=0).y/1000
        dendrite_z = dendrite_inputs[i].mean(axis=0).z/1000

        distance = ((dendrite_x-axon_x)**2 + (dendrite_y-axon_y)**2 + (dendrite_z-axon_z)**2)**(1/2)
        axon_dendrite_centroids.append([order2_names[name_index], order2[name_index][i], (axon_x, axon_y, axon_z), (dendrite_x, dendrite_y, dendrite_z), distance])

    axon_dendrite_centroid_order2.append(pd.DataFrame(axon_dendrite_centroids, columns = ['modality', 'skid', 'axon_centroids', 'dendrite_centroids', 'distance']))
'''
# %%
# plot distance between axon and dendrite

fig, ax = plt.subplots(1,1,figsize=(4,2))
sns.violinplot(data=pd.concat(axon_dendrite_centroid_order2), x='modality', y='distance', orient='v', scale='width', ax=ax)
plt.xticks(rotation=45, ha='right')
# %%
# 

uPNs = pymaid.get_skids_by_annotation('mw uPN')
LNs = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain LNs')
order2 = [uPNs] + [LNs] + order2
order2_names = ['uPNs', 'LNs'] + order2_names

order2_data = []
for i, skids in enumerate(order2):
    df = []
    for skid in skids:
        df.append(skel.axon_dendrite_centroid(skid))

    df = pd.concat(df)
    df['celltype'] = len(df.index)*[order2_names[i]]
    order2_data.append(df)

# %%
# plot distance between axon and dendrite

fig, ax = plt.subplots(1,1,figsize=(4,2))
sns.violinplot(data=pd.concat(order2_data), x='celltype', y='distance', orient='v', scale='width', ax=ax)
plt.xticks(rotation=45, ha='right')
# %%
