#%%
import sys
import os

os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory
sys.path.append('/Users/mwinding/repos/maggot_models')

from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx

from src.data import load_metagraph
from src.visualization import CLASS_COLOR_DICT, adjplot
from src.traverse import Cascade, to_transmission_matrix
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot

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

# %%
# two hop partners upstream and downstream of output types

output_names = ['mw dVNC', 'mw dSEZ', 'mw RGN']
outputs = [pymaid.get_skids_by_annotation(x) for x in output_names]
all_outputs = [x for sublist in outputs for x in sublist]

# use pregenerated edge list
edges = pd.read_csv('data/edges_threshold/pairwise-threshold_ad_all-edges.csv', index_col=0)

# upstream and downstream 2-hops of outputs
outputs_downstream = [pm.Promat.downstream_multihop(edges=edges, sources=output, hops=2, exclude_skids_from_source=all_outputs) for output in outputs]
outputs_upstream = [pm.Promat.upstream_multihop(edges=edges, sources=output, hops=2) for output in outputs]

# upstream of pre-outputs; remove outputs from pre-outputs before checking
pre_outputs = [np.setdiff1d(x[0], all_outputs) for x in outputs_upstream]
pre_outputs_downstream = [pm.Promat.downstream_multihop(edges=edges, sources=output, hops=1) for output in pre_outputs]

# %%
# cascades from output types and pre-output types
import pickle

output_names = ['mw dVNC', 'mw dSEZ', 'mw RGN']
outputs = [pymaid.get_skids_by_annotation(annot) for annot in output_names]
preoutput_names = ['mw pre-dVNC 1%', 'mw pre-dSEZ 1%', 'mw pre-RGN 1%']
preoutputs = [pymaid.get_skids_by_annotation(annot) for annot in preoutput_names]

adj=pm.Promat.pull_adj('ad', subgraph='brain and accessory')
p=0.05
max_hops = 10
n_init = 1000
simultaneous = True

#outputs_hit_hists = casc.Cascade_Analyzer.run_cascades_parallel(outputs, output_names, stop_skids=[], adj=adj, p=p, max_hops=max_hops, n_init=n_init, simultaneous=simultaneous)
#preoutputs_hit_hists = casc.Cascade_Analyzer.run_cascades_parallel(preoutputs, preoutput_names, stop_skids=[], adj=adj, p=p, max_hops=max_hops, n_init=n_init, simultaneous=simultaneous)
#pickle.dump(outputs_hit_hists, open('data/cascades/outputs-cascades_1000-n_init.p', 'wb'))
#pickle.dump(preoutputs_hit_hists, open('data/cascades/preoutputs-cascades_1000-n_init.p', 'wb'))

outputs_hit_hists = pickle.load(open('data/cascades/outputs-cascades_1000-n_init.p', 'rb'))
preoutputs_hit_hists = pickle.load(open('data/cascades/preoutputs-cascades_1000-n_init.p', 'rb'))

# %%
# threshold data and identify neurons of interest

threshold = n_init/2
hops = 3

outputs_fb = [hit_hist.pairwise_threshold(threshold=threshold, hops=hops) for hit_hist in outputs_hit_hists]
preoutputs_fb = [hit_hist.pairwise_threshold(threshold=threshold, hops=hops) for hit_hist in preoutputs_hit_hists]

# %%
# organize data into celltype objects, generate membership matrices

# generate official cell types
exclusive = False

if(exclusive):
    _, celltypes = ct.Celltype_Analyzer.default_celltypes() # exclusive celltypes

if(exclusive==False):
    # non-exclusive celltypes
    celltypes_names = ['sensories', 'PNs', 'LNs', 'LHNs', 'FFNs', 'MBINs', 'KCs', 'MBONs', 'MB-FBNs', 'CNs', 'ascendings', 'pre-dSEZs', 'pre-dVNCs', 'RGNs', 'dSEZs', 'dVNCs']
    celltypes_annots = ['mw brain ' + x for x in celltypes_names]
    celltypes_skids = [ct.Celltype_Analyzer.get_skids_from_meta_annotation(annot) for annot in celltypes_annots]
    celltypes = [ct.Celltype(celltypes_names[i], skids) for i, skids in enumerate(celltypes_skids)]

# feedforward
names = ['pre-pre-dVNC', 'pre-dVNC', 'dVNC']
skids = [outputs_upstream[0][1], outputs_upstream[0][0], outputs[0]]
dVNC_ff = list(map(lambda x: ct.Celltype(*x), zip(names, skids)))
dVNC_ff = ct.Celltype_Analyzer(dVNC_ff)
dVNC_ff.set_known_types(celltypes)
dVNC_ff_mem = dVNC_ff.memberships()

names = ['pre-pre-dSEZ', 'pre-dSEZ', 'dSEZ']
skids = [outputs_upstream[1][1], outputs_upstream[1][0], outputs[1]]
dSEZ_ff = list(map(lambda x: ct.Celltype(*x), zip(names, skids)))
dSEZ_ff = ct.Celltype_Analyzer(dSEZ_ff)
dSEZ_ff.set_known_types(celltypes)
dSEZ_ff_mem = dSEZ_ff.memberships()

names = ['pre-pre-RGN', 'pre-RGN', 'RGN']
skids = [outputs_upstream[2][1], outputs_upstream[2][0], outputs[2]]
RGN_ff = list(map(lambda x: ct.Celltype(*x), zip(names, skids)))
RGN_ff = ct.Celltype_Analyzer(RGN_ff)
RGN_ff.set_known_types(celltypes)
RGN_ff_mem = RGN_ff.memberships()

# feedfeedback
names = ['ds-pre-dVNC', 'cascade-pre-dVNC', 'ds-dVNC', 'cascade-dVNC']
skids = [pre_outputs_downstream[0][0], preoutputs_fb[0], outputs_downstream[0][0], outputs_fb[0]]
dVNC_fb = list(map(lambda x: ct.Celltype(*x), zip(names, skids)))
dVNC_fb = ct.Celltype_Analyzer(dVNC_fb)
dVNC_fb.set_known_types(celltypes)
dVNC_fb_mem = dVNC_fb.memberships()

names = ['ds-pre-dSEZ', 'cascade-pre-dSEZ', 'ds-dSEZ', 'cascade-dSEZ']
skids = [pre_outputs_downstream[1][0], preoutputs_fb[1], outputs_downstream[1][0], outputs_fb[1]]
dSEZ_fb = list(map(lambda x: ct.Celltype(*x), zip(names, skids)))
dSEZ_fb = ct.Celltype_Analyzer(dSEZ_fb)
dSEZ_fb.set_known_types(celltypes)
dSEZ_fb_mem = dSEZ_fb.memberships()

names = ['ds-pre-RGN', 'cascade-pre-RGN', 'cascade-RGN']
skids = [pre_outputs_downstream[2][0], preoutputs_fb[2], outputs_fb[2]]
RGN_fb = list(map(lambda x: ct.Celltype(*x), zip(names, skids)))
RGN_fb = ct.Celltype_Analyzer(RGN_fb)
RGN_fb.set_known_types(celltypes)
RGN_fb_mem = RGN_fb.memberships()

# %%
# plot cell types within
import cmasher as cmr
cbar = False
cell_height=0.1
cell_width=0.2

# modify 'Blues' cmap to have a white background
cmap = plt.cm.get_cmap('Blues')
blue_cmap = cmap(np.linspace(0, 1, 20))
blue_cmap[0] = np.array([1, 1, 1, 1])
blue_cmap = mpl.colors.LinearSegmentedColormap.from_list(name='New_Blues', colors=blue_cmap)

cmap = blue_cmap
vmax = 0.6
ff = [dVNC_ff_mem, dSEZ_ff_mem, RGN_ff_mem]

for i, df in enumerate(ff):
    fig, ax = plt.subplots(1,1,figsize=(cell_width * df.shape[1], cell_height * df.shape[0]))
    sns.heatmap(df, annot=True, fmt='.0%', ax=ax, cmap=cmap, cbar=cbar, vmax=vmax)
    plt.savefig(f'cascades/feedback_through_brain/plots/ff_celltypes_{output_names[i]}_exclusive{exclusive}_cascade-hops{hops}.pdf', bbox_inches='tight')


# modify 'Oranges' cmap to have a white background
cmap = plt.cm.get_cmap('Oranges')
orange_cmap = cmap(np.linspace(0, 1, 20))
orange_cmap[0] = np.array([1, 1, 1, 1])
orange_cmap = mpl.colors.LinearSegmentedColormap.from_list(name='New_Oranges', colors=orange_cmap)

cmap = orange_cmap
vmax = 0.4
fb = [dVNC_fb_mem, dSEZ_fb_mem, RGN_fb_mem]
for i, df in enumerate(fb):
    fig, ax = plt.subplots(1,1,figsize=(cell_width * df.shape[1], cell_height * df.shape[0]))
    sns.heatmap(df, annot=True, fmt='.0%', ax=ax, cmap=cmap, cbar=cbar, vmax=vmax)
    plt.savefig(f'cascades/feedback_through_brain/plots/fb_celltypes_{output_names[i]}_exclusive{exclusive}_cascade-hops{hops}.pdf', bbox_inches='tight')

# %%
# old sets of celltypes

'''
# feedforward
names = ['pre-pre-dVNC', 'pre-dVNC', 'dVNC']
skids = [outputs_upstream[0][1], outputs_upstream[0][0], outputs[0]]
dVNC_ff = list(map(lambda x: ct.Celltype(*x), zip(names, skids)))
dVNC_ff = ct.Celltype_Analyzer(dVNC_ff)
dVNC_ff.set_known_types(celltypes)
dVNC_ff_mem = dVNC_ff.memberships()

names = ['pre-pre-dSEZ', 'pre-dSEZ', 'dSEZ']
skids = [outputs_upstream[1][1], outputs_upstream[1][0], outputs[1]]
dSEZ_ff = list(map(lambda x: ct.Celltype(*x), zip(names, skids)))
dSEZ_ff = ct.Celltype_Analyzer(dSEZ_ff)
dSEZ_ff.set_known_types(celltypes)
dSEZ_ff_mem = dSEZ_ff.memberships()

names = ['pre-pre-RGN', 'pre-RGN', 'RGN']
skids = [outputs_upstream[2][1], outputs_upstream[2][0], outputs[2]]
RGN_ff = list(map(lambda x: ct.Celltype(*x), zip(names, skids)))
RGN_ff = ct.Celltype_Analyzer(RGN_ff)
RGN_ff.set_known_types(celltypes)
RGN_ff_mem = RGN_ff.memberships()

# feedback
names = ['ds-pre-dVNC', 'ds-dVNC', 'ds-ds-dVNC']
skids = [pre_outputs_downstream[0][0], outputs_downstream[0][0], outputs_downstream[0][1]]
dVNC_fb = list(map(lambda x: ct.Celltype(*x), zip(names, skids)))
dVNC_fb = ct.Celltype_Analyzer(dVNC_fb)
dVNC_fb.set_known_types(celltypes)
dVNC_fb_mem = dVNC_fb.memberships()

names = ['ds-pre-dSEZ', 'ds-dSEZ', 'ds-ds-dSEZ']
skids = [pre_outputs_downstream[1][0], outputs_downstream[1][0], outputs_downstream[1][1]]
dSEZ_fb = list(map(lambda x: ct.Celltype(*x), zip(names, skids)))
dSEZ_fb = ct.Celltype_Analyzer(dSEZ_fb)
dSEZ_fb.set_known_types(celltypes)
dSEZ_fb_mem = dSEZ_fb.memberships()

names = ['ds-pre-RGN','ds-pre-RGN','ds-pre-RGN']
skids = [pre_outputs_downstream[2][0],pre_outputs_downstream[2][0],pre_outputs_downstream[2][0]]
RGN_fb = list(map(lambda x: ct.Celltype(*x), zip(names, skids)))
RGN_fb = ct.Celltype_Analyzer(RGN_fb)
RGN_fb.set_known_types(celltypes)
RGN_fb_mem = RGN_fb.memberships()
'''