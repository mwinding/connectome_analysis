#%%
import sys
import os

os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory
sys.path.append('/Users/mwinding/repos/maggot_models')

from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

from src.data import load_metagraph
from src.visualization import CLASS_COLOR_DICT, adjplot
from src.traverse import Cascade, to_transmission_matrix
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot

import connectome_tools.cascade_analysis as casc
import connectome_tools.celltype as ct
import connectome_tools.process_matrix as pm

adj_ad = pm.Promat.pull_adj(type_adj='ad', subgraph='brain')

#%%
# pull sensory annotations and then pull associated skids
order = ['ORN', 'AN sensories', 'MN sensories', 'photoreceptors', 'thermosensories', 'v\'td', 'A1 ascending noci', 'A1 ascending mechano', 'A1 ascending proprio', 'A1 ascending class II_III']
sens = [ct.Celltype(name, pymaid.get_skids_by_annotation(f'mw {name}')) for name in order]
input_skids_list = [x.get_skids() for x in sens]
input_skids = [val for sublist in input_skids_list for val in sublist]

output_names = pymaid.get_annotated('mw brain outputs').name
output_skids_list = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain outputs').name))
output_skids = [val for sublist in output_skids_list for val in sublist]

#%%
# cascades from each sensory modality

p = 0.05
max_hops = 10
n_init = 100
simultaneous = True
adj=adj_ad

input_hit_hist_list = casc.Cascade_Analyzer.run_cascades_parallel(source_skids_list=input_skids_list, source_names = order, stop_skids=output_skids, 
                                                                    adj=adj_ad, p=p, max_hops=max_hops, n_init=n_init, simultaneous=simultaneous)


# %%
# pairwise thresholding of hit_hists

threshold = n_init/2
hops = 7

sens_types = [ct.Celltype(hit_hist.get_name(), list(hit_hist.pairwise_threshold(threshold=threshold, hops=hops))) for hit_hist in input_hit_hist_list]
sens_types_analyzer = ct.Celltype_Analyzer(sens_types)

upset_threshold = 20
upset_threshold_dual_cats = 10
path = f'cascades/plots/sens-cascades_upset_{upset_threshold}-threshold'
upset_members, members_selected, skids_excluded = sens_types_analyzer.upset_members(threshold=upset_threshold, path=path, plot_upset=True, 
                                                                                    exclude_singletons_from_threshold=True, exclude_skids=input_skids, threshold_dual_cats=upset_threshold_dual_cats)

# check what's in the skids_excluded group
# this group is assorted integrative that were excluded from the plot for simplicity
_, celltypes = ct.Celltype_Analyzer.default_celltypes()
test_excluded = ct.Celltype_Analyzer([ct.Celltype('excluded25', skids_excluded)])
test_excluded.set_known_types(celltypes)
excluded_data = test_excluded.memberships()

# plot cell identity of all of these categories
upset_analyzer = ct.Celltype_Analyzer(members_selected)
upset_analyzer.set_known_types(celltypes)
upset_data = upset_analyzer.memberships() # the data is all out of order compared to upset plot

# plot cell identity of labelled line vs integrative

# %%
# plot hops from input labelled line vs integrative