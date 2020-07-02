#%%
import os
import sys
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:
    pass

# %%
import sys
sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/maggot_models/')
sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
from pymaid_creds import url, name, password, token
import pymaid

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from graspy.plot import gridplot, heatmap
from graspy.utils import binarize, pass_to_ranks
from src.data import load_metagraph
from src.visualization import CLASS_COLOR_DICT, adjplot

rm = pymaid.CatmaidInstance(url, name, password, token)

mg = load_metagraph("Gad", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
#mg = load_metagraph("G", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg.calculate_degrees(inplace=True)

adj = mg.adj  # adjacency matrix from the "mg" object

#%% 

# identify left and right side for each skid category
def split_hemilateral_to_indices(skids, left, right, mg):
    intersect_left = np.intersect1d(skids, left)
    indices_left = np.where([x in intersect_left for x in mg.meta.index])[0]
    intersect_right = np.intersect1d(skids, right)
    indices_right = np.where([x in intersect_right for x in mg.meta.index])[0]

    return(indices_left, indices_right, intersect_left, intersect_right)

ORN_skids = pymaid.get_skids_by_annotation('mw ORN')
dVNC_skids = pymaid.get_skids_by_annotation('mw dVNC')

left = pymaid.get_skids_by_annotation('mw left')
right = pymaid.get_skids_by_annotation('mw right')

output_skids = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain outputs').name))
output_skids = [val for sublist in output_skids for val in sublist]

ORN_indices_left, ORN_indices_right, ORN_left, ORN_right = split_hemilateral_to_indices(ORN_skids, left, right, mg)
dVNC_indices_left, dVNC_indices_right, dVNC_left, dVNC_right = split_hemilateral_to_indices(dVNC_skids, left, right, mg)

#%%
from src.traverse import Cascade, to_transmission_matrix
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot
import numpy.random as random

def static_random_subset_cascade(s_indices, r_indices, subset_number, n_init, cdispatch):
# combination of a randomized subset of indices and a static subset will be used for cascade start_nodes
# make sure that input cdispath is set with n_init = 1

    hit_hist_list = []
    random_indices_list = []
    for i in range(0, n_init):
        random.seed(i)
        random_nums = random.choice(len(r_indices), subset_number, replace = False)
        random_indices = r_indices[random_nums]
        all_indices = np.concatenate([random_indices, s_indices])
        subset_hit_hist = cdispatch.multistart(start_nodes = all_indices)

        hit_hist_list.append(subset_hit_hist)
        random_indices_list.append(all_indices)

    return(sum(hit_hist_list), random_indices_list)

p = 0.05
max_hops = 10
n_init = 1
simultaneous = True
transition_probs = to_transmission_matrix(adj, p)

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    stop_nodes = output_indices,
    max_hops=max_hops,
    allow_loops = False,
    n_init=n_init,
    simultaneous=simultaneous,
)

input_indices_left = ORN_indices_left
its = 100

num_full = int(np.round(len(ORN_indices_left)))
num_75L = int(np.round(len(ORN_indices_left)*3/4))
num_50L = int(np.round(len(ORN_indices_left)/2))
num_25L = int(np.round(len(ORN_indices_left)/4))
num_10L = int(np.round(len(ORN_indices_left)/10))
num_5L = int(np.round(len(ORN_indices_left)/20))

fullR_fullL_hist, fullR_fullL_indices = static_random_subset_cascade(ORN_indices_right, ORN_indices_left, num_full, its, cdispatch)
fullR_75L_hist, fullR_75L_indices = static_random_subset_cascade(ORN_indices_right, ORN_indices_left, num_75L, its, cdispatch)
fullR_50L_hist, fullR_50L_indices = static_random_subset_cascade(ORN_indices_right, ORN_indices_left, num_50L, its, cdispatch)
fullR_25L_hist, fullR_25L_indices = static_random_subset_cascade(ORN_indices_right, ORN_indices_left, num_25L, its, cdispatch)
fullR_10L_hist, fullR_10L_indices = static_random_subset_cascade(ORN_indices_right, ORN_indices_left, num_10L, its, cdispatch)
fullR_5L_hist, fullR_5L_indices = static_random_subset_cascade(ORN_indices_right, ORN_indices_left, num_5L, its, cdispatch)


import os
os.system('say "code executed"')

# %%
fig, axs = plt.subplots(
    3, 1, figsize=(6, 20)
)

fig.tight_layout(pad=2.5)
ax = axs[0]
sns.heatmap(fullR_fullL_hist[dVNC_indices_left], ax = ax)

ax = axs[1]
sns.heatmap(fullR_10L_hist[dVNC_indices_left], ax = ax)

ax = axs[2]
sns.heatmap((fullR_fullL_hist[dVNC_indices_left] - fullR_10L_hist[dVNC_indices_left]), ax = ax)

#fig.savefig('cascades/interhemisphere_plots/assymetric_input_test.pdf', format='pdf', bbox_inches='tight')


# %%
