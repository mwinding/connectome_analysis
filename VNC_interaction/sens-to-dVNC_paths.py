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

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 5})

rm = pymaid.CatmaidInstance(url, name, password, token)

mg = load_metagraph("Gad", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg.calculate_degrees(inplace=True)

adj = mg.adj  # adjacency matrix from the "mg" object
#adj = pd.DataFrame(adj, columns = mg.meta.index, index = mg.meta.index)

pairs = pd.read_csv('VNC_interaction/data/pairs-2020-10-26.csv', header = 0) # import pairs

# %%
# import various cell types
from connectome_tools.process_matrix import Adjacency_matrix, Promat

input_names = pymaid.get_annotated('mw brain inputs').name
general_names = ['ORN', 'thermo', 'photo', 'AN', 'MN', 'vtd', 'noci']
input_skids_list = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain inputs').name))
input_skids = [val for sublist in input_skids_list for val in sublist]

dVNC = pymaid.get_skids_by_annotation('mw dVNC')
dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')
RGN = pymaid.get_skids_by_annotation('mw RGN')
all_outputs = dVNC+dSEZ+RGN

dVNC_pairs = Promat.extract_pairs_from_list(dVNC, pairs)[0]

#%%
# cascades from each sensory type, ending at outputs 
from src.traverse import Cascade, to_transmission_matrix
from src.traverse import to_markov_matrix, RandomWalk
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot
from joblib import Parallel, delayed

def run_cascade(i, cdispatch):
    return(cdispatch.multistart(start_nodes = i))

# convert skids to indices
output_indices_list = []
for i in range(0, len(dVNC_pairs)):
    skids = list(dVNC_pairs.iloc[i])
    indices = np.where([x in skids for x in mg.meta.index])[0]
    output_indices_list.append(indices)

input_indices_list = []
for skids in input_skids_list:
    indices = np.where([x in skids for x in mg.meta.index])[0]
    input_indices_list.append(indices)

all_outputs_indices = np.where([x in all_outputs for x in mg.meta.index])[0]
all_input_indices = np.where([x in input_skids for x in mg.meta.index])[0]

p = 0.05
max_hops = 11
n_init = 100
simultaneous = True
transition_probs = to_transmission_matrix(adj=adj, p=p)

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    stop_nodes = all_outputs_indices,
    max_hops=max_hops,
    allow_loops = False,
    n_init=n_init,
    simultaneous=simultaneous,
)

all_sens_hit_hist = run_cascade(all_input_indices, cdispatch)
sens_hit_hist_list = Parallel(n_jobs=-1)(delayed(run_cascade)(i, cdispatch) for i in input_indices_list)

# %%
# analyze results of all-sensory cascade

from connectome_tools.cascade_analysis import Cascade_Analyzer, Celltype, Celltype_Analyzer

all_sens_cascade = Cascade_Analyzer(sum(sens_hit_hist_list), mg, pairs)

hop_data_raw = []
hop_data_summed = []
for i in range(0, len(dVNC_pairs)):
    raw = all_sens_cascade.skid_hit_hist.loc[dVNC_pairs.iloc[i].values]
    summed = [dVNC_pairs.iloc[i].leftid] + list(raw.sum(axis=0))

    hop_data_raw.append(raw)
    hop_data_summed.append(summed)

hop_data_summed = pd.DataFrame(hop_data_summed, columns = ['pair_id'] + [x for x in range(0, max_hops)])
hop_data_summed_index = hop_data_summed.set_index('pair_id')

hop_data_scatter = []
for j in range(1, len(hop_data_summed_index.columns)):
    for i in range(0, len(hop_data_summed_index.index)):
        #if(hop_data_summed_index.iloc[i, j]!=0):
        hop_data_scatter.append([hop_data_summed_index.iloc[i, j], j])

hop_data_scatter = pd.DataFrame(hop_data_scatter, columns = ['count', 'hops'])

fig, axs = plt.subplots(
    2, 1, figsize=(1.5, 2)
)
vmax = 800
fig.tight_layout(pad=0)
ax=axs[0]
#sns.boxplot(x='hops', y='count', data=hop_data_scatter, ax=ax, fliersize=0, linewidth=0.25, whis=np.inf)
sns.stripplot(x='hops', y='count', data=hop_data_scatter, ax=ax, size=0.75, jitter=0.2)
ax.set(xticks=[], ylim=(0, vmax))
ax.set_xlabel('')

ax=axs[1]
sns.heatmap(hop_data_summed_index.iloc[:, 1:len(hop_data_summed_index.columns)], ax=ax, cmap = 'Blues', cbar=False, vmax=vmax)
plt.savefig('VNC_interaction/plots/dVNC_upstream/paths_from_sens.pdf', format='pdf', bbox_inches='tight')

# %%
# analyze individual sensory cascades to descending neurons

from connectome_tools.cascade_analysis import Cascade_Analyzer, Celltype, Celltype_Analyzer

sens_cascades_list = [Cascade_Analyzer(sens_hist_list, mg, pairs) for sens_hist_list in sens_hit_hist_list]

hop_data_summed_list = []
hop_data_summed_index_list = []
for sens_cascade in sens_cascades_list:
    hop_data_raw = []
    hop_data_summed = []
    for i in range(0, len(dVNC_pairs)):
        raw = sens_cascade.skid_hit_hist.loc[dVNC_pairs.iloc[i].values]
        summed = [dVNC_pairs.iloc[i].leftid] + list(raw.sum(axis=0))

        hop_data_raw.append(raw)
        hop_data_summed.append(summed)

    hop_data_summed = pd.DataFrame(hop_data_summed, columns = ['pair_id'] + [x for x in range(0, max_hops)])
    hop_data_summed_index = hop_data_summed.set_index('pair_id')

    hop_data_summed_list.append(hop_data_summed)
    hop_data_summed_index_list.append(hop_data_summed_index)

hop_data_scatter_list = []
for hop_data_summed_index in hop_data_summed_index_list:
    hop_data_scatter = []
    for j in range(1, len(hop_data_summed_index.columns)):
        for i in range(0, len(hop_data_summed_index.index)):
            hop_data_scatter.append([hop_data_summed_index.iloc[i, j], j])

    hop_data_scatter = pd.DataFrame(hop_data_scatter, columns = ['count', 'hops'])
    hop_data_scatter_list.append(hop_data_scatter)
    
for i in range(0, len(hop_data_scatter_list)):
    fig, axs = plt.subplots(
        2, 1, figsize=(1.5, 2)
    )
    vmax = 200
    fig.tight_layout(pad=0)
    ax=axs[0]
    sns.stripplot(x='hops', y='count', data=hop_data_scatter_list[i], ax=ax, size=0.75, jitter=0.2)
    ax.set(xticks=[], ylim=(0, vmax))
    ax.set_xlabel('')

    ax=axs[1]
    sns.heatmap(hop_data_summed_index_list[i].iloc[:, 1:len(hop_data_summed_index_list[i].columns)], ax=ax, cmap = 'Blues', cbar=False, vmax=vmax)
    plt.savefig(f'VNC_interaction/plots/dVNC_upstream/paths_to_dVNCs_from_{general_names[i]}.pdf', format='pdf', bbox_inches='tight')

# %%
# parallel coordinate plot
# not great 

import plotly.express as px
from pandas.plotting import parallel_coordinates

fig, axs = plt.subplots(
    1, 1, figsize=(6, 6)
)

color = 'blue'
ax=axs
parallel_coordinates(hop_data_summed, class_column = 'pair_id', ax = ax, alpha = 0.75, linewidth = 0.75)

# parallel coordinate plot of each sensory modality

fig, axs = plt.subplots(
    1, 1, figsize=(6, 6)
)
colors = ['blue', 'red', 'purple', 'orange', 'gray', 'yellow', 'violet']
for i, hop_data_summed in enumerate(hop_data_summed_list):
    color = 'blue'
    ax=axs
    #ax.set(ylim = (0, 100))
    parallel_coordinates(hop_data_summed, class_column = 'pair_id', ax = ax, alpha = 0.5, linewidth = 0.75, color=colors[i])

# %%
