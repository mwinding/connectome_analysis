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
# pull sensory annotations and then associated skids
ORN_skids = pymaid.get_skids_by_annotation('mw ORN')
output_skids = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain outputs').name))
output_skids = [val for sublist in output_skids for val in sublist]

ORN2_skids = pymaid.get_skids_by_annotation('mw ORN 2nd_order')

#%%
# better understanding how stop nodes work in the context of complex cascades
from src.traverse import Cascade, to_transmission_matrix
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot

# convert skids to indices
ORN_indices = np.where([x in ORN_skids for x in mg.meta.index])[0]
output_indices = np.where([x in output_skids for x in mg.meta.index])[0]
ORN2_indices = np.where([x in ORN2_skids for x in mg.meta.index])[0]

p = 0.05
max_hops = 20
n_init = 100
simultaneous = True
transition_probs = to_transmission_matrix(adj, p)
start_nodes = ORN_indices

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    max_hops=max_hops,
    allow_loops = False,
    n_init=n_init,
    simultaneous=simultaneous,
)
hit_hist = cdispatch.multistart(start_nodes)

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    max_hops=max_hops,
    allow_loops = False,
    n_init=n_init,
    stop_nodes = np.where(hit_hist[:, 1]>0)[0],
    simultaneous=simultaneous,
)
hit_hist_stop = cdispatch.multistart(start_nodes)

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    max_hops=max_hops,
    allow_loops = False,
    n_init=n_init,
    stop_nodes = np.where(hit_hist[:, 1]>5)[0],
    simultaneous=simultaneous,
)
hit_hist_stop5 = cdispatch.multistart(start_nodes)

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    max_hops=max_hops,
    allow_loops = False,
    n_init=n_init,
    stop_nodes = np.where(hit_hist[:, 1]>10)[0],
    simultaneous=simultaneous,
)
hit_hist_stop10 = cdispatch.multistart(start_nodes)

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    max_hops=max_hops,
    allow_loops = False,
    n_init=n_init,
    stop_nodes = np.where(hit_hist[:, 1]>25)[0],
    simultaneous=simultaneous,
)
hit_hist_stop25 = cdispatch.multistart(start_nodes)

import os
os.system('say "code executed"')
# %%
# plot comparison 

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

fig, axs = plt.subplots(
    6, 1, figsize=(10, 10)
)

ax = axs[0]
matrixplot(hit_hist.T, ax=ax, cbar=True)

ax = axs[1]
matrixplot(hit_hist_stop.T, ax=ax, cbar=True)

ax = axs[2]
matrixplot(hit_hist_stop5.T, ax=ax, cbar=True)

ax = axs[3]
matrixplot(hit_hist_stop10.T, ax=ax, cbar=True)

ax = axs[4]
matrixplot(hit_hist_stop25.T, ax=ax, cbar=True)

ax = axs[5]
ax.axis("off")
caption = f"Hop histogram, cascade starting at ORNs and ending at 2nd_order ORN.\n"
caption += "Plot 1 - No stops, Plot 2 - Stops at 2nd_order, \nPlot 3 - Stops at 2nd_order >5 visits, stop nodes, Plot 4 - Stops at 2nd_order >10 visits, \nPlot 5 - Stops at 2nd_order >25 visits\n"
caption += f"p={p}, simultaneous={simultaneous}."
ax.text(0, 1, caption, va="top")

plt.savefig('cascades/plots/noloops_stop_testing.pdf', format='pdf', bbox_inches='tight')

import os
os.system('say "code executed"')

# %%
# comparing loop vs no loop cascades downstream of ORNs
from src.traverse import Cascade, to_transmission_matrix
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot

# convert skids to indices
ORN_indices = np.where([x in ORN_skids for x in mg.meta.index])[0]
output_indices = np.where([x in output_skids for x in mg.meta.index])[0]

p = 0.05
max_hops = 20
n_init = 100
simultaneous = True
transition_probs = to_transmission_matrix(adj, p)
start_nodes = ORN_indices

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    max_hops=max_hops,
    allow_loops = False,
    n_init=n_init,
    simultaneous=simultaneous,
)
hit_hist = cdispatch.multistart(start_nodes)

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    max_hops=max_hops,
    allow_loops = False,
    n_init=n_init,
    stop_nodes = output_indices, 
    simultaneous=simultaneous,
)
hit_hist_stop = cdispatch.multistart(start_nodes)

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    max_hops=max_hops,
    allow_loops = True,
    n_init=n_init,
    simultaneous=simultaneous,
)
hit_hist_loop = cdispatch.multistart(start_nodes)

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    max_hops=max_hops,
    allow_loops = True,
    n_init=n_init,
    stop_nodes = output_indices,
    simultaneous=simultaneous,
)
hit_hist_stop_loop = cdispatch.multistart(start_nodes)

import os
os.system('say "code executed"')

# %%
# plot comparison of loops to no loops, with and without end points

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

fig, axs = plt.subplots(
    5, 1, figsize=(10, 10)
)

ax = axs[0]
matrixplot(hit_hist.T, ax=ax, cbar=True)

ax = axs[1]
matrixplot(hit_hist_stop.T, ax=ax, cbar=True)

ax = axs[2]
matrixplot(hit_hist_loop.T, ax=ax, cbar=True)

ax = axs[3]
matrixplot(hit_hist_stop_loop.T, ax=ax, cbar=True)

ax = axs[4]
ax.axis("off")
caption = f"Hop histogram, cascade starting at ORNs and ending at brain outputs.\n"
caption += "Plot 1 - No loops, no stop_nodes, Plot 2 - No loops, stop_nodes at brain outputs\n Plot 3 - Loops, no stop_nodes, Plot 4 - Loops, stop_nodes at brain outputs"
caption += f"p={p}, simultaneous={simultaneous}."
ax.text(0, 1, caption, va="top")

plt.savefig('cascades/plots/stoping_at_descending_loops-noloops_test.pdf', format='pdf', bbox_inches='tight')

import os
os.system('say "code executed"')

# %%
#
output_hits_loop = hit_hist_loop[output_indices]
output_hits_stop_loop = hit_hist_stop_loop[output_indices]

output_hits = hit_hist[output_indices]
output_hits_stop = hit_hist_stop[output_indices]

fig, axs = plt.subplots(
    1, 1, figsize=(6, 4)
)

axs.set_xlabel('Hops')
axs.set_ylabel('Visits to brain output neurons')
axs.set(xticks=np.arange(1,21,1))


sns.lineplot(x = range(len(output_hits_loop.sum(axis = 0))), y = output_hits_loop.sum(axis = 0), label = 'Loops allowed')
sns.lineplot(x = range(len(output_hits_stop_loop.sum(axis = 0))), y = output_hits_stop_loop.sum(axis = 0), label = 'Loops allowed, end_nodes = outputs')
sns.lineplot(x = range(len(output_hits.sum(axis = 0))), y = output_hits.sum(axis = 0), label = 'Loops not allowed')
sns.lineplot(x = range(len(output_hits_stop.sum(axis = 0))), y = output_hits_stop.sum(axis = 0), label = 'Loops not allowed, end_nodes = outputs')

plt.savefig('cascades/plots/vists_to_outputs_per_hop.pdf', format='pdf', bbox_inches='tight')

# %%
# Identfy hits at each hop level

ORN1 = mg.meta[hit_hist_stop[:, 0]>0]
ORN2 = mg.meta[hit_hist_stop[:, 1]>0]
ORN3 = mg.meta[hit_hist_stop[:, 2]>0]
ORN4 = mg.meta[hit_hist_stop[:, 3]>0]
ORN5 = mg.meta[hit_hist_stop[:, 4]>0]
ORN6 = mg.meta[hit_hist_stop[:, 5]>0]
ORN7 = mg.meta[hit_hist_stop[:, 6]>0]
ORN8 = mg.meta[hit_hist_stop[:, 7]>0]
ORN9 = mg.meta[hit_hist_stop[:, 8]>0]
ORN10 = mg.meta[hit_hist_stop[:, 9]>0]


ORN1_hits = hit_hist_stop[hit_hist_stop[:, 0]>0, 0]
ORN2_hits = hit_hist_stop[hit_hist_stop[:, 1]>0, 1]
ORN3_hits = hit_hist_stop[hit_hist_stop[:, 2]>0, 2]
ORN4_hits = hit_hist_stop[hit_hist_stop[:, 3]>0, 3]
ORN5_hits = hit_hist_stop[hit_hist_stop[:, 4]>0, 4]
ORN6_hits = hit_hist_stop[hit_hist_stop[:, 5]>0, 5]
ORN7_hits = hit_hist_stop[hit_hist_stop[:, 6]>0, 6]
ORN8_hits = hit_hist_stop[hit_hist_stop[:, 7]>0, 7]
ORN9_hits = hit_hist_stop[hit_hist_stop[:, 8]>0, 8]
ORN10_hits = hit_hist_stop[hit_hist_stop[:, 9]>0, 9]

fig, axs = plt.subplots(
    10, 1, figsize=(10, 20)
)

fig.tight_layout(pad=3.0)

ax = axs[0]
ax.set(xlim = (1, 100))
ax.set_xlabel('Hits in Hop 1')    
sns.distplot(np.append(ORN1_hits, [0]), ax = ax, bins = 100, kde = False)

ax = axs[1]
ax.set(xlim = (1, 100))
ax.set_xlabel('Hits in Hop 2')    
sns.distplot(ORN2_hits, ax = ax, bins = 100, kde = False)

ax = axs[2]
ax.set(xlim = (1, 100))
ax.set_xlabel('Hits in Hop 3')    
sns.distplot(ORN3_hits, ax = ax, bins = 100, kde = False)

ax = axs[3]
ax.set(xlim = (1, 100))
ax.set_xlabel('Hits in Hop 4')
sns.distplot(ORN4_hits, ax = ax, bins = 100, kde = False)

ax = axs[4]
ax.set(xlim = (1, 100))
ax.set_xlabel('Hits in Hop 5')
sns.distplot(ORN5_hits, ax = ax, bins = 100, kde = False)

ax = axs[5]
ax.set(xlim = (1, 100))
ax.set_xlabel('Hits in Hop 6')
sns.distplot(ORN6_hits, ax = ax, bins = 100, kde = False)

ax = axs[6]
ax.set(xlim = (1, 100))
ax.set_xlabel('Hits in Hop 7')
sns.distplot(ORN7_hits, ax = ax, bins = 100, kde = False)

ax = axs[7]
ax.set(xlim = (1, 100))
ax.set_xlabel('Hits in Hop 8')
sns.distplot(ORN8_hits, ax = ax, bins = 100, kde = False)

ax = axs[8]
ax.set(xlim = (1, 100))
ax.set_xlabel('Hits in Hop 9')
sns.distplot(ORN9_hits, ax = ax, bins = 100, kde = False)

ax = axs[9]
ax.set(xlim = (1, 100))
ax.set_xlabel('Hits in Hop 10')
sns.distplot(ORN10_hits, ax = ax, bins = 100, kde = False)

plt.savefig('cascades/plots/hits_per_hop_no_loops.pdf', format='pdf', bbox_inches='tight')
# %%
