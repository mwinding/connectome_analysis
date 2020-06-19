#%%
import os
import sys
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/maggot_models/')
    print(os.getcwd())
except:
    pass

# %%
sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
from pymaid_creds import url, name, password, token
import pymaid

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from graspy.plot import gridplot, heatmap
from graspy.utils import binarize, pass_to_ranks
from src.data import load_metagraph
from src.visualization import CLASS_COLOR_DICT, adjplot

rm = pymaid.CatmaidInstance(url, name, password, token)

mg = load_metagraph("G", version="2020-04-23", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg.calculate_degrees(inplace=True)

adj = mg.adj  # adjacency matrix from the "mg" object
# %%
from src.traverse import to_markov_matrix, RandomWalk

# Random walk test
ORN = pymaid.get_skids_by_annotation('mw ORN')
ORN_index = np.where(mg.meta.index == ORN[0])[0][0] # index associated with skid; will use in rw

transition_probs = to_markov_matrix(adj)
rw = RandomWalk(transition_probs, allow_loops=False, stop_nodes=[], max_hops=20)

np.random.seed(0)
rw.start(ORN_index)
print(rw.traversal_)  

np.random.seed(1)
rw.start(ORN_index)
print(rw.traversal_)


# %%
# Cascade test
from src.traverse import Cascade, to_transmission_matrix

p = 0.05
transition_probs = to_transmission_matrix(adj, p)
casc = Cascade(transition_probs, max_hops = 20)

np.random.seed(0)
casc.start(ORN_index)
print(casc.traversal_)

# %%
# Cascade test with multiple neurons

ORNs_index = np.where([x in ORN for x in mg.meta.index])[0] # indices associated with all ORNs

p = 0.05
transition_probs = to_transmission_matrix(adj, p)
casc = Cascade(transition_probs, max_hops = 20)

np.random.seed(0)
casc.start(ORNs_index)
print(casc.traversal_)
print()


# %%
# Plotting test
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot

p = 0.05
max_hops = 20
n_init = 100
simultaneous = True
transition_probs = to_transmission_matrix(adj, p)
start_nodes = ORNs_index

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    allow_loops=False,
    max_hops=max_hops,
    n_init=n_init,
    simultaneous=simultaneous,
)
hit_hist = cdispatch.multistart(start_nodes)

fig, axs = plt.subplots(
    3, 1, figsize=(10, 10), gridspec_kw=dict(height_ratios=[0.45, 0.45, 0.1])
)

ax = axs[0]
matrixplot(hit_hist.T, ax=ax, cbar=True)
ax.set_xlabel("Block")
ax.set_yticks(np.arange(1, max_hops + 1) - 0.5)
ax.set_yticklabels(np.arange(1, max_hops + 1))
ax.set_ylabel("Hops")
ax = axs[1]
matrixplot(np.log10(hit_hist.T + 1), ax=ax, cbar=True)
ax.set_yticks(np.arange(1, max_hops + 1) - 0.5)
ax.set_yticklabels(np.arange(1, max_hops + 1))
ax.set_ylabel("Hops")
ax = axs[2]
ax.axis("off")
caption = f"Figure x: Hop histogram, cascade on feedforward SBM.\n"
caption += "Top - linear scale, Bottom - Log10 scale.\n"
caption += f"p={p}, simultaneous={simultaneous}."
ax.text(0, 1, caption, va="top")
#stashfig(f"hop-hist-cascade-p{p}-simult{simultaneous}")

# %%
