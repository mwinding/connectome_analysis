#%%
import os
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/maggot_models/')
    print(os.getcwd())
except:
    pass

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from graspy.plot import gridplot, heatmap
from graspy.utils import binarize, pass_to_ranks
from src.data import load_metagraph
from src.visualization import CLASS_COLOR_DICT, adjplot

mg = load_metagraph("G", version="2020-04-23", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg.calculate_degrees(inplace=True)

adj = mg.adj  # adjacency matrix from the "mg" object
# %%
# Random walk test

from src.traverse import to_markov_matrix, RandomWalk

transition_probs = to_markov_matrix(adj)  # row normalize!
rw = RandomWalk(transition_probs, allow_loops=True, stop_nodes=[], max_hops=10)

np.random.seed(8888)
rw.start(0)
print(rw.traversal_)  # this one stops at max hops
print()

np.random.seed(2222)
rw.start(0)
print(rw.traversal_)  # note that this one lands on a node with no output (41) and stops


# %%
