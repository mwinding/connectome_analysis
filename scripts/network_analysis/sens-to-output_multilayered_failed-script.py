# %%
#

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from joblib import Parallel, delayed
from tqdm import tqdm

from pymaid_creds import url, name, password, token
import pymaid
import connectome_tools.cluster_analysis as clust
import connectome_tools.celltype as ct
import connectome_tools.process_graph as pg
import connectome_tools.process_matrix as pm

import random

dVNC_paths = pg.Prograph.open_simple_paths(f'data/paths/all_paths_sens-to-dVNC_cutoff6.csv.gz')

dVNC_paths_sample = random.sample(dVNC_paths, 100000)
sources = list(np.unique([x[0] for x in dVNC_paths_sample]))
dVNCs = list(np.unique([x[len(x)-1] for x in dVNC_paths_sample]))

path_lengths_list = []

for source in sources:
    for sink in dVNCs:
        # identify paths and count
        path_lengths = [len(x) for x in dVNC_paths_sample if (x[0]==source)&(x[len(x)-1]==sink)]

        path_lengths_list.append(path_lengths)

# even with this small bit of data, this is not computationally viable
# glancing at the distribution of path lengths, I think we aren't interested in this analysis anyways
# I will re-attempt with signal cascades




# %%
# individual sensory to descending neurons


