#%%
import os

try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:

    pass

#%%
import sys
sys.path.append("/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/")

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

aa = pd.read_csv('data/axon-axon.csv', header = 0, index_col = 0)
ad = pd.read_csv('data/axon-dendrite.csv', header = 0, index_col = 0)
dd = pd.read_csv('data/dendrite-dendrite.csv', header = 0, index_col = 0)
da = pd.read_csv('data/dendrite-axon.csv', header = 0, index_col = 0)

# %%
# edge numbers
aa_bin = aa
ad_bin = ad
dd_bin = dd
da_bin = da


aa_bin[aa > 0] = 1
ad_bin[ad > 0] = 1
dd_bin[dd > 0] = 1
da_bin[da > 0] = 1

aa_edge = np.matrix(aa_bin).sum(axis=None)
ad_edge = np.matrix(ad_bin).sum(axis=None)
dd_edge = np.matrix(dd_bin).sum(axis=None)
da_edge = np.matrix(da_bin).sum(axis=None)

# Density

total_edges = len(ad.index)*len(ad.index)

print(ad_edge/total_edges)
print(aa_edge/total_edges)
print(dd_edge/total_edges)
print(da_edge/total_edges)

# %%
import networkx as nx 

#node_names = dict(zip(np.arange(0, len(ad.index), 1), ad.index))
#Gad = nx.relabel_nodes(Gad, node_names)

Gad = nx.DiGraph(np.matrix(ad), parallel_edges = False)
Gaa = nx.DiGraph(np.matrix(aa), parallel_edges = False)
Gdd = nx.DiGraph(np.matrix(dd), parallel_edges = False)
Gda = nx.DiGraph(np.matrix(da), parallel_edges = False)

# %%
# calculating diameter of graph, i.e. longest shortest path between two vertices
# mean shortest path
# median shortest path

from tqdm import tqdm

def path_length_stats(paths):
    path_lengths = []
    for i in tqdm(paths.keys()):
        for j in paths[i].keys():
            path_lengths.append(len(paths[i][j]))

    path_lengths = np.array(path_lengths)
    diameter = np.max(path_lengths)
    mean = np.mean(path_lengths)
    median = np.median(path_lengths)

    return(diameter, mean, median)

Gad_paths = nx.shortest_path(Gad)
Gad_diameter, Gad_mean, Gad_median = path_length_stats(Gad_paths)

Gaa_paths = nx.shortest_path(Gaa)
Gaa_diameter, Gaa_mean, Gaa_median = path_length_stats(Gaa_paths)

Gdd_paths = nx.shortest_path(Gdd)
Gdd_diameter, Gdd_mean, Gdd_median = path_length_stats(Gdd_paths)

Gda_paths = nx.shortest_path(Gda)
Gda_diameter, Gda_mean, Gda_median = path_length_stats(Gda_paths)


print('Diameters\nGad: %f\nGaa: %f\nGdd: %f\nGda: %f' %(Gad_diameter, Gaa_diameter, Gdd_diameter, Gda_diameter))
print('Mean\nGad: %f\nGaa: %f\nGdd: %f\nGda: %f' %(Gad_mean, Gaa_mean, Gdd_mean, Gda_mean))
print('Median\nGad: %f\nGaa: %f\nGdd: %f\nGda: %f' %(Gad_median, Gaa_median, Gdd_median, Gda_median))

# %%
# average clustering coefficient of all nodes in graph
# 1 means "small world", 0 means unconnected
print(nx.average_clustering(Gad))
print(nx.average_clustering(Gaa))
print(nx.average_clustering(Gdd))
print(nx.average_clustering(Gda))


# %%
