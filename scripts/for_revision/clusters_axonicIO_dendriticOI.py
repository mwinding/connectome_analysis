# %%
# identify clusters with higher than 2*std mean axonic IO or dendritic OI

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
from data_settings import data_date, pairs_path
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

from contools import Celltype, Celltype_Analyzer, Promat

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'arial'

cluster_lvl = 7 
clusters = Celltype_Analyzer.get_skids_from_meta_annotation(f'mw brain clusters level {cluster_lvl}', split=True, return_celltypes=True)

axon_inputs = pd.read_csv('data/adj/inputs_' + data_date + '.csv', index_col=0)
axon_outputs = pd.read_csv('data/adj/outputs_' + data_date + '.csv', index_col=0)
input_output = pd.concat([axon_inputs, axon_outputs], axis=1)

# axonic IO
axonicIO = input_output[input_output.axon_output>0]
axonicIO = axonicIO.axon_input/axonicIO.axon_output

# dendritic OI
dendriticOI = input_output[input_output.dendrite_input>0]
dendriticOI = dendriticOI.dendrite_output/dendriticOI.dendrite_input

# %%
# identifying per cluster

# collecting data into df

# axonic input/output ratio
cluster_axon_io = []
for i in axonicIO.index:
    for cluster in clusters:
        if(i in cluster.skids):
            cluster_axon_io.append([i, cluster.name, axonicIO.loc[i]])

cluster_axon_io_df = pd.DataFrame(cluster_axon_io, columns=['skid', 'cluster', 'axonic_IO'])

# dendritic output/input ratio
cluster_den_oi = []
for i in dendriticOI.index:
    for cluster in clusters:
        if(i in cluster.skids):
            cluster_den_oi.append([i, cluster.name, dendriticOI.loc[i]])

cluster_den_oi_df = pd.DataFrame(cluster_den_oi, columns=['skid', 'cluster', 'dendritic_OI'])

# %%
# plot per cluster data
fig, ax = plt.subplots(1,1,figsize=(10,4))
sns.barplot(data=cluster_axon_io_df, x='cluster', y='axonic_IO', order = [x.name for x in clusters], errwidth=0.5, color='gray', ax=ax)
ax.set(ylim=[0,1.5], xticklabels=[], yticks=[0.0, 0.5, 1.0, 1.5], ylabel=f'Axonic Input/Output Ratio (average per cluster)')
axonIO_line = np.mean(cluster_axon_io_df.groupby('cluster').mean().axonic_IO) + 2*np.std(cluster_axon_io_df.groupby('cluster').mean().axonic_IO)
plt.axhline(axonIO_line, color='gray', alpha=0.5, linewidth=0.5)
plt.savefig(f'plots/clusters_axonic-IO.pdf', format='pdf')

fig, ax = plt.subplots(1,1,figsize=(10,4))
sns.barplot(data=cluster_den_oi_df, x='cluster', y='dendritic_OI', order = [x.name for x in clusters], errwidth=0.5, color='gray', ax=ax)
ax.set(ylim=[0,2], xticklabels=[], ylabel=f'Dendritic Output/Input Ratio (average per cluster)')
denOI_line = np.mean(cluster_den_oi_df.groupby('cluster').mean().dendritic_OI) + 2*np.std(cluster_den_oi_df.groupby('cluster').mean().dendritic_OI)
plt.axhline(denOI_line, color='gray', alpha=0.5, linewidth=0.5)
plt.savefig(f'plots/clusters_dendritic-OI.pdf', format='pdf')
# %%
# identify clusters over 2*std

axonIO_clusters_2std = cluster_axon_io_df.groupby('cluster').mean()
axonIO_clusters_2std = axonIO_clusters_2std[axonIO_clusters_2std.axonic_IO>axonIO_line]

denOI_clusters_2std = cluster_den_oi_df.groupby('cluster').mean()
denOI_clusters_2std = denOI_clusters_2std[denOI_clusters_2std.dendritic_OI>denOI_line]

# %%
# donut plots of identified prominent clusters

# pull skids for each cluster
axonIO_clusters_2std_skids = [pymaid.get_skids_by_annotation(cluster) for cluster in axonIO_clusters_2std.index]
axonIO_clusters_2std_skids = [x for sublist in axonIO_clusters_2std_skids for x in sublist]

denOI_clusters_2std_skids = [pymaid.get_skids_by_annotation(cluster) for cluster in denOI_clusters_2std.index]
denOI_clusters_2std_skids = [x for sublist in denOI_clusters_2std_skids for x in sublist]

denOI_ct = Celltype('denOI_clusters', denOI_clusters_2std_skids)
axonIO_ct = Celltype('axonIO_clusters', axonIO_clusters_2std_skids)

# make Celltype_Analyzer to plot celltypes within
cta = Celltype_Analyzer([denOI_ct, axonIO_ct])

_, celltypes = Celltype_Analyzer.default_celltypes()
cta.set_known_types(celltypes)
num_df = cta.memberships(raw_num=True)
num_df.loc['sensories', 'denOI_clusters (158)']=0 # sensories have no dendrites

# pull official celltype colors
official_order = ['sensories', 'ascendings', 'PNs', 'PNs-somato', 'LNs', 'LHNs', 'FFNs', 'MBINs', 'KCs', 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'RGNs', 'dSEZs', 'dVNCs', 'unknown']
colors = list(pymaid.get_annotated('mw brain simple colors').name)
colors_names = [x.name.values[0] for x in list(map(pymaid.get_annotated, colors))] # use order of colors annotation for now
color_sort = [np.where(x.replace('mw brain ', '')==np.array(official_order))[0][0] for x in colors_names]
colors = [element for _, element in sorted(zip(color_sort, colors))]
colors = colors + ['tab:gray']

# donut plot of aa cell types
fig, ax = plt.subplots(1,1,figsize=(2,2))
ax.pie(num_df.loc[official_order, 'axonIO_clusters (74)'], colors=colors)

my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

plt.savefig('plots/clusters_axonIO_celltypes.pdf', format='pdf', bbox_inches='tight')

# donut plot of dd cell types
fig, ax = plt.subplots(1,1,figsize=(2,2))
ax.pie(num_df.loc[official_order, 'denOI_clusters (158)'], colors=colors)

my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

plt.savefig('plots/clusters_denOI_celltypes.pdf', format='pdf', bbox_inches='tight')
