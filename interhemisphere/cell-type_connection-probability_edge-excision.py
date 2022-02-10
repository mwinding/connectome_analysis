#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import connectome_tools.celltype as ct
import connectome_tools.process_graph as pg
import connectome_tools.process_matrix as pm
import navis

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

ad_edges = pd.read_csv('data/edges_threshold/ad_all-paired-edges.csv', index_col=0)
ad_edges_split = pd.read_csv('data/edges_threshold/pairwise-threshold_ad_all-edges.csv', index_col=0)

graph = pg.Analyze_Nx_G(ad_edges)
graph_split = pg.Analyze_Nx_G(ad_edges_split, split_pairs=True)

pairs = pm.Promat.get_pairs()

# %%
# connection probability between ipsi/bilateral/contra

celltypes_names = ['sensories', 'PNs', 'LNs', 'LHNs', 'FFNs', 'MBINs', 'KCs', 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'RGNs', 'dSEZs', 'dVNCs']
celltypes_annots = ['mw brain ' + x for x in celltypes_names]
celltypes_skids = [ct.Celltype_Analyzer.get_skids_from_meta_annotation(annot) for annot in celltypes_annots]
celltypes = [ct.Celltype(celltypes_names[i], skids) for i, skids in enumerate(celltypes_skids)]

celltypes_pairs = [pm.Promat.get_pairs_from_list(x.skids, pairList=pairs, return_type='pairs') for x in celltypes]
all_celltypes = [list(x.leftid) for x in celltypes_pairs] + [list(x.rightid) for x in celltypes_pairs]
data_adj = ad_edges_split.set_index(['upstream_skid', 'downstream_skid'])


mat = np.zeros(shape=(len(all_celltypes), len(all_celltypes)))
for i, pair_type1 in enumerate(all_celltypes):
    for j, pair_type2 in enumerate(all_celltypes):
        connection = []
        for skid1 in pair_type1:
            for skid2 in pair_type2:
                if((skid1, skid2) in graph_split.G.edges): connection.append(1)
                if((skid1, skid2) not in graph_split.G.edges): connection.append(0)

            mat[i, j] = sum(connection)/len(connection)

df = pd.DataFrame(mat, columns = [x.name + '-left' for x in celltypes] + [x.name + '-right' for x in celltypes],
                        index = [x.name + '-left' for x in celltypes] + [x.name + '-right' for x in celltypes])

(df.iloc[0:16, 0:16] + df.iloc[16:32, 16:32])/2

vmax = 0.05

# plot interhemispheric cell type interactions
fig, ax = plt.subplots(1,1, figsize=(4,4))
sns.heatmap(df, square=True, cmap='Greys', vmax=vmax)
plt.savefig(f'interhemisphere/plots/connection-probability_all-celltypes-types_interhemisphere.pdf', format='pdf', bbox_inches='tight')

ipsi_connections = (df.iloc[0:len(celltypes), 0:len(celltypes)].values + df.iloc[len(celltypes):len(celltypes)*2, len(celltypes):len(celltypes)*2].values)/2
contra_connections = (df.iloc[0:len(celltypes), len(celltypes):len(celltypes)*2].values + df.iloc[len(celltypes):len(celltypes)*2, 0:len(celltypes)].values)/2
ipsi_df = pd.DataFrame(ipsi_connections, index = [x.name for x in celltypes], columns = [x.name for x in celltypes])
contra_df = pd.DataFrame(contra_connections, index = [x.name for x in celltypes], columns = [x.name for x in celltypes])
diff_df = pd.DataFrame(contra_connections-ipsi_connections, index = [x.name for x in celltypes], columns = [x.name for x in celltypes])

# plot increases of cell type interactions based on contralateral edges
cmap = plt.cm.get_cmap('Blues') # modify 'Blues' cmap to have a white background
blue_cmap = cmap(np.linspace(0, 1, 20))
blue_cmap[0] = np.array([1, 1, 1, 1])
blue_cmap = mpl.colors.LinearSegmentedColormap.from_list(name='New_Blues', colors=blue_cmap)

cmap = blue_cmap
fig, ax = plt.subplots(1,1, figsize=(4,4))
sns.heatmap(diff_df, square=True, cmap=cmap, vmax=vmax, vmin=0)
plt.savefig(f'interhemisphere/plots/connection-probability-diff_all-celltypes-types_interhemisphere.pdf', format='pdf', bbox_inches='tight')

# plot decreases in cell type interactions based on contralateral edges
cmap = plt.cm.get_cmap('Reds') # modify 'Blues' cmap to have a white background
red_cmap = cmap(np.linspace(0, 1, 20))
red_cmap[0] = np.array([1, 1, 1, 1])
red_cmap = mpl.colors.LinearSegmentedColormap.from_list(name='New_Reds', colors=red_cmap)

cmap = red_cmap
fig, ax = plt.subplots(1,1, figsize=(4,4))
sns.heatmap(-(diff_df), square=True, cmap=cmap, vmax=vmax, vmin=0)
plt.savefig(f'interhemisphere/plots/connection-probability-neg-diff_all-celltypes-types_interhemisphere.pdf', format='pdf', bbox_inches='tight')

# %%
# plot examples of largest change for figure

indices = np.where(diff_df>0)

data = []
for i in range(0, len(indices[0])):
    ipsi = ipsi_df.iloc[indices[0][i], indices[1][i]]
    contra = contra_df.iloc[indices[0][i], indices[1][i]]
    diff = diff_df.iloc[indices[0][i], indices[1][i]]
    connection_type = ipsi_df.index[indices[0][i]].replace('s', '') + ' > ' + ipsi_df.index[indices[1][i]].replace('s', '')
    data.append([connection_type, ipsi, contra, diff])

data = pd.DataFrame(data, columns = ['connection', 'ipsi', 'contra', 'delta'])

df = data[(data.delta>0.002) | (data.ipsi==0)]
df.reset_index(inplace=True, drop=True)
df['x'] = df.index

'''
fig, ax = plt.subplots(1,1, figsize=(4,4))
sns.scatterplot(data = df, x='x', y='ipsi', ax=ax, s=0.5)
sns.scatterplot(data=df, x='x', y='contra', ax=ax, s=0.5)
plt.savefig(f'interhemisphere/plots/connection-probability-diff_example-celltypes-types_interhemisphere.pdf', format='pdf', bbox_inches='tight')
'''
cmap = blue_cmap

fig, ax = plt.subplots(1,1, figsize=(4,4))
sns.heatmap(df.loc[:, ['contra', 'ipsi']].T, annot=True, square=True, cmap=cmap, vmin=0)
plt.savefig(f'interhemisphere/plots/connection-probability-diff_example-celltypes-types_interhemisphere_heatmap.pdf', format='pdf', bbox_inches='tight')


fig, ax = plt.subplots(1,1, figsize=(4,4))
sns.heatmap(df.loc[[0,1,4,6], ['contra', 'ipsi']].T, annot=True, square=True, cmap=cmap, vmin=0)
plt.savefig(f'interhemisphere/plots/connection-probability-diff_example-celltypes-types_interhemisphere_heatmap-modulated.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1, figsize=(4,4))
sns.heatmap(df.loc[[2,3,5,7,8,9], ['contra', 'ipsi']].T, annot=True, square=True, cmap=cmap, vmin=0)
plt.savefig(f'interhemisphere/plots/connection-probability-diff_example-celltypes-types_interhemisphere_heatmap-new.pdf', format='pdf', bbox_inches='tight')

# %%
