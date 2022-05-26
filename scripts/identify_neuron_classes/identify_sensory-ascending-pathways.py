# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
from data_settings import data_date, pairs_path
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

from contools import Celltype, Celltype_Analyzer, Promat, Analyze_Nx_G
import navis
import networkx as nx

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

pairs = Promat.get_pairs(pairs_path=pairs_path)

# load previously generated edge list with % threshold
ad_edges = Promat.pull_edges(type_edges='ad', threshold=0.01, data_date=data_date, pairs_combined=True)
Gad = Analyze_Nx_G(ad_edges)

# %%
# load 2nd-order, 3rd-order, 4th-order and generate pathways

order2 = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain inputs 2nd_order', split=True, return_celltypes=True)
order3 = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain inputs 3rd_order', split=True, return_celltypes=True)
order4 = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain inputs 4th_order', split=True, return_celltypes=True)



# %%
# all pathways from PNs to dVNCs

PN_annots = pymaid.get_annotated('mw brain inputs 2nd_order PN').name
PNs_pairids = [Promat.load_pairs_from_annotation(annot, pairList=pairs, return_type='all_pair_ids') for annot in PN_annots]
dVNC_pairids = Promat.load_pairs_from_annotation('mw dVNC', pairList=pairs, return_type='all_pair_ids')

# identify shortest paths
paths_modalities = []
for PNs in PNs_pairids:
    paths = []
    for dVNC in dVNC_pairids:
        for PN in PNs:
            try: 
                new_paths = [nx.shortest_path(G=Gad.G, source=PN, target=dVNC)]
                paths = paths + new_paths
            except: print(f'No path between {PN} and {dVNC}')

    #paths = [x for x in paths if len(x)<=8]

    paths_modalities.append(paths)

# load paths into Celltype_Analyzer
paths_cta = Celltype_Analyzer([Celltype(PN_annots[i].replace('mw ', '') + ' paths', [x[1:len(x)-1] for x in paths_modalities[i] if x[1:len(x)-1]!=[] ]) for i in range(len(paths_modalities))])
paths_similarity = paths_cta.compare_membership(sim_type='dice')

# plot similarity between paths from different modalities
fig, ax = plt.subplots(1,1,figsize=(2,2))
sns.heatmap(paths_similarity, ax=ax, square=True)
plt.savefig('plots/pathway_similarity.pdf', format='pdf', bbox_inches='tight')

# %%
# clustering of intra-modality pathway types
# takes ~40min
from joblib import Parallel, delayed
from tqdm import tqdm

names = [x.name.replace('2nd_order PN ', '') for x in paths_cta.Celltypes]
names = [x.replace('/', '_') for x in names] # replace '/' in mechano-II/III to prevent issues in path

def cluster_paths(paths_cta, modalitiy_num, modality_names):    
    modality_paths = Celltype_Analyzer([Celltype(f'path-{i}', x) for i, x in enumerate(paths_cta.Celltypes[modalitiy_num].get_skids())])
    modality_paths_sim = modality_paths.compare_membership(sim_type='dice')

    #g = sns.clustermap(modality_paths_sim)
    #g.fig.suptitle(modality_names[modalitiy_num])
    #plt.savefig(f'plots/pathway-similarity_{modality_names[modalitiy_num]}_clustering.pdf', format='pdf', bbox_inches='tight')

    return(modality_paths_sim)

modality_paths_sims = Parallel(n_jobs=-1)(delayed(cluster_paths)(paths_cta=paths_cta, modalitiy_num=i, modality_names=names) for i in tqdm(range(0, len(paths_cta.Celltypes))))

import pickle
pickle.dump(modality_paths_sims, open('plots/modality_paths_sims.p', 'wb'))

# %%
# finding clusters of paths; plotting heatmaps
import scipy

color_threshold=[3.0, 3.3, 3.0, 2.5, 1.8, 2.0, 2.3, 3.5, 3.3, 2.6, 2.6, 1.5]
'''
# test color_threshold
threshold = 3.0
num = 2
df = modality_paths_sims[num]
df.index = [str(x) for x in paths_cta.Celltypes[num].get_skids()]
df.columns = [str(x) for x in paths_cta.Celltypes[num].get_skids()]
g = sns.clustermap(df)
den = scipy.cluster.hierarchy.dendrogram(g.dendrogram_col.linkage,
                                        labels = df.index,
                                        color_threshold=threshold)

g = sns.clustermap(df, tree_kws={'colors':den['color_list']}, cbar=False)
g.cax.set_visible(False)
g.fig.suptitle(names[num])
'''

figsize = (3,3)
dendrogram_ratio = 0.15

from scipy.cluster.hierarchy import fcluster

# save png of each similarity matrix and dendrogram
# extract the members of clusters
modality_clusters = []
for i in (range(len(modality_paths_sims))):
    df = modality_paths_sims[i]
    df.index = [str(x) for x in paths_cta.Celltypes[i].get_skids()]
    df.columns = [str(x) for x in paths_cta.Celltypes[i].get_skids()]
    g = sns.clustermap(df)
    den = scipy.cluster.hierarchy.dendrogram(g.dendrogram_col.linkage,
                                            labels = df.index,
                                            color_threshold=color_threshold[i])

    # extract members (add here)
    clusters = fcluster(Z=g.dendrogram_col.linkage, t=color_threshold[i], criterion='distance')
    modality_clusters.append(clusters)

    g = sns.clustermap(df, tree_kws={'colors':den['color_list']}, yticklabels=False, xticklabels=False, figsize=figsize, dendrogram_ratio=dendrogram_ratio)
    g.cax.set_visible(False)
    plt.savefig(f'plots/pathway-similarity_{names[i]}_clusters_linkage-threshold{color_threshold[i]}.png', format='png', dpi=300, transparent=True)

# piece the individual PNGs together to make grid of clustermaps 
import matplotlib.image as mpimg
save_paths = [f'plots/pathway-similarity_{names[i]}_clusters_linkage-threshold{color_threshold[i]}.png' for i in range(len(modality_paths_sims))]

n_cols = 4
n_rows = 3
fig = plt.figure(figsize=(n_cols*2, n_rows*2))
gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=0, hspace=0.01)
axs = np.empty((n_rows, n_cols), dtype=object)
names_plot = ['Olfactory', 'Gustatory-external', 'Gustatory-pharyngeal', 'Gut', 'Thermo-warm', 'Thermo-cold', 'Visual', 'Nociceptive', 'Mechano-Ch', 'Mechano-II/III', 'Proprioceptive', 'Respiratory']

for i in range(len(modality_paths_sims)):
    inds = np.unravel_index(i, shape=(n_rows, n_cols))
    ax = fig.add_subplot(gs[inds])
    axs[inds] = ax

    ax.imshow(mpimg.imread(save_paths[i]))
    ax.text((ax.get_xlim()[0] + ax.get_xlim()[1])/2 - ax.get_xlim()[1]*0.05, y=20,
                s=names_plot[i], transform=ax.transData, color='k', alpha=1)

    # turn off x and y axis

    plt.tight_layout()
    ax.set_axis_off()

    plt.savefig(f'plots/pathway-similarity_clusters_all.png', format='png', dpi=600, transparent=True)

# %%
# extract clusters from each modality

paths_clusters_modalities = []
for i in range(len(modality_clusters)):
    paths_df = pd.DataFrame(zip(paths_cta.Celltypes[i].get_skids(), modality_clusters[i]), columns=['path', 'cluster'])
    paths_df = paths_df.set_index('cluster')

    cluster_names = list(np.unique(paths_df.index))

    paths_clusters_modalities.append([list(paths_df.loc[cluster_name].path) for cluster_name in cluster_names])

# connectivity between path clusters within modality
adj_ad = Promat.pull_adj(type_adj='ad', data_date=data_date)
connectivity_modalities = []
paths_clusters_ctas = []
for paths_mod in paths_clusters_modalities:

    paths_cluster_mod = []
    for paths_cluster in paths_mod:
        if(type(paths_cluster[0])!=np.int64):
            paths_cluster = list(np.unique([x for sublist in paths_cluster for x in sublist]))
        paths_cluster_mod.append(paths_cluster)

    paths_cluster_mod_cta = Celltype_Analyzer([Celltype(f'cluster-{i}', Promat.get_paired_skids(skids, pairs, unlist=True)) for i, skids in enumerate(paths_cluster_mod)])
    connectivity = paths_cluster_mod_cta.connectivity(adj=adj_ad, normalize_pre_num=True)
    connectivity_modalities.append(connectivity)
    paths_clusters_ctas.append(paths_cluster_mod_cta)

# connectivity between all pathway clusters

# set up coloring for modalities
clusters_len = [len(x.Celltypes) for x in paths_clusters_ctas]
colors  = ['#00A651', '#8DC63F', '#D7DF23', '#DBA728', '#35B3E7', '#ed1c24',
            '#5c62ad', '#f15a29', '#007c70', '#49b292', '#e04f9c', '#662d91']
colors_clusters = [x*[colors[i]] for i, x in enumerate(clusters_len)]
colors_clusters = [x for sublist in colors_clusters for x in sublist]

all_path_clusters_cta = Celltype_Analyzer([x for sublist in [x.Celltypes for x in paths_clusters_ctas] for x in sublist])
all_path_clusters_connectivity = all_path_clusters_cta.connectivity(adj=adj_ad, normalize_pre_num=True)
g = sns.clustermap(all_path_clusters_connectivity, figsize=(5,5), row_cluster=False, col_cluster=False, row_colors=colors_clusters, col_colors=colors_clusters, vmax=10, xticklabels=False, yticklabels=False, dendrogram_ratio=0.01)
ax = g.ax_heatmap
g.cax.set_visible(False)
from matplotlib.patches import Rectangle

for i in range(len(paths_clusters_ctas)):
    # Create a Rectangle patch
    if(i==0):
        rect = Rectangle((0, 0), clusters_len[i], clusters_len[i], linewidth=2, edgecolor=colors[i], facecolor='none')

    if(i>0):
        rect = Rectangle((sum(clusters_len[0:i]), sum(clusters_len[0:i])), clusters_len[i], clusters_len[i], linewidth=2, edgecolor=colors[i], facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

plt.savefig(f'plots/pathway-clusters_all-connectivity_vmax10.png', format='png', dpi=600, transparent=True)
# %%
# incomplete

# unique pathways across all modalities
fraction_unique=[]
for i in range(len(paths_cta.Celltypes)):
    all_other_paths = [x for j, x in enumerate(paths_cta.Celltypes) if j!=i]
    all_other_paths = list(np.unique([x for sublist in all_other_paths for x in sublist.get_skids()]))

    paths_this_celltype = paths_cta.Celltypes[i].get_skids()
    fraction_unique.append(len(np.setdiff1d(paths_this_celltype, all_other_paths))/len(paths_this_celltype))

# truly unique pathways (don't share any nodes within modalities)



