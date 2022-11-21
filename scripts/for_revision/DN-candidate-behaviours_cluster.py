# %%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from pymaid_creds import url, name, password, token
from data_settings import data_date, pairs_path
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

from contools import Celltype, Celltype_Analyzer, Promat

candidate_behaviours_cts = Celltype_Analyzer.get_skids_from_meta_annotation('mw dVNC candidate behaviors', split=True, return_celltypes=True)
for ct in candidate_behaviours_cts:
    ct.name = ct.name.replace('mw candidate ', '')

cluster_level = 7
clusters_cts = Celltype_Analyzer.get_skids_from_meta_annotation(f'mw brain clusters level {cluster_level}', split=True, return_celltypes=True)

DN = pymaid.get_skids_by_annotation('mw dVNC')
clusters_cta = Celltype_Analyzer(clusters_cts)
clusters_cta.set_known_types([Celltype('DN-VNC', DN)])

DN_membership = clusters_cta.memberships(raw_num=True)
DN_membership.columns = [x.name for x in clusters_cta.Celltypes]

# %%
#

data = []
for behaviour in candidate_behaviours_cts:
    for skid in behaviour.skids:
        for cluster in clusters_cts:
            if(skid in cluster.skids):
                data.append([skid, behaviour.name, cluster.name])
            
df = pd.DataFrame(data, columns=['skid', 'behaviour', 'cluster'])
crosstab = pd.crosstab(df.cluster, df.behaviour)


# %%
# Cramer's V

import scipy.stats as stats
import numpy as np

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorical-categorical association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

cramers_v = cramers_corrected_stat(crosstab)

# normalize by number of DNs in each category for plotting purposes
DN_counts_per_cluster = DN_membership.loc[:, crosstab.index].loc['DN-VNC']

for i, idx in enumerate(crosstab.index):
    crosstab.loc[idx] = crosstab.loc[idx]/DN_counts_per_cluster.loc[idx]

fig, ax = plt.subplots(1,1, figsize=(8,8))
sns.heatmap(crosstab.sort_values(by=['backup', 'forward', 'hunch_head-move', 'speed-modulation', 'turn']), ax=ax, square=True)
plt.savefig('plots/correlation_cluster-behaviour.pdf', format='pdf', bbox_inches='tight')

print(f'Correlation between cluster and behavioural category is {cramers_v:.2f} (Cramers V correlation)')

# %%
# plot fraction of cluster with each behavioural category


def plot_marginal_cell_type_cluster(size, particular_cell_type, particular_color, cluster_level, path, only_consider_DNs=True, all_celltypes=None, ylim=(0,1), yticks=([])):

    # all cell types plot data
    if(all_celltypes==None):
        _, all_celltypes = Celltype_Analyzer.default_celltypes()
        
    clusters = Celltype_Analyzer.get_skids_from_meta_annotation(f'mw brain clusters level {cluster_level}', split=True, return_celltypes=True)
    cluster_analyze = Celltype_Analyzer(clusters)

    cluster_analyze.set_known_types(all_celltypes)
    celltype_colors = [x.get_color() for x in cluster_analyze.get_known_types()]
    all_memberships = cluster_analyze.memberships()
    all_memberships = all_memberships.iloc[[0,1,2,3,4,5,6,7,8,9,10,11,12,17,13,14,15,16], :] # switching order so unknown is not above outputs and RGNs before pre-outputs
    celltype_colors = [celltype_colors[i] for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,17,13,14,15,16]] # switching order so unknown is not above outputs and RGNs before pre-outputs
    
    if(type(particular_cell_type)==Celltype):
        # particular cell type data
        cluster_analyze.set_known_types([particular_cell_type])
        membership = cluster_analyze.memberships()

        # plot
        fig = plt.figure(figsize=size) 
        fig.subplots_adjust(hspace=0.1)
        gs = plt.GridSpec(4, 1)

        ax = fig.add_subplot(gs[0:3, 0])
        ind = np.arange(0, len(cluster_analyze.Celltypes))
        ax.bar(ind, membership.iloc[0, :], color=particular_color)
        ax.set(xlim = (-1, len(ind)), ylim=ylim, xticks=([]), yticks=yticks, title=particular_cell_type.get_name())

    if(type(particular_cell_type)==list): # if there are multiple celltypes in main plot

        if(only_consider_DNs):
            DN_VNC = pymaid.get_skids_by_annotation('mw dVNC')

            celltypes = []
            for celltype in cluster_analyze.Celltypes:
                celltype.skids = list(np.intersect1d(celltype.skids, DN_VNC))
                celltypes.append(celltype)

        cluster_analyze = Celltype_Analyzer(celltypes)
        cluster_analyze.set_known_types(particular_cell_type)
        membership = cluster_analyze.memberships()

        # plot
        fig = plt.figure(figsize=size)
        fig.subplots_adjust(hspace=0.1)
        gs = plt.GridSpec(4, 1)

        ax = fig.add_subplot(gs[0:3, 0])
        ind = np.arange(0, len(membership.iloc[0, :]))
        bottom = membership.iloc[0, :]
        plt.bar(ind, membership.iloc[0, :], color=particular_cell_type[0].color)
        for i in range(1, (len(membership.index)-1)):
            plt.bar(ind, membership.iloc[i, :], bottom = bottom, color=particular_cell_type[i].color)
            bottom = bottom + membership.iloc[i, :]

        ax.set(xlim = (-1, len(ind)), ylim=ylim, xticks=([]), yticks=yticks)

    ax = fig.add_subplot(gs[3, 0])
    ind = np.arange(0, len(cluster_analyze.Celltypes))
    ax.bar(ind, all_memberships.iloc[0, :], color=celltype_colors[0])
    bottom = all_memberships.iloc[0, :]
    for i in range(1, len(all_memberships.index)):
        plt.bar(ind, all_memberships.iloc[i, :], bottom = bottom, color=celltype_colors[i])
        bottom = bottom + all_memberships.iloc[i, :]

    ax.set(xlim = (-1, len(ind)), ylim=(0,1), xticks=([]), yticks=([]))
    ax.axis('off')
    ax.axis('off') 
    plt.savefig(path, format='pdf', bbox_inches='tight')

# add colors to Celltypes for plotting
for i, celltype in enumerate(candidate_behaviours_cts):
    celltype.color = sns.color_palette()[i]

plot_marginal_cell_type_cluster(size=(6,2), particular_cell_type=candidate_behaviours_cts, particular_color='', cluster_level=7, path='plots/DN-behaviours_in_clusters.pdf')

# %%
