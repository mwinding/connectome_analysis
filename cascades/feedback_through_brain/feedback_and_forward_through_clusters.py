#%%
import sys
import os

os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory
sys.path.append('/Users/mwinding/repos/maggot_models')

from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.data import load_metagraph
from src.visualization import CLASS_COLOR_DICT, adjplot
from src.traverse import Cascade, to_transmission_matrix
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot

import connectome_tools.process_matrix as pm
import connectome_tools.cascade_analysis as casc
import connectome_tools.celltype as ct
import connectome_tools.cluster_analysis as clust

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 5})

# cluster object
cluster_level = 6
clusters = clust.Analyze_Cluster(cluster_lvl=cluster_level)

# load adj matrices
adj_ad = pm.Promat.pull_adj(type_adj='ad', subgraph='brain and accessory')
adj_aa = pm.Promat.pull_adj(type_adj='aa', subgraph='brain and accessory')

# load input and output neurons
all_sensories = ct.Celltype_Analyzer.get_skids_from_meta_meta_annotation('mw brain sensory modalities')
all_outputs = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain outputs')

# %%
## cascades starting at each cluster

p = 0.05
max_hops = 4
n_init = 100

cluster_cascades = clusters.ff_fb_cascades(adj=adj_ad, p=p, max_hops=max_hops, n_init=n_init)
ff_fb_df = clusters.all_ff_fb_df(cluster_cascades, divide_by_skids=True)

sns.heatmap(ff_fb_df.T, square=True, robust=True, cmap='Reds')
plt.savefig('cascades/test.pdf')

# %%
# OLD CODE
# categorize neurons in each cluster cascade by cluster

def hit_hist_to_clusters(hit_hist_list, lvl7, order):
    # breaking signal cascades into cluster groups
    output_hit_hist_lvl7 = []
    for hit_hist in hit_hist_list:
        clustered_hist = []

        for key in order:
            skids = lvl7.groups[key]
            indices = np.where([x in skids for x in mg_ad.meta.index])[0]
            cluster_hist = hit_hist[indices]
            cluster_hist = pd.DataFrame(cluster_hist, index = indices)

            clustered_hist.append(cluster_hist)
        
        output_hit_hist_lvl7.append(clustered_hist)
    
    return(output_hit_hist_lvl7)

def sum_cluster_hit_hist(hit_hist_cluster, order, normalized = True):
    # summed signal cascades per cluster group (hops remain intact)
    summed_hist = []
    for hit_hist in hit_hist_cluster:
        sum_hist = []
        for i, cluster in enumerate(hit_hist):
            if(normalized==True):
                sum_cluster = cluster.sum(axis = 0)/(len(cluster.index)) # normalize by number of neurons in cluster
            if(normalized==False):
                sum_cluster = cluster.sum(axis = 0)
            sum_hist.append(sum_cluster)

        sum_hist = pd.DataFrame(sum_hist) # column names will be hop number
        sum_hist.index = order # uses cluster name for index of each summed cluster row
        summed_hist.append(sum_hist)

    return(summed_hist)

def alt_sum_cluster(summed_hops_hist_lvl7):
    
    alt_summed_hops_hist_lvl7 = []
    for hop in summed_hops_hist_lvl7[0].columns:
        summed_hist_lvl7 = []
        for hit_hist in summed_hops_hist_lvl7:
            summed_hist_lvl7.append(hit_hist.iloc[:, hop])

        summed_hist_lvl7 = pd.DataFrame(summed_hist_lvl7, index = summed_hops_hist_lvl7[0].index).T
        alt_summed_hops_hist_lvl7.append(summed_hist_lvl7)
    
    return(alt_summed_hops_hist_lvl7)

cluster_hit_hist_lvl7_ad = hit_hist_to_clusters(cluster_hit_hist_list_ad, lvl7, order)
summed_hops_hist_lvl7_ad = sum_cluster_hit_hist(cluster_hit_hist_lvl7_ad, order)
alt_summed_hops_hist_lvl7_ad = alt_sum_cluster(summed_hops_hist_lvl7_ad)

cluster_hit_hist_lvl7_aa = hit_hist_to_clusters(cluster_hit_hist_list_aa, lvl7, order)
summed_hops_hist_lvl7_aa = sum_cluster_hit_hist(cluster_hit_hist_lvl7_aa, order)
alt_summed_hops_hist_lvl7_aa = alt_sum_cluster(summed_hops_hist_lvl7_aa)
'''
cluster_hit_hist_lvl7_dd = hit_hist_to_clusters(cluster_hit_hist_list_dd, lvl7, order)
summed_hops_hist_lvl7_dd = sum_cluster_hit_hist(cluster_hit_hist_lvl7_dd, order)
alt_summed_hops_hist_lvl7_dd = alt_sum_cluster(summed_hops_hist_lvl7_dd)

cluster_hit_hist_lvl7_da = hit_hist_to_clusters(cluster_hit_hist_list_da, lvl7, order)
summed_hops_hist_lvl7_da = sum_cluster_hit_hist(cluster_hit_hist_lvl7_da, order)
alt_summed_hops_hist_lvl7_da = alt_sum_cluster(summed_hops_hist_lvl7_da)
'''
# %%
# plot visits to different groups, normalized to group size

def plot_graphs(alt_summed_hops_hist_lvl7, connection_type):
    # activity in each cluster, based on hop number
    for i in range(10):
        fig, axs = plt.subplots(
        1, 1, figsize = (8, 7)
        )
        ax = axs

        sns.heatmap(alt_summed_hops_hist_lvl7[0] + alt_summed_hops_hist_lvl7[i], ax = ax, rasterized = True, square=True)
        ax.set_title('Hop %i' %i, fontsize = 10)
        ax.set_ylabel('Individual Clusters', fontsize = 10)
        ax.set_xlabel('Individual Clusters', fontsize = 10)
        ax.set_yticks([]);
        ax.set_xticks([]);

        fig.savefig('cascades/feedback_through_brain/plots/cluster_cascades_png/%s_feedback_vs_feedforward_clusters_hop%i.png' %(connection_type, i), bbox_inches='tight')

    # summed data with varying hop end point
    for i in range(10):
        fig, axs = plt.subplots(
        1, 1, figsize = (8, 7)
        )
        ax = axs

        sns.heatmap(sum(alt_summed_hops_hist_lvl7[0:(i+1)]), ax = ax, rasterized = True, square=True)
        ax.set_title('Summed to Hop %i' %i, fontsize = 10)
        ax.set_ylabel('Individual Clusters', fontsize = 10)
        ax.set_xlabel('Individual Clusters', fontsize = 10)
        ax.set_yticks([]);
        ax.set_xticks([]);

        fig.savefig('cascades/feedback_through_brain/plots/%s_feedback_vs_feedforward_clusters_%ihops_summed.pdf' %(connection_type, i+1), bbox_inches='tight')

plot_graphs(alt_summed_hops_hist_lvl7_ad, 'ad')
plot_graphs(alt_summed_hops_hist_lvl7_aa, 'aa')

# %%
# plot visits to different groups, normalized to group size, displaying all hops

# plot only first 3 hops
panel_width = 10
panel_height = 9
fig, axs = plt.subplots(
    panel_width, panel_height, figsize = (30, 30), sharey = True
)

for x in range(len(summed_hops_hist_lvl7_ad)):
    for j in range(panel_height):
        for i in range(panel_width):
            ax = axs[i, j]
            sns.heatmap(summed_hops_hist_lvl7_ad[x], ax = ax, rasterized = True, cbar = False)
            ax.set_xlabel('Hops from source')
            ax.set_ylabel('Individual Clusters')
            ax.set_yticks([]);

fig.savefig('cascades/feedback_through_brain/plots/feedback_vs_feedforward_clusters_allhops.pdf', bbox_inches='tight')


# %%
# some examples
width = 0.75
height = 1.5

fig, axs = plt.subplots(
    1, 1, figsize = (width, height)
)
ax = axs
sns.heatmap(summed_hops_hist_lvl7_ad[0].iloc[:, 0:4], ax = axs)
ax.set_yticks([])
ax.set_ylabel('Individual Clusters')
fig.savefig('cascades/feedback_through_brain/plots/feedback_vs_feedforward_cluster0.pdf', bbox_inches='tight')

fig, axs = plt.subplots(
    1, 1, figsize = (width, height)
)
ax = axs
sns.heatmap(summed_hops_hist_lvl7_ad[40].iloc[:, 0:4], ax = axs)
ax.set_yticks([])
ax.set_ylabel('Individual Clusters')
fig.savefig('cascades/feedback_through_brain/plots/feedback_vs_feedforward_cluster40.pdf', bbox_inches='tight')

fig, axs = plt.subplots(
    1, 1, figsize = (width, height)
)
ax = axs
sns.heatmap(pd.concat([summed_hops_hist_lvl7_ad[50].iloc[:, 0:4], summed_hops_hist_lvl7_ad[50].iloc[:, 0:4].sum(axis = 1)], axis = 1), ax = axs)
ax.set_yticks([])
ax.set_ylabel('Individual Clusters')
fig.savefig('cascades/feedback_through_brain/plots/feedback_vs_feedforward_cluster50.pdf', bbox_inches='tight')

fig, axs = plt.subplots(
    1, 1, figsize = (width, height)
)
ax = axs
sns.heatmap(summed_hops_hist_lvl7_ad[86].iloc[:, 0:4], ax = axs)
ax.set_yticks([])
ax.set_ylabel('Individual Clusters')
fig.savefig('cascades/feedback_through_brain/plots/feedback_vs_feedforward_cluster86.pdf', bbox_inches='tight')

# %%
# feedback character of clusters

feedback_mat_ad = sum(alt_summed_hops_hist_lvl7_ad[0:4])

ff_fb_character_ad_output = []
for i in range(len(feedback_mat_ad.columns)):
    cols = feedback_mat_ad.columns
    column = feedback_mat_ad.loc[:, cols[i]]
    
    fb = sum(column[0:i])
    ff = sum(column[(i+1):len(column)])

    if((ff>0) | (fb>0)):
        ff_fb_character_ad_output.append([column.name, ff, fb, ff/(ff+fb), fb/(ff+fb)])
    if((ff==0) & (fb==0)):
        ff_fb_character_ad_output.append([column.name, 0, 0, 0, 0])

ff_fb_character_ad_output = pd.DataFrame(ff_fb_character_ad_output, columns = ['cluster', 'feedforward', 'feedback', 'p_ff', 'p_fb'])

ff_fb_character_ad_input = []
for i in range(len(feedback_mat_ad.columns)):
    cols = feedback_mat_ad.columns
    column = feedback_mat_ad.loc[cols[i]]
    
    ff = sum(column[0:i])
    fb = sum(column[(i+1):len(column)])

    if((ff>0) | (fb>0)):
        ff_fb_character_ad_input.append([column.name, ff, fb, ff/(ff+fb), fb/(ff+fb)])
    if((ff==0) & (fb==0)):
        ff_fb_character_ad_input.append([column.name, 0, 0, 0, 0])

ff_fb_character_ad_input = pd.DataFrame(ff_fb_character_ad_input, columns = ['neuron', 'feedforward', 'feedback', 'p_ff', 'p_fb'])
'''
fig, axs = plt.subplots(
    1, 1, figsize = (1.5, 2)
)
ax = axs
sns.lineplot(x = range(len(ff_fb_character_ad.p_ff)), y = ff_fb_character_ad.p_ff, ax = ax)
sns.lineplot(x = range(len(ff_fb_character_ad.p_ff)), y = ff_fb_character_ad.p_fb, ax = ax)
ax.set_ylabel('Fraction of Signal')
ax.set_xticks([])
ax.set_xlabel('Individual Clusters')

fig.savefig('cascades/feedback_through_brain/plots/ff_fb_character_clusters_ad.pdf', bbox_inches='tight')
'''
fig, axs = plt.subplots(
    2, 1, figsize = (1.5, 2)
)
ax = axs[0]
sns.lineplot(x = range(len(ff_fb_character_ad_output.feedforward)), y = ff_fb_character_ad_output.feedforward, ax = ax)
sns.lineplot(x = range(len(ff_fb_character_ad_output.feedback)), y = ff_fb_character_ad_output.feedback, ax = ax)
ax.set_ylabel('Signal Intensity\n3-Hop Output')
ax.set_xticks([])

ax = axs[1]
sns.lineplot(x = range(len(ff_fb_character_ad_input.feedforward)), y = ff_fb_character_ad_input.feedforward, ax = ax)
sns.lineplot(x = range(len(ff_fb_character_ad_input.feedback)), y = ff_fb_character_ad_input.feedback, ax = ax)
ax.set_ylabel('Signal Intensity\n3-Hop Input')
ax.set_xticks([])
ax.set_xlabel('Individual Clusters')

fig.savefig('cascades/feedback_through_brain/plots/ff_fb_character_clusters_ad_input_output.pdf', bbox_inches='tight')
'''
# aa version

feedback_mat_aa = sum(alt_summed_hops_hist_lvl7_aa[0:4])

ff_fb_character_aa = []
for i in range(len(feedback_mat_aa.columns)):
    cols = feedback_mat_aa.columns
    column = feedback_mat_aa.loc[:, cols[i]]
    if(i==0):
        fb = 0
    if(i>0):
        fb = sum(column[0:(i+1)])
    
    if(i==(len(column)-1)):
        ff = 0
    if(i<(len(column)-1)):
        ff = sum(column[(i+1):len(column)])

    ff_fb_character_aa.append([column.name, ff, fb, ff/(ff+fb), fb/(ff+fb)])

ff_fb_character_aa = pd.DataFrame(ff_fb_character_aa, columns = ['cluster', 'feedforward', 'feedback', 'p_ff', 'p_fb'])

fig, axs = plt.subplots(
    1, 1, figsize = (1.5, 2)
)
ax = axs
sns.lineplot(x = range(len(ff_fb_character_aa.p_ff)), y = ff_fb_character_aa.p_ff, ax = ax)
sns.lineplot(x = range(len(ff_fb_character_aa.p_ff)), y = ff_fb_character_aa.p_fb, ax = ax)
ax.set_ylabel('Fraction of Signal')
ax.set_xticks([])
ax.set_xlabel('Individual Clusters')

fig.savefig('cascades/feedback_through_brain/plots/ff_fb_character_clusters_aa.pdf', bbox_inches='tight')

fig, axs = plt.subplots(
    1, 1, figsize = (1.5, 2)
)
ax = axs
sns.lineplot(x = range(len(ff_fb_character_aa.feedforward)), y = ff_fb_character_aa.feedforward, ax = ax)
sns.lineplot(x = range(len(ff_fb_character_aa.feedback)), y = ff_fb_character_aa.feedback, ax = ax)
ax.set(ylim = (0, max(ff_fb_character_ad.feedforward)*1.025))
ax.set_ylabel('Signal Intensity')
ax.set_xticks([])
ax.set_xlabel('Individual Clusters')

fig.savefig('cascades/feedback_through_brain/plots/ff_fb_character_clusters_aa_raw.pdf', bbox_inches='tight')
'''

# %%
# how much "signal flow" distance does each signal travel? in feedforward and feedback directions?



# %%
