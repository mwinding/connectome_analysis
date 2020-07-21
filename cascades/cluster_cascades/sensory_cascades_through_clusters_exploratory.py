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
mg.calculate_degrees(inplace=True)

adj = mg.adj  # adjacency matrix from the "mg" object

clusters = pd.read_csv('cascades/data/meta-method=color_iso-d=8-bic_ratio=0.95-min_split=32.csv', index_col = 0, header = 0)
order = pd.read_csv('cascades/data/signal_flow_order_lvl7.csv').values

# make array from list of lists
order_delisted = []
for sublist in order:
    order_delisted.append(sublist[0])

order = np.array(order_delisted)

#%%
# pull sensory annotations and then pull associated skids
input_names = pymaid.get_annotated('mw brain inputs').name
input_skids_list = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain inputs').name))
input_skids = [val for sublist in input_skids_list for val in sublist]

output_names = pymaid.get_annotated('mw brain outputs').name
output_skids_list = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain outputs').name))
output_skids = [val for sublist in output_skids_list for val in sublist]

# order names and skids in desired way for the rest of analysis
sensory_order = [0, 3, 4, 1, 2, 6, 5]
input_names_format = ['ORN', 'thermo', 'photo', 'AN', 'MN', 'vtd', 'A00c']
input_names_format_reordered = [input_names_format[i] for i in sensory_order]
input_skids_list_reordered = [input_skids_list[i] for i in sensory_order]

#%%
# cascades from each sensory modality
from src.traverse import Cascade, to_transmission_matrix
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot

# convert skids to indices
input_indices_list = []
for input_skids in input_skids_list_reordered:
    indices = np.where([x in input_skids for x in mg.meta.index])[0]
    input_indices_list.append(indices)

output_indices_list = []
for input_skids in output_skids_list:
    indices = np.where([x in input_skids for x in mg.meta.index])[0]
    output_indices_list.append(indices)

all_input_indices = np.where([x in input_skids for x in mg.meta.index])[0]
output_indices = np.where([x in output_skids for x in mg.meta.index])[0]

p = 0.05
max_hops = 10
n_init = 100
simultaneous = True
transition_probs = to_transmission_matrix(adj, p)

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    stop_nodes = output_indices,
    max_hops=max_hops,
    allow_loops = False,
    n_init=n_init,
    simultaneous=simultaneous,
)

input_hit_hist_list = []
for input_indices in input_indices_list:
    hit_hist = cdispatch.multistart(start_nodes = input_indices)
    input_hit_hist_list.append(hit_hist)

all_input_hit_hist = cdispatch.multistart(start_nodes = all_input_indices)

#%%
# grouping cascade indices by cluster type

# level 7 clusters
lvl7 = clusters.groupby('lvl7_labels')

# cluster order and number of neurons per cluster
cluster_lvl7 = []
for key in lvl7.groups.keys():
    cluster_lvl7.append([key, len(lvl7.groups[key])])

cluster_lvl7 = pd.DataFrame(cluster_lvl7, columns = ['key', 'num_cluster'])

# breaking signal cascades into cluster groups
input_hit_hist_lvl7 = []
for hit_hist in input_hit_hist_list:
    sensory_clustered_hist = []

    for key in lvl7.groups.keys():
        skids = lvl7.groups[key]
        indices = np.where([x in skids for x in mg.meta.index])[0]
        cluster_hist = hit_hist[indices]
        cluster_hist = pd.DataFrame(cluster_hist, index = indices)

        sensory_clustered_hist.append(cluster_hist)
    
    input_hit_hist_lvl7.append(sensory_clustered_hist)

# summed signal cascades per cluster group (hops remain intact)
summed_hist_lvl7 = []
for input_hit_hist in input_hit_hist_lvl7:
    sensory_sum_hist = []
    for i, cluster in enumerate(input_hit_hist):
        sum_cluster = cluster.sum(axis = 0)/(len(cluster.index)) # normalize by number of neurons in cluster
        sensory_sum_hist.append(sum_cluster)

    sensory_sum_hist = pd.DataFrame(sensory_sum_hist) # column names will be hop number
    sensory_sum_hist.index = cluster_lvl7.key # uses cluster name for index of each summed cluster row
    summed_hist_lvl7.append(sensory_sum_hist)


# level 7 clusters
lvl7 = clusters.groupby('lvl7_labels')

# for cascade from all sensories simultaneously
all_sensory_clustered_hist = []
for key in lvl7.groups.keys():
        skids = lvl7.groups[key]
        indices = np.where([x in skids for x in mg.meta.index])[0]
        cluster_hist = all_input_hit_hist[indices]

        all_sensory_clustered_hist.append(cluster_hist)
    
all_sensory_sum_hist = []
for i, cluster in enumerate(all_sensory_clustered_hist):
    sum_cluster = cluster.sum(axis = 0)/(cluster_lvl7.iloc[i].num_cluster) # normalize by number of neurons in cluster
    all_sensory_sum_hist.append(sum_cluster)

all_sensory_sum_hist = pd.DataFrame(all_sensory_sum_hist) # column names will be hop number
all_sensory_sum_hist.index = cluster_lvl7.key # uses cluster name for index of each summed cluster row

# %%
# plot signal of all sensories through clusters

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

fig, axs = plt.subplots(
    1, 1, figsize=(3, 5)
)

vmax = 300

ax = axs
sns.heatmap(sum(summed_hist_lvl7).loc[order, 0:7], ax = ax, vmax = vmax, rasterized=True, cbar_kws={'label': 'Visits from sensory signal', 'orientation': 'horizontal'})
ax.set_ylabel('Individual Clusters')
ax.set_yticks([])
ax.set_xlabel('Hops from sensory signal')

plt.savefig('cascades/cluster_plots/all_sensory_through_clusters_lvl7.pdf', format='pdf', bbox_inches='tight')

# %%
# plot signal of each sensory modality through clusters

fig, axs = plt.subplots(
    4, 2, figsize=(10, 10)
)

fig.tight_layout(pad=2.0)
vmax = n_init*.8

for i in range(0, len(input_names_format_reordered)):
    if(i<4):
        ax = axs[i, 0]
    if(i>=4):
        ax = axs[i-4,1]
    
    sns.heatmap(summed_hist_lvl7[i].loc[order], ax = ax, rasterized=True, vmax = vmax)
    ax.set_ylabel('Individual Clusters')
    ax.set_yticks([])

    ax.set_xlabel('Hops from %s signal' %input_names_format_reordered[i])

    #sns.heatmap(summed_hist_lvl7[1].loc[sort], ax = ax, rasterized=True)

ax = axs[3, 1]
ax.axis("off")
caption = f"Figure: Hop histogram of individual level 7 clusters\nCascades starting at each sensory modality\nending at brain output neurons"
ax.text(0, 1, caption, va="top")

plt.savefig('cascades/cluster_plots/sensory_through_clusters_lvl7.pdf', format='pdf', bbox_inches='tight')

# %%
# mutlisensory nature of each cluster

collapsed_hops_lvl7_list = []
for hist in summed_hist_lvl7:
    collapsed_hops_lvl7_list.append(hist.sum(axis = 1))

collapsed_hops_lvl7 = pd.DataFrame(collapsed_hops_lvl7_list, index = input_names_format_reordered).T

fg = sns.clustermap(collapsed_hops_lvl7.loc[order], col_cluster = False, yticklabels=False, rasterized = True)
ax = fg.ax_heatmap
ax.set_ylabel('Individual Clusters')
fg.savefig('cascades/cluster_plots/multimodal_nature_of_clusters_lvl7.pdf', format='pdf', bbox_inches='tight')

#%%
#
# checking where inputs, outputs, etc are located in these reordered clusters
fg.dendrogram_row.reordered_ind
dendrogram_order = order[fg.dendrogram_row.reordered_ind]
lvl7 = list(clusters.groupby('lvl7_labels'))

cluster_membership = []
for cluster_key in dendrogram_order:
    cluster_temp = clusters[clusters.lvl7_labels==cluster_key]
    cluster_dSEZ = sum(cluster_temp.dSEZ == True)
    cluster_dVNC = sum(cluster_temp.dVNC == True)
    cluster_RG = sum(cluster_temp.RG == True)
    cluster_input = sum(cluster_temp.input == True)
    cluster_output = sum(cluster_temp.output == True)
    cluster_brain = sum(cluster_temp.brain_neurons == True)
    cluster_all = len(cluster_temp.index)

    cluster_membership.append(dict({'cluster_key': cluster_key,
                                'total_neurons': cluster_all, 'brain_neurons': cluster_brain/cluster_all, 
                                'outputs': cluster_output/cluster_all, 'inputs': cluster_input/cluster_all,
                                'dVNC': cluster_dVNC/cluster_all, 'dSEZ': cluster_dSEZ/cluster_all, 
                                'RG': cluster_RG/cluster_all}))

cluster_membership = pd.DataFrame(cluster_membership)
sns.heatmap(cluster_membership.iloc[:, 3:8])

# %%
# how many clusters integrate X number of sensory modalities?
# as cumulative dist plot
from upsetplot import plot
from upsetplot import from_contents
from upsetplot import from_memberships
import itertools

threshold = n_init/2
binary_integration_mat = (collapsed_hops_lvl7>threshold).sum(axis = 1)

lvl7_sensory_inputs = collapsed_hops_lvl7>threshold

lvl7_sensory_content = {}
for column in lvl7_sensory_inputs:
    content = []
    for i, row in enumerate(lvl7_sensory_inputs[column]):
        if(row == True):
            content.append(lvl7_sensory_inputs.index[i])

    lvl7_sensory_content[column] = content
    

plot(from_contents(lvl7_sensory_content), sort_by = 'cardinality', sort_categories_by = None)
#%%
# counts of each type
counts = []
for count in range(0, max(binary_integration_mat)+1):
    current_count = 0
    for i in range(0, len(binary_integration_mat)):
        if (binary_integration_mat[i] == count):
            current_count += 1
    
    counts.append(current_count)

cml_counts = [0,0,0,0,0,0,0,0]
for i, count in enumerate(counts):
    if(i==0):
        cml_counts[i] = count
    if(i>0):
        cml_counts[i] = cml_counts[i-1] + count


sns.lineplot(x = range(0, len(cml_counts)), y = np.array(cml_counts)/sum(counts))
#plt.hist(binary_integration_mat,cumulative=True, density=True, bins=7)
#plt.show()

# %%
# identifying locations of intersecting signals in individual neurons
# which clusters and which neurons?

threshold = n_init/2

# all individual neurons hit by each sensory modality >threshold
# ignores hop number
indices_sensory_hits = []
for hit_hist in input_hit_hist_list:
    indices_hit = []
    for i in range(0, len(hit_hist)):
        if(True in (hit_hist[i]>threshold)):
            indices_hit.append(i)
    
    indices_sensory_hits.append(indices_hit)

# intersection between particular sensory modalities


# intersection between all sensory modalities, ignoring hops
sensory_hits_all = []
for i in range(0, len(input_hit_hist_list[0])):
    sensory_hits = []
    for input_hit_hist in input_hit_hist_list:
        sensory_hits.append(sum(input_hit_hist[i]>threshold)>0)
        #sensory_hits.append(sum(input_hit_hist[i])>0)

    sensory_hits_all.append(sensory_hits)
    
sensory_hits_all = pd.DataFrame(sensory_hits_all, columns = input_names_format_reordered, index = mg.meta.index)

# intersection between all sensory modalities, including hops

sensory_hits_all_hops = []
for hop in range(0, len(input_hit_hist_list[0][0, :])):
    hops = []
    for i in range(0, len(input_hit_hist_list[0])):
        sensory_hits = []
        for input_hit_hist in input_hit_hist_list:
            sensory_hits.append(input_hit_hist[i, hop]>threshold)

        hops.append(sensory_hits)
    
    hops = pd.DataFrame(hops, columns = [s + ('_hop%i' %hop) for s in input_names_format_reordered],
                                index = mg.meta.index)

    sensory_hits_all_hops.append(hops)

# %%
# Upset plot of sensory integration with cascade threshold
# major interesting intersection of signals seems to be AN/MN/ORN and A00c/thermo/visual
from upsetplot import plot
from upsetplot import from_contents
from upsetplot import from_memberships
import itertools

all_hops_content = []
for hop in sensory_hits_all_hops:
    hop_content = {}
    for column in hop:
        content = []
        for i, row in enumerate(hop[column]):
            if(row == True):
                content.append(hop.index[i])

        hop_content[column] = content
    all_hops_content.append(hop_content)

hops = 6

intersections = []
for i in range(0, hops):
    if(i<(hops/2)):
        ax = axs[i, 0]
    if(i>=(hops/2)):
        ax = axs[i-int(hops/2),1]
    
    fg = plot(from_contents(all_hops_content[i]), sort_by = 'cardinality', sort_categories_by = None)
    plt.savefig('cascades/cluster_plots/multimodal_UpSet_plots_hop%i.pdf' %i, format='pdf', bbox_inches='tight')

# %%
# Upset plot of sensory integration with cascade threshold and no hops
# looks a bit different any other plotting methods
# it seems that thresholding before or after collapsing hops changes things a bit
from upsetplot import plot
from upsetplot import from_contents
from upsetplot import from_memberships
import itertools

all_content = {}
for column in sensory_hits_all:
    content = []
    for row in sensory_hits_all.loc[:, column].index:
        if(sensory_hits_all.loc[row, column] == True):
            content.append(row)

    all_content[column] = content

fg = plot(from_contents(all_content), sort_by = 'cardinality', sort_categories_by = None)
#plt.savefig('cascades/cluster_plots/multimodal_UpSet_plots_hop%i.pdf' %i, format='pdf', bbox_inches='tight')

# %%
#
# map back to clusters

# exclusive intersection of modality types
hop_intersection_AN_MN_ORN = []
hop_intersection_thermo_photo_A00c = []

for hop in sensory_hits_all_hops:
    skids_AN_MN_ORN = []
    skids_thermo_photo_A00c = []

    for row in range(0, len(hop.iloc[:,0])):
        if((hop.iloc[row,0]==True) & 
            (hop.iloc[row,1]==True) & 
            (hop.iloc[row,2]==True) &
            (hop.iloc[row,3]==False) &
            (hop.iloc[row,4]==False) &
            (hop.iloc[row,5]==False) &
            (hop.iloc[row,6]==False)):
            skids_AN_MN_ORN.append(hop.iloc[row, :].name)

        if((hop.iloc[row,0]==False) & 
            (hop.iloc[row,1]==False) & 
            (hop.iloc[row,2]==False) &
            (hop.iloc[row,3]==True) &
            (hop.iloc[row,4]==True) &
            (hop.iloc[row,5]==True) &
            (hop.iloc[row,6]==False)):
            skids_thermo_photo_A00c.append(hop.iloc[row, :].name)

    hop_intersection_AN_MN_ORN.append(skids_AN_MN_ORN)
    hop_intersection_thermo_photo_A00c.append(skids_thermo_photo_A00c)

# breakdown into clusters and determine intersection/union for each cluster per hop

def intersect_to_clusters_by_hop(hop_intersection, cluster_lvl):
    hop_intersects = []
    hop_intersects_fraction = []
    hop_intersects_num = []

    for hop in hop_intersection:
        cluster_intersects = []
        cluster_intersects_fraction = []
        cluster_intersects_num = []

        for key in cluster_lvl.groups.keys():
            intersect = np.intersect1d(cluster_lvl.groups[key], hop)
            cluster_intersects.append(intersect)
            cluster_intersects_fraction.append(len(intersect)/len(cluster_lvl.groups[key]))
            cluster_intersects_num.append(len(intersect))

        hop_intersects.append(cluster_intersects)
        hop_intersects_fraction.append(cluster_intersects_fraction)
        hop_intersects_num.append(cluster_intersects_num)
        
    hop_intersects_fraction = pd.DataFrame(hop_intersects_fraction).T
    hop_intersects_fraction.index = cluster_lvl.groups.keys()
        
    hop_intersects_num = pd.DataFrame(hop_intersects_num).T
    hop_intersects_num.index = cluster_lvl.groups.keys()

    return(hop_intersects, hop_intersects_fraction, hop_intersects_num)

lvl7 = clusters.groupby('lvl7_labels')
cluster_hop_inter_AN_MN_ORN, cluster_hop_interFraction_AN_MN_ORN, cluster_hop_interNum_AN_MN_ORN = intersect_to_clusters_by_hop(hop_intersection_AN_MN_ORN, lvl7)
cluster_hop_inter_thermo_photo_A00c, cluster_hop_interFraction_thermo_photo_A00c, cluster_hop_interNum_thermo_photo_A00c = intersect_to_clusters_by_hop(hop_intersection_thermo_photo_A00c, lvl7)

# %%
# plot intersects 

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import cmasher as cmr

fig, axs = plt.subplots(
    2, 1, figsize = (4,5)
)
fig.tight_layout(pad = 2.0)
vmax = 0.5

ax = axs[0]
sns.heatmap(cluster_hop_interFraction_AN_MN_ORN.loc[order, 0:5], 
            vmax = vmax, ax = ax, cbar_kws={'label': 'Overlap between signals'}, cmap = cmr.lavender, rasterized = True)
ax.set_ylabel('Individual Clusters')
ax.set_yticks([])
ax.set_title('Integration of AN, MN, and ORN', fontsize = 10)

ax = axs[1]
sns.heatmap(cluster_hop_interFraction_thermo_photo_A00c.loc[order, 0:5], 
            vmax = vmax, ax = ax, cbar_kws={'label': 'Overlap between signals'}, cmap = cmr.lavender, rasterized = True)

ax.set_ylabel('Individual Clusters')
ax.set_yticks([])
ax.set_xlabel('Hops from sensory signal')
ax.set_title('Integration of thermo, photo, and noci', fontsize = 10)

fig.savefig('cascades/cluster_plots/integration_centers_by_cluster.pdf', bbox_inches='tight')

# %%
# all permutations of possible combinations of signals at different hops

def integration_intersect(sensory_hits_all_hops, which_array, cluster_lvl):
    hop_intersection = []

    for hop in sensory_hits_all_hops:
        skids = []

        for row in range(0, len(hop.iloc[:,0])):

            if((hop.iloc[row,0]==which_array[0]) & 
                (hop.iloc[row,1]==which_array[1]) & 
                (hop.iloc[row,2]==which_array[2]) &
                (hop.iloc[row,3]==which_array[3]) &
                (hop.iloc[row,4]==which_array[4]) &
                (hop.iloc[row,5]==which_array[5]) &
                (hop.iloc[row,6]==which_array[6])):
                skids.append(hop.iloc[row, :].name)

        hop_intersection.append(skids)

    return(intersect_to_clusters_by_hop(hop_intersection, cluster_lvl))

from itertools import product
from tqdm import tqdm

permut7 = list(itertools.product([True, False], repeat=7))
permut7 = [permut7[x] for x in range(len(permut7)-1)] # remove the all False scenario

hop_intersects_list = []
hop_intersects_fraction_list = []
hop_intersects_num_list = []

for i in tqdm(range(0, len(permut7))):
    hop_intersects, hop_intersects_fraction, hop_intersects_num = integration_intersect(sensory_hits_all_hops, permut7[i], lvl7)
    hop_intersects_list.append(hop_intersects)
    hop_intersects_fraction_list.append(hop_intersects_fraction)
    hop_intersects_num_list.append(hop_intersects_num)

permuts_cascade_intersects = []
for permut in hop_intersects_num_list:
    permuts_cascade_intersects.append(permut.sum(axis = 1))

permuts_cascade_intersects_df = pd.DataFrame(permuts_cascade_intersects).T.iloc[:, 0:len(permuts_cascade_intersects)-1]
permut_sort = permuts_cascade_intersects_df.sum(axis = 0).sort_values(ascending=False).index

#%%
# plot permutations, including per hop information
fig, axs = plt.subplots(
    1, 1, figsize = (5, 4)
)
ax = axs
vmax = 20

sns.heatmap(permuts_cascade_intersects_df.loc[order, permut_sort], vmax = vmax, ax = ax, rasterized = True)
ax.set_ylabel('Individual Clusters')
ax.set_xlabel('All Possible Permutations of Sensory Integration')
ax.set_yticks([]);
ax.set_xticks([]);

fig.savefig('cascades/cluster_plots/all_permutations_of_sens_integration.pdf', bbox_inches='tight')

# names for plot
permut_names = []
for permut in permut7:
    names = []
    for i in range(0, len(permut)):
        if(permut[i]==True):
            names.append(input_names_format_reordered[i])
    sep = '_'
    permut_names.append(sep.join(names))

fig, axs = plt.subplots(
    1, 1, figsize = (5, 4)
)
ax = axs

sns.barplot(x = np.array(permut_names)[permut_sort[0:15]], y = permuts_cascade_intersects_df.sum(axis = 0).sort_values(ascending=False)[0:15], ax = ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right');

fig.savefig('cascades/cluster_plots/top_permutations_num_neurons.pdf', bbox_inches='tight')

#%%
#
# make associated UpSet plot
all_hops_content = []
for hop in sensory_hits_all_hops:
    hop_content = {}
    for i, column in enumerate(hop):
        content = []
        for j, row in enumerate(hop[column]):
            if(row == True):
                content.append(hop.index[j])

        hop_content[input_names_format_reordered[i]] = content
    all_hops_content.append(hop_content)

all_hops_from_content = [from_contents(hop) for hop in all_hops_content]
skids_per_hop = [list(hop.id) for hop in all_hops_from_content]

# determine intersection over union for skids included in each hop layer category
# mostly exclusive 
mat_iou = []
for hop in skids_per_hop:
    column = []
    for hop2 in skids_per_hop:
        set1 = np.array(hop)
        set2 = np.array(hop2)
        if(len(np.union1d(set1, set2))>0):
            iou = len(np.intersect1d(set1, set2))/len(np.union1d(set1, set2))
        if(len(np.union1d(set1, set2))==0):
            iou = 0
        column.append(iou)
    
    mat_iou.append(column)

sns.heatmap(mat_iou)

fg = plot(pd.concat(all_hops_from_content), sort_categories_by = None, threshold = 50)
plt.savefig('cascades/cluster_plots/top_permutations_num_neurons_UpSet.pdf', bbox_inches='tight')

# %%
# intersection between all sensory modality permutations, excluding hops 

permut_skids_all = []
for permut_iter in permut7:
    permut_skids = []

    for index, row in sensory_hits_all.iterrows():
        if(tuple(row) == permut_iter):
            #print(tuple(row))
            #print(permut_iter)
            #print(index)
            permut_skids.append(index)

    #print(permut_skids)
    permut_skids_all.append(permut_skids)
    
permut_skids_len = [len(x) for x in permut_skids_all]
permut_sort_nohops = pd.DataFrame(permut_skids_len).sort_values(by = 0, ascending=False).index
permut_skids_all_sorted = [permut_skids_all[x] for x in permut_sort_nohops]
# %%

# intersection between all sensory modality permutations, including hop information
permut_skids_all_hops = []
for permut_iter in permut7:
    permut_skids = []

    for hits_all in sensory_hits_all_hops:
        
        for index, row in hits_all.iterrows():

            if(tuple(row) == permut_iter):

                permut_skids.append(index)

    permut_skids = np.unique(np.array(permut_skids))
    permut_skids_all_hops.append(permut_skids)

permut_skids_hops_len = [len(x) for x in permut_skids_all_hops]
permut_sort_hops = pd.DataFrame(permut_skids_hops_len).sort_values(by = 0, ascending=False).index
permut_skids_all_hops_sorted = [permut_skids_all_hops[x] for x in permut_sort_hops]

[permut7[x] for x in permut_sort_hops[0:12]]
