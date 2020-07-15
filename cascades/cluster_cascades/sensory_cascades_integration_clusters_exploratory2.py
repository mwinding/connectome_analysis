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

# %%
# identifying locations of intersecting signals in individual neurons
# which clusters and which neurons?

threshold = n_init/2

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

#%%
from upsetplot import plot
from upsetplot import from_contents
from upsetplot import from_memberships

threshold = n_init/2

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

fg = plot(pd.concat(all_hops_from_content), sort_categories_by = None)
plt.savefig('cascades/cluster_plots/top_permutations_num_neurons_UpSet.pdf', bbox_inches='tight')

# %%
# all permutations of possible combinations of signals at different hops
import itertools 
from tqdm import tqdm

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

permut7 = list(itertools.product([True, False], repeat=7))
permut7 = [permut7[x] for x in range(len(permut7)-1)] # remove the all False scenario

hop_intersects_list = [] # list of list of array of int; levels: sensory integration type (see permut7), hop number, cluster number, skids
hop_intersects_fraction_list = [] # list of pandas dataframe; levels: sensory integration type (see permut7), columns = hop number, rows = fraction of cluster members involved in integration
hop_intersects_num_list = [] # list of pandas dataframe; levels: sensory integration type (see permut7), columns = hop number, rows = number of cluster members involved in integration 

for i in tqdm(range(0, len(permut7))):
    hop_intersects, hop_intersects_fraction, hop_intersects_num = integration_intersect(sensory_hits_all_hops, permut7[i], lvl7)
    hop_intersects_list.append(hop_intersects)
    hop_intersects_fraction_list.append(hop_intersects_fraction)
    hop_intersects_num_list.append(hop_intersects_num)

permuts_cascade_intersects = []
for permut in hop_intersects_num_list:
    permuts_cascade_intersects.append(permut.sum(axis = 1))

permuts_cascade_intersects_df = pd.DataFrame(permuts_cascade_intersects).T.iloc[:, 0:len(permuts_cascade_intersects)]
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

#%%
# plot cascades of major integration types through clusters and major non-integration types through clusters
# probably supplemental figure

# names for plot
permut_names = []
for permut in permut7:
    names = []
    for i in range(0, len(permut)):
        if(permut[i]==True):
            names.append(input_names_format_reordered[i])
    sep = '_'
    permut_names.append(sep.join(names))

integraton_names = [permut_names[x] for x in permut_sort if sum(permut7[x])>1][0:6] # names of top 6 integration types
integration_types = [hop_intersects_fraction_list[x] for x in permut_sort if sum(permut7[x])>1][0:6] # fraction in clusters for top 6 integration types

nonintegration_names = [permut_names[x] for x in permut_sort if sum(permut7[x])==1] # names of non-integration types
nonintegration_types = [hop_intersects_fraction_list[x] for x in permut_sort if sum(permut7[x])==1] # fraction in clusters for non-integration types

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import cmasher as cmr

fig, axs = plt.subplots(
    6, 2, figsize = (10,15)
)
fig.tight_layout(pad = 2.0)
vmax = 0.5
for i in range(0, 6):
    ax = axs[i, 0]
    sns.heatmap(integration_types[i].loc[order, 0:5], 
                vmax = vmax, ax = ax, cbar_kws={'label': 'Fraction of Cluster'}, cmap = cmr.lavender, rasterized = True)
    ax.set_ylabel('Individual Clusters')
    ax.set_yticks([])
    ax.set_title('Integration of %s' %integraton_names[i], fontsize = 10)
    if(i == 5):    
        ax.set_xlabel('Hops from Sensory')

for i in range(0, 6):
    ax = axs[i, 1]
    sns.heatmap(nonintegration_types[i].loc[order, 0:5], 
                vmax = vmax, ax = ax, cbar_kws={'label': 'Fraction of Cluster'}, cmap = cmr.sunburst, rasterized = True)
    ax.set_ylabel('Individual Clusters')
    ax.set_yticks([])
    ax.set_title('%s-specific Neurons' %nonintegration_names[i], fontsize = 10)
    if(i == 5):    
        ax.set_xlabel('Hops from Sensory')

fig.savefig('cascades/cluster_plots/top_permutations_of_sens_integration_with_hops_vertical.pdf', bbox_inches='tight')

# %%
# same plot but horizontal 

fig, axs = plt.subplots(
    2, 7, figsize = (15,5)
)
fig.tight_layout(pad = 2.0)
vmax = 0.5
for i in range(0, 6):
    ax = axs[0, i]
    sns.heatmap(integration_types[i].loc[order, 0:5], 
                vmax = vmax, ax = ax, cbar_kws={'label': 'Fraction of Cluster'}, cmap = cmr.lavender, rasterized = True, cbar_ax = axs[0, 6])
    ax.set_yticks([])
    ax.set_title('%s' %integraton_names[i], fontsize = 10)
    if(i == 0):    
        ax.set_ylabel('Individual Clusters')


for i in range(0, 6):
    ax = axs[1, i]
    sns.heatmap(nonintegration_types[i].loc[order, 0:5], 
                vmax = vmax, ax = ax, cbar_kws={'label': 'Fraction of Cluster'}, cmap = cmr.sunburst, rasterized = True, cbar_ax = axs[1, 6])
    ax.set_xlabel('Hops from Sensory')
    ax.set_yticks([])
    ax.set_title('%s-specific' %nonintegration_names[i], fontsize = 10)
    if(i == 0):    
        ax.set_ylabel('Individual Clusters')

fig.savefig('cascades/cluster_plots/top_permutations_of_sens_integration_with_hops_horizontal.pdf', bbox_inches='tight')

# %%
# Character of each cluster in stacked plot

# calculation of number of neurons with no input from any of the current sensories
#hop_intersects_nothing, hop_intersects_fraction_nothing, hop_intersects_num_nothing = integration_intersect(sensory_hits_all_hops, [False for x in range(0, 7)], lvl7)

# from all types separate
cluster_character = []
for permut in hop_intersects_fraction_list:
    cluster_character.append(permut.sum(axis = 1))

cluster_character_df = pd.DataFrame(cluster_character).T.iloc[:, 0:len(cluster_character)]
cluster_character_df_max = cluster_character_df.sum(axis = 1)

cluster_character_df_norm = []
for key in cluster_character_df.index:
    if (cluster_character_df_max.loc[key]>1):
        cluster_character_df_norm.append(cluster_character_df.loc[key]/cluster_character_df_max.loc[key])

    if(cluster_character_df_max.loc[key]<=1):
        cluster_character_df_norm.append(cluster_character_df.loc[key])

cluster_character_df_norm = pd.DataFrame(cluster_character_df_norm)

sns.heatmap(cluster_character_df_norm.loc[order, permut_sort[0:12]])


# from either modality-specific or multi-modal
from pandas.plotting import parallel_coordinates

nonintegrative = [hop_intersects_fraction_list[x] for x in permut_sort if sum(permut7[x])==1]
total_nonintegrative = sum(nonintegrative).sum(axis = 1) 
integrative = [hop_intersects_fraction_list[x] for x in permut_sort if sum(permut7[x])>1]
total_integrative = sum(integrative).sum(axis = 1) 

both_df = pd.DataFrame([total_nonintegrative, total_integrative], index = ['Sensory-specific', 'Integrative']).T
both_df_plot = both_df.T

fig, axs = plt.subplots(
    2, 1, figsize = (8,5)
)
alpha = 1

ax = axs[0]
sns.lineplot(x = list(range(0, len(both_df_plot.loc['Integrative', order]))), y = both_df_plot.loc['Integrative', order], color = sns.color_palette()[0], ax = ax, alpha = alpha)

ax = axs[1]
sns.lineplot(x = list(range(0, len(both_df_plot.loc['Sensory-specific', order]))), y = both_df_plot.loc['Sensory-specific', order], color = sns.color_palette()[1], ax = ax, alpha = alpha)
# need to identify each unique cell and generate fractions that way for the integration side

# %%
# plotting integrative vs non-integrative types by hop

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

nonintegration_types_hops = sum(nonintegration_types).T
integration_types_hops = sum(integration_types).T

fig, axs = plt.subplots(
    2, 1, figsize = (8,5)
)

fig.tight_layout(pad = 3.0)
alpha = 0.8
labels = ['hop 0', 'hop 1', 'hop 2', 'hop 3', 'hop 4', 'hop 5']

ax = axs[0]
ax.stackplot(list(range(0, len(nonintegration_types_hops.loc[0, order]))), nonintegration_types_hops.loc[0:5, order], alpha = alpha)
#ax.set_xlabel('Individual Clusters (ordered from high to low signal flow)')
ax.set_ylabel('Fraction of Cluster')
ax.set_title('Participation in Sensory-specific Signals')
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.set(xlim = (0, len(nonintegration_types_hops.loc[0, order])))
ax.set(ylim = (0, 1))

ax = axs[1]
ax.stackplot(list(range(0, len(integration_types_hops.loc[0, order]))), integration_types_hops.loc[0:5, order], alpha= alpha, labels = labels) 
ax.set_xlabel('Individual Clusters (ordered from high to low signal flow)')
ax.set_ylabel('Fraction of Cluster')
ax.set_title('Participation in Sensory Signal Integration')
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.set(xlim = (0, len(integration_types_hops.loc[0, order])))
ax.set(ylim = (0, 1.7))
ax.legend(loc='center left', bbox_to_anchor=(1.02, 1.20))

fig.savefig('cascades/cluster_plots/sensory-specific_vs_integrative_hops.pdf', bbox_inches='tight')

# %%
