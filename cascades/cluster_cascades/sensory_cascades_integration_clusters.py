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

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 6})

rm = pymaid.CatmaidInstance(url, token, name, password)

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

# tested a few different methods to define "integration neurons"
# settled on summing all visits across hops and then 50% signal threshold 
#   (majority of the time the signal goes through each particular neuron)
#   *** this is summed_sensory_hits_all

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
    
    hops = pd.DataFrame(hops, columns = input_names_format_reordered,
                                index = mg.meta.index)

    sensory_hits_all_hops.append(hops)

# intersection between all sensory modalities, thresholding then ignoring hops
sensory_hits_all = []
for i in range(0, len(input_hit_hist_list[0])):
    sensory_hits = []
    for input_hit_hist in input_hit_hist_list:
        sensory_hits.append(sum(input_hit_hist[i]>threshold)>0)

    sensory_hits_all.append(sensory_hits)
    
sensory_hits_all = pd.DataFrame(sensory_hits_all, columns = input_names_format_reordered, index = mg.meta.index)

# intersection between all sensory modalities, summing hops then thresholding
summed_sensory_hits_all = []
for i in range(0, len(input_hit_hist_list[0])):
    sensory_hits = []
    for input_hit_hist in input_hit_hist_list:
        sensory_hits.append(sum(input_hit_hist[i][0:8])>threshold)

    summed_sensory_hits_all.append(sensory_hits)
    
summed_sensory_hits_all = pd.DataFrame(summed_sensory_hits_all, columns = input_names_format_reordered, index = mg.meta.index)

# intersection between all sensory modalities, summing hops (all 10 hops) then thresholding
summed_sensory_hits_all2 = []
for i in range(0, len(input_hit_hist_list[0])):
    sensory_hits = []
    for input_hit_hist in input_hit_hist_list:
        sensory_hits.append(sum(input_hit_hist[i][0:10])>threshold)

    summed_sensory_hits_all2.append(sensory_hits)
    
summed_sensory_hits_all2 = pd.DataFrame(summed_sensory_hits_all2, columns = input_names_format_reordered, index = mg.meta.index)

# intersection between all sensory modalities, sliding window sum hops then thresholding

summed_window_sensory_hits_all = []
for i in range(0, len(input_hit_hist_list[0])):
    sensory_hits = []
    for input_hit_hist in input_hit_hist_list:
        max_sum = max([sum(input_hit_hist[i][0:3]),
                    sum(input_hit_hist[i][1:4]),
                    sum(input_hit_hist[i][2:5]),
                    sum(input_hit_hist[i][3:6]),
                    sum(input_hit_hist[i][4:7]),
                    sum(input_hit_hist[i][5:8])])
        sensory_hits.append(max_sum>threshold)

    summed_window_sensory_hits_all.append(sensory_hits)
    
summed_window_sensory_hits_all = pd.DataFrame(summed_window_sensory_hits_all, columns = input_names_format_reordered, index = mg.meta.index)

summed_window_small_sensory_hits_all = []
for i in range(0, len(input_hit_hist_list[0])):
    sensory_hits = []
    for input_hit_hist in input_hit_hist_list:
        max_sum = max([sum(input_hit_hist[i][0:2]),
                    sum(input_hit_hist[i][1:3]),
                    sum(input_hit_hist[i][2:4]),
                    sum(input_hit_hist[i][3:5]),
                    sum(input_hit_hist[i][4:6]),
                    sum(input_hit_hist[i][5:7]),
                    sum(input_hit_hist[i][6:8])])
        sensory_hits.append(max_sum>threshold)

    summed_window_small_sensory_hits_all.append(sensory_hits)
    
summed_window_small_sensory_hits_all = pd.DataFrame(summed_window_small_sensory_hits_all, columns = input_names_format_reordered, index = mg.meta.index)

# %%
# all permutations of sensory integration, ignoring hops

from upsetplot import plot
from upsetplot import from_contents
from upsetplot import from_memberships

mi = pd.MultiIndex.from_frame(sum(sensory_hits_all_hops)>0)
data = pd.DataFrame(sensory_hits_all_hops[0].index, index = mi)
plot(data, sort_by = 'cardinality', sort_categories_by = None)

plt.savefig('cascades/cluster_plots/threshold_types_on-hops.pdf', format='pdf', bbox_inches='tight')

mi = pd.MultiIndex.from_frame(summed_sensory_hits_all)
data = pd.DataFrame(sensory_hits_all_hops[0].index, index = mi)
sum_plot = plot(data, sort_by = 'cardinality', sort_categories_by = None)

plt.savefig('cascades/cluster_plots/threshold_types_summed.pdf', format='pdf', bbox_inches='tight')

mi = pd.MultiIndex.from_frame(summed_sensory_hits_all2)
data = pd.DataFrame(sensory_hits_all_hops[0].index, index = mi)
sum_plot = plot(data, sort_by = 'cardinality', sort_categories_by = None)

plt.savefig('cascades/cluster_plots/threshold_types_summed2.pdf', format='pdf', bbox_inches='tight')

mi = pd.MultiIndex.from_frame(summed_window_sensory_hits_all)
data = pd.DataFrame(sensory_hits_all_hops[0].index, index = mi)
plot(data, sort_by = 'cardinality', sort_categories_by = None)

plt.savefig('cascades/cluster_plots/threshold_types_summed-window.pdf', format='pdf', bbox_inches='tight')

mi = pd.MultiIndex.from_frame(summed_window_small_sensory_hits_all)
data = pd.DataFrame(sensory_hits_all_hops[0].index, index = mi)
plot(data, sort_by = 'cardinality', sort_categories_by = None)
plt.savefig('cascades/cluster_plots/threshold_types_summed-small-window.pdf', format='pdf', bbox_inches='tight')

# %%
# identify skids in each category of sensory combination

import itertools
from tqdm import tqdm

permut7 = list(itertools.product([True, False], repeat=7))
permut7 = [permut7[x] for x in range(len(permut7)-1)] # remove the all False scenario

permut_members = []

for i, permut in enumerate(permut7):
    skids = []
    for row in summed_sensory_hits_all.iterrows():
        if((row[1][0]==permut[0]) & 
            (row[1][1]==permut[1]) & 
            (row[1][2]==permut[2]) &
            (row[1][3]==permut[3]) &
            (row[1][4]==permut[4]) &
            (row[1][5]==permut[5]) &
            (row[1][6]==permut[6])):
            skids.append(row[1].name)

    permut_members.append(skids)
'''
permut_members2 = []

for i, permut in enumerate(permut7):
    skids = []
    for row in summed_sensory_hits_all2.iterrows():
        if((row[1][0]==permut[0]) & 
            (row[1][1]==permut[1]) & 
            (row[1][2]==permut[2]) &
            (row[1][3]==permut[3]) &
            (row[1][4]==permut[4]) &
            (row[1][5]==permut[5]) &
            (row[1][6]==permut[6])):
            skids.append(row[1].name)

    permut_members2.append(skids)
'''
# where does sensory integration occur? for each type?
def skid_to_index(skid, mg):
    index_match = np.where(mg.meta.index == skid)[0]
    if(len(index_match)==1):
        return(index_match[0])
    if(len(index_match)!=1):
        print('Not one match for skid %i!' %skid)

def index_to_skid(index, mg):
    return(mg.meta.iloc[index, :].name)

def counts_to_list(count_list):
    expanded_counts = []
    for i, count in enumerate(count_list):
        expanded = np.repeat(i, count)
        expanded_counts.append(expanded)
    
    return([x for sublist in expanded_counts for x in sublist])


# identify median hop number per skid
hops_from_sens = []

for i, input_hit_hist in enumerate(input_hit_hist_list):
    for j, row in enumerate(input_hit_hist):
        median_hop = np.median(counts_to_list(row))
        skid = index_to_skid(j, mg)
        if(summed_sensory_hits_all.loc[skid, input_names_format_reordered[i]] == True):
            hops_from_sens.append([skid, median_hop, input_names_format_reordered[i]])

hops_from_sens = pd.DataFrame(hops_from_sens, columns = ['skid', 'hops_from_sens', 'sensory_modality'])

hops_from_sens_skid = hops_from_sens.groupby('skid')
hops_keys = list(hops_from_sens_skid.groups.keys())

# %%
# UpSet plot for figure

length_permut_members = [len(x) for x in permut_members]
sort = sorted(range(len(length_permut_members)), reverse = True, key = lambda k: length_permut_members[k])

#length_permut_members2 = [len(x) for x in permut_members2]
#sort = sorted(range(len(length_permut_members2)), reverse = True, key = lambda k: length_permut_members2[k])

#subset = [permut_members[x] for x in [0, 1, 95, 111, 63, 15, 3, 123, 125, 126, 119]] # sort[0:8] + all sensory-specific
subset = [permut_members[x] for x in sort[0:17]]
subset = [item for sublist in subset for item in sublist]

subset_upset = summed_sensory_hits_all.loc[subset]

mi = pd.MultiIndex.from_frame(subset_upset)
data = pd.DataFrame(subset, index = mi)
sum_plot = plot(data, sort_categories_by = None)
plt.savefig('cascades/cluster_plots/Integration_Upset_Plot.pdf', format='pdf', bbox_inches='tight')


# %%
# integrative vs nonintegrative

indices_nonintegrative = [i for i, x in enumerate(permut7) if sum(x)==1]
indices_integrative = [i for i, x in enumerate(permut7) if sum(x)>1]

# no comment on brain vs non-brain neurons
# ****change to brain neurons only?
total_nonintegrative = sum([len(permut_members[x]) for x in indices_nonintegrative])
total_integrative = sum([len(permut_members[x]) for x in indices_integrative])
no_signal = len(summed_sensory_hits_all.iloc[:, 0]) - total_nonintegrative - total_integrative

integrative_skids = [permut_members[x] for x in indices_integrative]
integrative_skids = [item for sublist in integrative_skids for item in sublist]
nonintegrative_skids = [permut_members[x] for x in indices_nonintegrative]
nonintegrative_skids = [item for sublist in nonintegrative_skids for item in sublist]

# how many hops from each sensory type

mean_hops_integrative = []
for skid in integrative_skids:
    mean_hop = np.mean(hops_from_sens.iloc[hops_from_sens_skid.groups[skid], 1])
    mean_hops_integrative.append([skid, mean_hop, 'Integrative'])

#mean_hops = pd.DataFrame(np.concatenate([mean_hops_integrative, mean_hops_nonintegrative]), 
#            columns = ['skid', 'mean_hops', 'integration'])
#mean_hops['mean_hops'] = mean_hops['mean_hops'].astype('float64')

all_hops_integrative = []
for skid in integrative_skids:
    hops = hops_from_sens.iloc[hops_from_sens_skid.groups[skid], 1]
    for hop in hops:
        all_hops_integrative.append([skid, hop, 'Integrative'])

all_hops_nonintegrative = []
for skid in nonintegrative_skids:
    hop = hops_from_sens.iloc[hops_from_sens_skid.groups[skid], 1]
    all_hops_nonintegrative.append([skid, hop, 'Labelled Line'])

all_hops = pd.DataFrame(np.concatenate([all_hops_integrative, all_hops_nonintegrative]), 
            columns = ['skid', 'all_hops', 'integration'])
all_hops['all_hops'] = all_hops['all_hops'].astype('float64')


fig, axs = plt.subplots(
    1, 1, figsize = (1.5,1.75)
)
fig.tight_layout(pad = 2.0)
ax = axs

sns.violinplot(data = all_hops, x = 'integration', y = 'all_hops', ax = ax, linewidth = 0.5)
ax.set_ylabel('Hops from Sensory', fontsize = 6)
ax.set_xlabel('')
ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
ax.set_xticklabels( ('Integrative\nN=%i' %len(mean_hops_integrative), 'Labelled Line\nN=%i' %len(all_hops_nonintegrative)) )
fig.savefig('cascades/cluster_plots/Integrative_LabelledLine_plot.pdf', format='pdf', bbox_inches='tight')
plt.rcParams["font.family"] = "Arial"

# %%
# Where do integrations occur?

all_hops_integrative_first_vs_other = []
for skid in integrative_skids:
    hops = hops_from_sens.iloc[hops_from_sens_skid.groups[skid], 1]
    gate = 0
    for hop in hops:
        if((hop == min(hops)) & (gate == 0)):
            all_hops_integrative_first_vs_other.append([skid, hop, 'Integrative', 'First'])
            gate += 1
        if((hop != min(hops)) | (gate != 0)):
            all_hops_integrative_first_vs_other.append([skid, hop, 'Integrative', 'Additional'])

all_hops_integrative_first_vs_other = pd.DataFrame(all_hops_integrative_first_vs_other, 
        columns = ['skid', 'all_hops', 'integration', 'step'])

all_hops_integrative_first_vs_other.index = all_hops_integrative_first_vs_other.skid

fig, axs = plt.subplots(
    1, 1, figsize = (2,2)
)
fig.tight_layout(pad = 2.0)
ax = axs

vplt = sns.violinplot(data = all_hops_integrative_first_vs_other, x = 'integration', y = 'all_hops', 
                hue='step', split = True, ax = ax, linewidth = 0.5, legend_out=True, hue_order = ['First', 'Additional'])
ax.set_ylabel('Hops from Sensory', fontsize = 6)
ax.set_xlabel('')
ax.set_xticklabels( ('Integrative\nN=%i' %len(mean_hops_integrative), 'Labelled Line\nN=%i' %len(all_hops_nonintegrative)) )
fig.savefig('cascades/cluster_plots/Integrative_detail_plot.pdf', format='pdf')
plt.rcParams["font.family"] = "Arial"

# %%
# integration detail hop plots for each type in UpSet plot

# permutations to sensory names
# names for plot
permut_names = []
for permut in permut7:
    names = []
    for i in range(0, len(permut)):
        if(permut[i]==True):
            names.append(input_names_format_reordered[i])
    sep = ' + '
    permut_names.append(sep.join(names))

permut_names = np.array(permut_names)

# identifying indices to be used in permut_members
nonintegrative_indices = [i for i, x in enumerate(permut7) if sum(x)==1]
integrative_indices = [x for x in sort[0:17] if (x not in nonintegrative_indices)]

permut_col = []
for skid in all_hops_integrative_first_vs_other.skid:
    for i, permut in enumerate(permut_members):
        for permut_skid in permut:
            if(skid == permut_skid):
                permut_col.append([skid, i])

permut_col = pd.DataFrame(permut_col, columns = ['skid', 'permut_index'])
all_hops_integrative_first_vs_other['permut_index'] = permut_col.permut_index.values
all_hops_integrative_first_vs_other['permut_name'] = permut_names[permut_col.permut_index.values]
all_hops_integrative_first_vs_other.index = permut_col.permut_index.values

fig, axs = plt.subplots(
    1, 1, figsize = (3.75,1.25)
)
fig.tight_layout(pad = 2.0)
ax = axs
sns.violinplot(data = all_hops_integrative_first_vs_other.loc[integrative_indices], 
                x = 'permut_name', y = 'all_hops', scale = 'width', hue = 'step', split=True, 
                hue_order=['First','Additional'], ax = ax, linewidth = 0.5)
ax.set_ylabel('Hops from Sensory', fontsize = 6)
ax.set_xlabel('')
plt.xticks(rotation=45, ha = 'right')
ax.set(ylim = (0, 8))
ax.set_yticks(np.arange(0, 9, 1))
plt.savefig('cascades/cluster_plots/Integrative_hop_violinplots_labels.pdf', format='pdf', bbox_inches='tight')

fig, axs = plt.subplots(
    1, 1, figsize = (3.75,1.25)
)
fig.tight_layout(pad = 2.0)
ax = axs
sns.violinplot(data = all_hops_integrative_first_vs_other.loc[integrative_indices], 
                x = 'permut_name', y = 'all_hops', scale = 'width', hue = 'step', split=True, 
                hue_order=['First','Additional'], ax = ax, linewidth = 0.5)
ax.set_ylabel('Hops from Sensory', fontsize = 6)
ax.set_xlabel('')
plt.xticks([])
ax.set(ylim = (0, 8))
ax.set_yticks(np.arange(0, 9, 1))
plt.savefig('cascades/cluster_plots/Integrative_hop_violinplots_nolabels.pdf', format='pdf', bbox_inches='tight')

#%%
# same but with nonintegrative

all_hops_nonintegrative = []
for skid in nonintegrative_skids:
    hops = hops_from_sens.iloc[hops_from_sens_skid.groups[skid], 1]
    gate = 0
    for hop in hops:
        if((hop == min(hops)) & (gate == 0)):
            all_hops_nonintegrative.append([skid, hop, 'Labelled Line', 'First'])
            gate += 1
        if((hop != min(hops)) | (gate != 0)):
            all_hops_nonintegrative.append([skid, hop, 'Labelled Line', 'Additional'])

all_hops_nonintegrative = pd.DataFrame(all_hops_nonintegrative, 
        columns = ['skid', 'all_hops', 'integration', 'step'])

permut_col_nonintegrative = []
for skid in all_hops_nonintegrative.skid:
    for i, permut in enumerate(permut_members):
        for permut_skid in permut:
            if(skid == permut_skid):
                permut_col_nonintegrative.append([skid, i])

permut_col_nonintegrative = pd.DataFrame(permut_col_nonintegrative, columns = ['skid', 'permut_index'])
all_hops_nonintegrative['permut_index'] = permut_col_nonintegrative.permut_index.values
all_hops_nonintegrative['permut_name'] = permut_names[permut_col_nonintegrative.permut_index.values]
all_hops_nonintegrative.index = permut_col_nonintegrative.permut_index.values

fig, axs = plt.subplots(
    1, 1, figsize = (3.75,1.25)
)
fig.tight_layout(pad = 2.0)
ax = axs
sns.violinplot(data = all_hops_nonintegrative.loc[nonintegrative_indices], 
                x = 'permut_name', y = 'all_hops', scale = 'width', ax = ax, linewidth = 0.5)
ax.set_ylabel('Hops from Sensory', fontsize = 6)
ax.set_xlabel('')
ax.set(ylim = (0, 8))
ax.set_yticks(np.arange(0, 9, 1))
plt.savefig('cascades/cluster_plots/Labelled_line_hop_violinplots.pdf', format='pdf', bbox_inches='tight')

# %%
# how many outputs associated with each type?

labelledline_descendings = []
for i in nonintegrative_indices:
    skids = np.unique(all_hops_nonintegrative.loc[i].skid)
    labelledline_descendings.append([i, sum(mg.meta.loc[skids].dVNC), sum(mg.meta.loc[skids].dSEZ), sum(mg.meta.loc[skids].RG)])

labelledline_descendings = pd.DataFrame(labelledline_descendings, columns = ['permut_number', 'dVNCs', 'dSEZs', 'RG'])
labelledline_descendings.index = labelledline_descendings.permut_number
labelledline_descendings['permut_name'] = permut_names[nonintegrative_indices]

integrative_descendings = []
for i in integrative_indices:
    skids = np.unique(all_hops_integrative_first_vs_other.loc[i].skid)
    integrative_descendings.append([i, 
                                    sum(mg.meta.loc[skids].dVNC), sum(mg.meta.loc[skids].dSEZ), sum(mg.meta.loc[skids].RG)])

integrative_descendings = pd.DataFrame(integrative_descendings, columns = ['permut_number', 'dVNCs', 'dSEZs', 'RG'])
integrative_descendings.index = integrative_descendings.permut_number
integrative_descendings['permut_name'] = permut_names[integrative_indices]

fig, axs = plt.subplots(
    1, 1, figsize=(3.75,.4)
)
ax = axs
sns.heatmap(labelledline_descendings.iloc[:, 1:4].T, ax = ax, annot=True, cmap = 'Blues')
fig.savefig('cascades/cluster_plots/Labelled_line_hop_violinplots_bottom.pdf', format='pdf', bbox_inches='tight')


fig, axs = plt.subplots(
    1, 1, figsize=(3.75,.4)
)
ax = axs
#ax.set_xticklables(integrative_descendings.permut_name.values)
sns.heatmap(integrative_descendings.iloc[:, 1:4].T, ax = ax, annot=True, cmap = 'Oranges')
fig.savefig('cascades/cluster_plots/Integrative_hop_violinplots_bottom.pdf', format='pdf', bbox_inches='tight')

# order in main figure
# 31, 79, 15, 47, 13, 3, 7, 1, 0
fig, axs = plt.subplots(
    1, 1, figsize=(3.75,.4)
)
ax = axs
sns.heatmap(integrative_descendings.loc[[31, 79, 15, 47, 13, 3, 7, 1, 0], ['dVNCs', 'dSEZs', 'RG']].T, ax = ax, annot=True, cmap = 'Oranges')
fig.savefig('cascades/cluster_plots/Integrative_hop_violinplots_bottom_alt.pdf', format='pdf', bbox_inches='tight')

# %%

#Questions
# which clusters contain these neuron types?
# which clusters display which types of sensory input and how much?

# %%
# plotting sensory integration make-up per cluster

# level 7 clusters
lvl7 = clusters.groupby('lvl7_labels')

# integration types per cluster
cluster_lvl7 = []
for key in lvl7.groups.keys():
    for i, permut in enumerate(permut_members):
        for permut_skid in permut:
            if((permut_skid in lvl7.groups[key].values) & (i in nonintegrative_indices)):
                cluster_lvl7.append([key, permut_skid, i, permut_names[i], 'labelled_line'])
            if((permut_skid in lvl7.groups[key].values) & (i not in nonintegrative_indices)):
                cluster_lvl7.append([key, permut_skid, i, permut_names[i], 'integrative'])

cluster_lvl7 = pd.DataFrame(cluster_lvl7, columns = ['key', 'skid', 'permut_index', 'permut_name', 'integration'])


cluster_lvl7_groups = cluster_lvl7.groupby('key')

percent_integrative = [sum(x[1].integration == 'integrative')/len(x[1].integration) for x in list(cluster_lvl7_groups)]
percent_labelled_line = [sum(x[1].integration == 'labelled_line')/len(x[1].integration) for x in list(cluster_lvl7_groups)]    

percent_labelled_line_subtypes = []
for index in nonintegrative_indices:
    percent_labelled_line_subtypes.append(
        [sum(x[1].permut_index == index)/len(x[1].permut_index) for x in list(cluster_lvl7_groups)]
    )

percent_integrative_subtypes = []
for index in integrative_indices:
    percent_integrative_subtypes.append(
        [sum(x[1].permut_index == index)/len(x[1].permut_index) for x in list(cluster_lvl7_groups)]
    )

cluster_character = pd.DataFrame([percent_integrative, percent_labelled_line], columns = lvl7.groups.keys(), index = ['integrative', 'labelled_line']).T
cluster_character_sub_labelled_line = pd.DataFrame(percent_labelled_line_subtypes, columns = lvl7.groups.keys(), 
                                    index = [permut_names[x] for x in nonintegrative_indices]).T
cluster_character_sub_integrative = pd.DataFrame(percent_integrative_subtypes, columns = lvl7.groups.keys(), 
                                    index = integrative_indices).T

import cmasher as cmr

fig, axs = plt.subplots(
    1, 1, figsize = (1.5, 2)
)
ax = axs
ax.set_ylabel('Individual Clusters')
ax.set_yticks([]);
ax.set_xticks([]);
sns.heatmap(cluster_character.loc[order, ['labelled_line', 'integrative']], cmap = 'Greens', rasterized = True)
fig.savefig('cascades/cluster_plots/Clusters_ll_vs_integrative.pdf', format='pdf', bbox_inches='tight')

ind = np.arange(0, len(cluster_character.index))
data1 = cluster_character.loc[order, ['labelled_line']].values
data1 = [x for sublist in data1 for x in sublist]
data2 = cluster_character.loc[order, ['integrative']].values
data2 = [x for sublist in data2 for x in sublist]
#data2 =  np.array(data1) + np.array(data2)

plt.bar(ind, data1, color = 'orange', alpha = 0.5)
plt.bar(ind, data2, bottom = data1, color = 'blue', alpha = 0.5)

fig, axs = plt.subplots(
    1, 1, figsize = (5, 5)
)
ax = axs
ax.set_ylabel('Individual Clusters')
ax.set_yticks([]);
ax.set_xticks([]);
sns.heatmap(cluster_character_sub_labelled_line.loc[order], cmap = 'Greens', rasterized = True, ax = ax)
fig.savefig('cascades/cluster_plots/Clusters_labelled_line_character.pdf', format='pdf', bbox_inches='tight')

fig, axs = plt.subplots(
    1, 1, figsize = (5, 5)
)
ax = axs
ax.set_ylabel('Individual Clusters')
ax.set_yticks([]);
ax.set_xticks([]);
sns.heatmap(cluster_character_sub_integrative.loc[order], cmap = 'Greens', rasterized = True, ax = ax)
fig.savefig('cascades/cluster_plots/Clusters_integrative_character.pdf', format='pdf', bbox_inches='tight')

# stacked bar plot for all types of integrative and non integrative
ORN_frac = cluster_character_sub_labelled_line.loc[order,'ORN'].values
AN_frac = cluster_character_sub_labelled_line.loc[order,'AN'].values
MN_frac = cluster_character_sub_labelled_line.loc[order,'MN'].values
thermo_frac = cluster_character_sub_labelled_line.loc[order,'thermo'].values
photo_frac = cluster_character_sub_labelled_line.loc[order,'photo'].values
A00c_frac = cluster_character_sub_labelled_line.loc[order,'A00c'].values
vtd_frac = cluster_character_sub_labelled_line.loc[order,'vtd'].values
labelledline_frac = ORN_frac + AN_frac + MN_frac + thermo_frac + photo_frac + A00c_frac + vtd_frac

all_integrative_frac = cluster_character_sub_integrative.loc[order, 0].values
most_integrative_frac = cluster_character_sub_integrative.loc[order, 1].values
OR_AN_MN_integrative_frac = cluster_character_sub_integrative.loc[order, 15].values
rest_integrative_frac = cluster_character_sub_integrative.loc[order, :].sum(axis = 1) - all_integrative_frac - most_integrative_frac - OR_AN_MN_integrative_frac

plt.bar(ind, ORN_frac, color = 'blue')
plt.bar(ind, AN_frac, bottom = ORN_frac, color = 'tab:blue')
plt.bar(ind, MN_frac, bottom = ORN_frac + AN_frac, color = 'tab:cyan')
plt.bar(ind, thermo_frac, bottom = ORN_frac + AN_frac + MN_frac, color = 'purple')
plt.bar(ind, photo_frac, bottom = ORN_frac + AN_frac + MN_frac + thermo_frac, color = 'tab:purple')
plt.bar(ind, A00c_frac, bottom = ORN_frac + AN_frac + MN_frac + thermo_frac + photo_frac, color = 'mediumorchid')
plt.bar(ind, vtd_frac, bottom = ORN_frac + AN_frac + MN_frac + thermo_frac + photo_frac + A00c_frac, color = 'plum')

plt.bar(ind, all_integrative_frac, bottom = labelledline_frac, color = 'maroon')
plt.bar(ind, most_integrative_frac, bottom = labelledline_frac + all_integrative_frac, color = 'firebrick')
plt.bar(ind, OR_AN_MN_integrative_frac, bottom = labelledline_frac + all_integrative_frac + most_integrative_frac, color = 'salmon')
plt.bar(ind, rest_integrative_frac, bottom = labelledline_frac + all_integrative_frac + most_integrative_frac + OR_AN_MN_integrative_frac, color = 'lightsalmon')
plt.savefig('cascades/cluster_plots/Clusters_multisensory_character.pdf', format='pdf', bbox_inches='tight')

# %%
