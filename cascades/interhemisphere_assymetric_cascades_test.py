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
#mg = load_metagraph("G", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg.calculate_degrees(inplace=True)

adj = mg.adj  # adjacency matrix from the "mg" object

#%%
# pull ORN skids and then divide into left and right hemisphere
ORN_skids = pymaid.get_skids_by_annotation('mw ORN')
AN_skids = pymaid.get_skids_by_annotation('mw AN sensories')
MN_skids = pymaid.get_skids_by_annotation('mw MN sensories')
A00c_skids = pymaid.get_skids_by_annotation('mw A00c')
vtd_skids = pymaid.get_skids_by_annotation('mw v\'td')
thermo_skids = pymaid.get_skids_by_annotation('mw thermosensories')
photo_skids = pymaid.get_skids_by_annotation('mw photoreceptors')
left = pymaid.get_skids_by_annotation('mw left')
right = pymaid.get_skids_by_annotation('mw right')

output_skids = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain outputs').name))
output_skids = [val for sublist in output_skids for val in sublist]

input_skids = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain inputs').name))
input_skids = [val for sublist in input_skids for val in sublist]

# identify left and right side for each skid category
def split_hemilateral_to_indices(skids, left, right, mg):
    intersect_left = np.intersect1d(skids, left)
    indices_left = np.where([x in intersect_left for x in mg.meta.index])[0]
    intersect_right = np.intersect1d(skids, right)
    indices_right = np.where([x in intersect_right for x in mg.meta.index])[0]

    return(indices_left, indices_right, intersect_left, intersect_right)

ORN_indices_left, ORN_indices_right, ORN_left, ORN_right = split_hemilateral_to_indices(ORN_skids, left, right, mg)
AN_indices_left, AN_indices_right, AN_left, AN_right = split_hemilateral_to_indices(AN_skids, left, right, mg)
MN_indices_left, MN_indices_right, MN_left, MN_right = split_hemilateral_to_indices(MN_skids, left, right, mg)
A00c_indices_left, A00c_indices_right, A00c_left, A00c_right = split_hemilateral_to_indices(A00c_skids, left, right, mg)
vtd_indices_left, vtd_indices_right, vtd_left, vtd_right = split_hemilateral_to_indices(vtd_skids, left, right, mg)
thermo_indices_left, thermo_indices_right, thermo_left, thermo_right = split_hemilateral_to_indices(thermo_skids, left, right, mg)
photo_indices_left, photo_indices_right, photo_left, photo_right = split_hemilateral_to_indices(photo_skids, left, right, mg)

input_indices_left, input_indices_right, input_left, input_right = split_hemilateral_to_indices(input_skids, left, right, mg)
output_indices = np.where([x in output_skids for x in mg.meta.index])[0]

# add A00c left to inputs right and A00c right to inputs left
    # this is because A00cs are ascending contralateral inputs

# identify appropriate indices for A00c neurons
input_indices_left_A00c_index = np.where([x in A00c_indices_left for x in input_indices_left])[0]
input_indices_right_A00c_index = np.where([x in A00c_indices_right for x in input_indices_right])[0]
input_left_A00c_index = np.where([x in A00c_left for x in input_left])[0]
input_right_A00c_index = np.where([x in A00c_right for x in input_right])[0]

# delete A00c skids/indices from each np.array
input_indices_left = np.delete(input_indices_left, input_indices_left_A00c_index)
input_indices_right = np.delete(input_indices_right, input_indices_right_A00c_index)
input_left = np.delete(input_left, input_left_A00c_index)
input_right = np.delete(input_right, input_right_A00c_index)

# add appropriate A00c skids/indices
input_indices_left = np.append(input_indices_left, A00c_indices_right)
input_indices_right = np.append(input_indices_right, A00c_indices_left)
input_left = np.append(input_left, A00c_right)
input_right = np.append(input_right, A00c_left)

# %%
from src.traverse import Cascade, to_transmission_matrix
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot
import numpy.random as random

def random_subset_cascade(indices, subset_number, n_init, cdispatch):
# make sure that input cdispath is set with n_init = 1

    hit_hist_list = []
    random_indices_list = []
    for i in range(0, n_init):
        random.seed(i)
        random_nums = random.choice(len(indices), subset_number, replace = False)
        random_indices = indices[random_nums]
        subset_hit_hist = cdispatch.multistart(start_nodes = random_indices)

        hit_hist_list.append(subset_hit_hist)
        random_indices_list.append(random_indices)

    return(sum(hit_hist_list), random_indices_list)

p = 0.05
max_hops = 10
n_init = 1
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

input_indices_left = ORN_indices_left
its = 1000

num_full_left = int(np.round(len(input_indices_left)))
num_three_quarter_left = int(np.round(len(input_indices_left)*3/4))
num_half_left = int(np.round(len(input_indices_left)/2))
num_quarter_left = int(np.round(len(input_indices_left)/4))
num_tenth_left = int(np.round(len(input_indices_left)/10))
num_twentieth_left = int(np.round(len(input_indices_left)/20))

input_full_left_hist, input_full_random_indices_left = random_subset_cascade(input_indices_left, num_full_left, its, cdispatch)
input_3quarter_left_hist, input_3quarter_random_indices_left = random_subset_cascade(input_indices_left, num_three_quarter_left, its, cdispatch)
input_half_left_hist, input_half_random_indices_left = random_subset_cascade(input_indices_left, num_half_left, its, cdispatch)
input_quarter_left_hist, input_quarter_random_indices_left = random_subset_cascade(input_indices_left, num_quarter_left, its, cdispatch)
input_tenth_left_hist, input_tenth_random_indices_left = random_subset_cascade(input_indices_left, num_tenth_left, its, cdispatch)
input_twentieth_left_hist, input_twentieth_random_indices_left = random_subset_cascade(input_indices_left, num_twentieth_left, its, cdispatch)

import os
os.system('say "code executed"')
# %%
# identifying ipsilateral and contralateral neurons

ipsi = pymaid.get_skids_by_annotation('mw brain ipsilateral')
contra = pymaid.get_skids_by_annotation('mw brain contralateral')

ipsi_indices_left, ipsi_indices_right, ipsi_left, ipsi_right = split_hemilateral_to_indices(ipsi, left, right, mg)
contra_indices_left, contra_indices_right, contra_left, contra_right = split_hemilateral_to_indices(contra, left, right, mg)

ipsi_indices_left = np.concatenate((ORN_indices_left, AN_indices_left,
                                    MN_indices_left, vtd_indices_left,
                                    thermo_indices_left, photo_indices_left,
                                    ipsi_indices_left), axis = 0)

ipsi_indices_right = np.concatenate((ORN_indices_right, AN_indices_right,
                                    MN_indices_right, vtd_indices_right,
                                    thermo_indices_right, photo_indices_right,
                                    ipsi_indices_right), axis = 0)

contra_indices_left = np.concatenate((A00c_indices_left,
                                    contra_indices_left), axis = 0)

contra_indices_right = np.concatenate((A00c_indices_right,
                                    contra_indices_right), axis = 0)

# %%
# total signal through ipsilateral and contralateral neurons

fig, axs = plt.subplots(
    6, 1, figsize=(6, 20)
)

threshold = 50
fig.tight_layout(pad=2.5)

full_left = pd.DataFrame([(input_full_left_hist[ipsi_indices_left].sum(axis = 0)), 
                            (input_full_left_hist[contra_indices_left].sum(axis = 0)),
                            (input_full_left_hist[contra_indices_right].sum(axis = 0)), 
                            (input_full_left_hist[ipsi_indices_right].sum(axis = 0))], 
                            index = ['Ipsilateral same side', 'Contralateral same side', 
                                    'Contralateral opposite side', 'Ipsilateral opposite side'])

threequarter_left = pd.DataFrame([(input_3quarter_left_hist[ipsi_indices_left].sum(axis = 0)), 
                            (input_3quarter_left_hist[contra_indices_left].sum(axis = 0)),
                            (input_3quarter_left_hist[contra_indices_right].sum(axis = 0)), 
                            (input_3quarter_left_hist[ipsi_indices_right].sum(axis = 0))], 
                            index = ['Ipsilateral same side', 'Contralateral same side', 
                                    'Contralateral opposite side', 'Ipsilateral opposite side'])

half_left = pd.DataFrame([(input_half_left_hist[ipsi_indices_left].sum(axis = 0)), 
                            (input_half_left_hist[contra_indices_left].sum(axis = 0)),
                            (input_half_left_hist[contra_indices_right].sum(axis = 0)), 
                            (input_half_left_hist[ipsi_indices_right].sum(axis = 0))], 
                            index = ['Ipsilateral same side', 'Contralateral same side', 
                                    'Contralateral opposite side', 'Ipsilateral opposite side'])

quarter_left = pd.DataFrame([(input_quarter_left_hist[ipsi_indices_left].sum(axis = 0)), 
                            (input_quarter_left_hist[contra_indices_left].sum(axis = 0)),
                            (input_quarter_left_hist[contra_indices_right].sum(axis = 0)), 
                            (input_quarter_left_hist[ipsi_indices_right].sum(axis = 0))], 
                            index = ['Ipsilateral same side', 'Contralateral same side', 
                                    'Contralateral opposite side', 'Ipsilateral opposite side'])

tenth_left = pd.DataFrame([(input_tenth_left_hist[ipsi_indices_left].sum(axis = 0)), 
                            (input_tenth_left_hist[contra_indices_left].sum(axis = 0)),
                            (input_tenth_left_hist[contra_indices_right].sum(axis = 0)), 
                            (input_tenth_left_hist[ipsi_indices_right].sum(axis = 0))], 
                            index = ['Ipsilateral same side', 'Contralateral same side', 
                                    'Contralateral opposite side', 'Ipsilateral opposite side'])

twentieth_left = pd.DataFrame([(input_twentieth_left_hist[ipsi_indices_left].sum(axis = 0)), 
                            (input_twentieth_left_hist[contra_indices_left].sum(axis = 0)),
                            (input_twentieth_left_hist[contra_indices_right].sum(axis = 0)), 
                            (input_twentieth_left_hist[ipsi_indices_right].sum(axis = 0))], 
                            index = ['Ipsilateral same side', 'Contralateral same side', 
                                    'Contralateral opposite side', 'Ipsilateral opposite side'])

full_left = full_left.fillna(0)
threequarter_left = threequarter_left.fillna(0)
half_left = half_left.fillna(0)
quarter_left = quarter_left.fillna(0)
tenth_left = tenth_left.fillna(0)
twentieth_left = twentieth_left.fillna(0)

vmax = np.amax(full_left.values)

ax = axs[0]
ax.set_title('Left input signal (Random selection all inputs)')
sns.heatmap(full_left, ax = ax, rasterized = True, cbar = True, vmax = vmax)

ax = axs[1]
ax.set_title('Left input signal (Random selection 0.75 of inputs)')
sns.heatmap(threequarter_left, ax = ax, rasterized = True, cbar = True, vmax = vmax)

ax = axs[2]
ax.set_title('Left input signal (Random selection 0.5 of inputs)')
sns.heatmap(half_left, ax = ax, rasterized = True, cbar = True, vmax = vmax)

ax = axs[3]
ax.set_title('Left input signal (Random selection 0.25 of inputs)')
sns.heatmap(quarter_left, ax = ax, rasterized = True, cbar = True, vmax = vmax)

ax = axs[4]
ax.set_title('Left input signal (Random selection 0.10 of inputs)')
sns.heatmap(tenth_left, ax = ax, rasterized = True, cbar = True, vmax = vmax)

ax = axs[5]
ax.set_title('Left input signal (Random selection 0.05 of inputs)')
sns.heatmap(twentieth_left, ax = ax, rasterized = True, cbar = True, vmax = vmax)

fig.savefig('cascades/interhemisphere_plots/assymetric_input_test.pdf', format='pdf', bbox_inches='tight')


# %%
# neuron of neurons per hop with >50 visits
list_hists = [input_full_left_hist, input_3quarter_left_hist, input_half_left_hist, 
                input_quarter_left_hist, input_tenth_left_hist, input_twentieth_left_hist]

xlabel_list = ['100%', '75%', '50%', '25%', '10%', '5%']

fig, axs = plt.subplots(
    len(list_hists), 1, figsize=(6, 20)
)

threshold = 500
vmax = 120
fig.tight_layout(pad=2.5)

for i in np.arange(0, len(list_hists)):
    df = pd.DataFrame([(list_hists[i][ipsi_indices_left]>threshold).sum(axis = 0),
                    (list_hists[i][contra_indices_left]>threshold).sum(axis = 0),
                    (list_hists[i][contra_indices_right]>threshold).sum(axis = 0),
                    (list_hists[i][ipsi_indices_right]>threshold).sum(axis = 0)],
                    index = ['Ipsilateral same side', 'Contralateral same side', 
                            'Contralateral opposite side', 'Ipsilateral opposite side']
                    )

    df = df.fillna(0)
    ax = axs[i]
    ax.set_title('Left input signal (Random selection of %s neurons)' %(xlabel_list[i]))
    sns.heatmap(df, ax = ax, rasterized = True, cbar = True, vmax = vmax, cbar_kws={'label': 'Number Neurons >%i Visits' %(threshold)})

fig.savefig('cascades/interhemisphere_plots/assymetric_input_test.pdf', format='pdf', bbox_inches='tight')

# %%
# comparing particular nodes between conditions, across different # hops

fig, axs = plt.subplots(
    5, 6, figsize=(15, 10)
)

fig.tight_layout(pad=2.5)
max_hops = 6

for hop in range(0, max_hops):

    ax = axs[0, hop]
    ax.set(xlim = [0, 1000], ylim = [0, 1000])
    sns.scatterplot(x = input_full_left_hist[:,hop], y = input_3quarter_left_hist[:,hop], ax = ax,  edgecolor = "none", alpha = 0.25)
    sns.lineplot(x = np.arange(0, 1000, 1), y = np.arange(0, 1000, 1), dashes = ([2, 2]), ax = ax, color='gray')

    ax = axs[1, hop]
    ax.set(xlim = [0, 1000], ylim = [0, 1000])
    sns.scatterplot(x = input_full_left_hist[:,hop], y = input_half_left_hist[:,hop], ax = ax,  edgecolor = "none", alpha = 0.25)
    sns.lineplot(x = np.arange(0, 1000, 1), y = np.arange(0, 1000, 1), dashes = ([2, 2]), ax = ax, color='gray')

    ax = axs[2, hop]
    ax.set(xlim = [0, 1000], ylim = [0, 1000])
    sns.scatterplot(x = input_full_left_hist[:,hop], y = input_quarter_left_hist[:,hop], ax = ax,  edgecolor = "none", alpha = 0.25)
    sns.lineplot(x = np.arange(0, 1000, 1), y = np.arange(0, 1000, 1), dashes = ([2, 2]), ax = ax, color='gray')

    ax = axs[3, hop]
    ax.set(xlim = [0, 1000], ylim = [0, 1000])
    sns.scatterplot(x = input_full_left_hist[:,hop], y = input_tenth_left_hist[:,hop], ax = ax,  edgecolor = "none", alpha = 0.25)
    sns.lineplot(x = np.arange(0, 1000, 1), y = np.arange(0, 1000, 1), dashes = ([2, 2]), ax = ax, color='gray')

    ax = axs[4, hop]
    ax.set(xlim = [0, 1000], ylim = [0, 1000])
    sns.scatterplot(x = input_full_left_hist[:,hop], y = input_twentieth_left_hist[:,hop], ax = ax,  edgecolor = "none", alpha = 0.25)
    sns.lineplot(x = np.arange(0, 1000, 1), y = np.arange(0, 1000, 1), dashes = ([2, 2]), ax = ax, color='gray')

fig.savefig('cascades/interhemisphere_plots/assymetric_input_test_pernode_perhop.pdf', format='pdf', bbox_inches='tight')

# %%
# comparing particular nodes between conditions

fig, axs = plt.subplots(
    5, 1, figsize=(4, 14)
)

fig.tight_layout(pad=2.5)
lim_min = 0
lim_max = 1000


ax = axs[0]
ax.set(xlim = [lim_min,lim_max], ylim = [lim_min, lim_max])
ax.set_title('100 left signal vs 75 left signal')
ax.set_xlabel('Visits per node')
sns.scatterplot(x = input_full_left_hist[:,1:5].sum(axis = 1), y = input_3quarter_left_hist[:,1:5].sum(axis = 1), ax = ax, edgecolor = "none", alpha = 0.25)
sns.lineplot(x = np.arange(0, 1000, 1), y = np.arange(0, 1000, 1), dashes = ([2, 2]), ax = ax, color='gray')

ax = axs[1]
ax.set(xlim = [lim_min,lim_max], ylim = [lim_min, lim_max])
ax.set_title('100 left signal vs 50 left signal')
sns.scatterplot(x = input_full_left_hist[:,1:5].sum(axis = 1), y = input_half_left_hist[:,1:5].sum(axis = 1), ax = ax, edgecolor = "none", alpha = 0.25)
sns.lineplot(x = np.arange(0, 1000, 1), y = np.arange(0, 1000, 1), dashes = ([2, 2]), ax = ax, color='gray')

ax = axs[2]
ax.set(xlim = [lim_min,lim_max], ylim = [lim_min, lim_max])
ax.set_title('100 left signal vs 25 left signal')
sns.scatterplot(x = input_full_left_hist[:,1:5].sum(axis = 1), y = input_quarter_left_hist[:,1:5].sum(axis = 1), ax = ax, edgecolor = "none", alpha = 0.25)
sns.lineplot(x = np.arange(0, 1000, 1), y = np.arange(0, 1000, 1), dashes = ([2, 2]), ax = ax, color='gray')

ax = axs[3]
ax.set(xlim = [lim_min,lim_max], ylim = [lim_min, lim_max])
ax.set_title('100 left signal vs 10 left signal')
sns.scatterplot(x = input_full_left_hist[:,1:5].sum(axis = 1), y = input_tenth_left_hist[:,1:5].sum(axis = 1), ax = ax, edgecolor = "none", alpha = 0.25)
sns.lineplot(x = np.arange(0, 1000, 1), y = np.arange(0, 1000, 1), dashes = ([2, 2]), ax = ax, color='gray')

ax = axs[4]
ax.set(xlim = [lim_min,lim_max], ylim = [lim_min, lim_max])
ax.set_title('100 left signal vs 5 left signal')
sns.scatterplot(x = input_full_left_hist[:,1:5].sum(axis = 1), y = input_twentieth_left_hist[:,1:5].sum(axis = 1), ax = ax, edgecolor = "none", alpha = 0.25)
sns.lineplot(x = np.arange(0, 1000, 1), y = np.arange(0, 1000, 1), dashes = ([2, 2]), ax = ax, color='gray')

fig.savefig('cascades/interhemisphere_plots/assymetric_input_test_pernode.pdf', format='pdf', bbox_inches='tight')

# %%
# distributions of visits per hop

df = pd.DataFrame(input_full_left_hist, columns = ['hop0', 'hop1', 'hop2', 'hop3', 'hop4', 'hop5', 'hop6', 'hop7', 'hop8', 'hop9'])

df = pd.DataFrame(columns = ['visits', 'hops'])

data_full = []
data_10 = []
data_5 = []

for i in np.arange(0, len(input_full_left_hist)):
    for j in np.arange(0, len(input_full_left_hist[i,])):
        data_full.append([input_full_left_hist[i, j], j])
        data_10.append([input_tenth_left_hist[i, j], j])
        data_5.append([input_twentieth_left_hist[i, j], j])

df_full = pd.DataFrame(data_full , columns = ['visits', 'hops'])
df_10 = pd.DataFrame(data_10 , columns = ['visits', 'hops'])
df_5 = pd.DataFrame(data_5 , columns = ['visits', 'hops'])

sns.violinplot(data = df_full[df_full.visits>0], x = 'hops', y = 'visits')
sns.violinplot(data = df_10[df_10.visits>0], x = 'hops', y = 'visits')
sns.violinplot(data = df_5[df_5.visits>0], x = 'hops', y = 'visits')

#df_full[df_full.visits>0].groupby('hops').mean()
#df_full[df_full.visits>0].groupby('hops').median()

#df_5[df_5.visits>0].groupby('hops').mean()
#df_5[df_5.visits>0].groupby('hops').median()


        