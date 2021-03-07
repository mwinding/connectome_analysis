#%%
# have to run some code in descending_categorization.py and projectome_format.py
# some variables are borrowed from there and weren't added to this very preliminary script
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

rm = pymaid.CatmaidInstance(url, token, name, password)

mg = load_metagraph("Gad", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
#mg = load_metagraph("G", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg.calculate_degrees(inplace=True)

adj = mg.adj  # adjacency matrix from the "mg" object

#%% 

# identify left and right side for each skid category
def split_hemilateral_to_indices(skids, left, right, mg):
    intersect_left = np.intersect1d(skids, left)
    indices_left = np.where([x in intersect_left for x in mg.meta.index])[0]
    intersect_right = np.intersect1d(skids, right)
    indices_right = np.where([x in intersect_right for x in mg.meta.index])[0]

    return(indices_left, indices_right, intersect_left, intersect_right)

ORN_skids = pymaid.get_skids_by_annotation('mw ORN')
dVNC_skids = pymaid.get_skids_by_annotation('mw dVNC')

left = pymaid.get_skids_by_annotation('mw left')
right = pymaid.get_skids_by_annotation('mw right')

output_skids = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain outputs').name))
output_skids = [val for sublist in output_skids for val in sublist]
output_indices = np.where([x in output_skids for x in mg.meta.index])[0]

ORN_indices_left, ORN_indices_right, ORN_left, ORN_right = split_hemilateral_to_indices(ORN_skids, left, right, mg)
dVNC_indices_left, dVNC_indices_right, dVNC_left, dVNC_right = split_hemilateral_to_indices(dVNC_skids, left, right, mg)

#%%
from src.traverse import Cascade, to_transmission_matrix
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot
import numpy.random as random

def static_random_subset_cascade(s_indices, r_indices, subset_number, n_init, cdispatch):
# combination of a randomized subset of indices and a static subset will be used for cascade start_nodes
# make sure that input cdispath is set with n_init = 1

    hit_hist_list = []
    random_indices_list = []
    for i in range(0, n_init):
        random.seed(i)
        random_nums = random.choice(len(r_indices), subset_number, replace = False)
        random_indices = r_indices[random_nums]
        all_indices = np.concatenate([random_indices, s_indices])
        subset_hit_hist = cdispatch.multistart(start_nodes = all_indices)

        hit_hist_list.append(subset_hit_hist)
        random_indices_list.append(all_indices)

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

num_full = int(np.round(len(ORN_indices_left)))
num_95L = int(np.round(len(ORN_indices_left)*9.5/10))
num_75L = int(np.round(len(ORN_indices_left)*3/4))
num_50L = int(np.round(len(ORN_indices_left)/2))
num_25L = int(np.round(len(ORN_indices_left)/4))
num_10L = int(np.round(len(ORN_indices_left)/10))
num_5L = int(np.round(len(ORN_indices_left)/20))

fullR_fullL_hist, fullR_fullL_indices = static_random_subset_cascade(ORN_indices_right, ORN_indices_left, num_full, its, cdispatch)
fullR_95L_hist, fullR_95L_indices = static_random_subset_cascade(ORN_indices_right, ORN_indices_left, num_95L, its, cdispatch)
#fullR_75L_hist, fullR_75L_indices = static_random_subset_cascade(ORN_indices_right, ORN_indices_left, num_75L, its, cdispatch)
fullR_50L_hist, fullR_50L_indices = static_random_subset_cascade(ORN_indices_right, ORN_indices_left, num_50L, its, cdispatch)
fullR_25L_hist, fullR_25L_indices = static_random_subset_cascade(ORN_indices_right, ORN_indices_left, num_25L, its, cdispatch)
#fullR_10L_hist, fullR_10L_indices = static_random_subset_cascade(ORN_indices_right, ORN_indices_left, num_10L, its, cdispatch)
#fullR_5L_hist, fullR_5L_indices = static_random_subset_cascade(ORN_indices_right, ORN_indices_left, num_5L, its, cdispatch)

fullR_fullL_hist = pd.DataFrame(fullR_fullL_hist)
fullR_95L_hist = pd.DataFrame(fullR_95L_hist)
#fullR_75L_hist = pd.DataFrame(fullR_95L_hist)
fullR_50L_hist = pd.DataFrame(fullR_50L_hist)
fullR_25L_hist = pd.DataFrame(fullR_25L_hist)
#fullR_10L_hist = pd.DataFrame(fullR_10L_hist)


# opposite direction, stronger on left than right
fullL_50R_hist, fullL_50R_indices = static_random_subset_cascade(ORN_indices_left, ORN_indices_right, num_50L, its, cdispatch)
fullL_25R_hist, fullL_25R_indices = static_random_subset_cascade(ORN_indices_left, ORN_indices_right, num_25L, its, cdispatch)
#fullL_10R_hist, fullL_10R_indices = static_random_subset_cascade(ORN_indices_left, ORN_indices_right, num_10L, its, cdispatch)

fullL_50R_hist = pd.DataFrame(fullL_25R_hist)
fullL_25R_hist = pd.DataFrame(fullL_25R_hist)
#fullL_10R_hist = pd.DataFrame(fullL_10R_hist)

import os
os.system('say "code executed"')

# %%
# convert to cumulative histograms of visits per hop

def cml_hit_hist(hit_hist):
    new_hit_hist = np.zeros((len(hit_hist[:,0]), len(hit_hist[0, :])))

    for i in range(0, len(hit_hist[:, 0])):
        for j in range(0, len(hit_hist[0, :])):
            if(j>0):
                new_hit_hist[i, j] = hit_hist[i, j] + sum(hit_hist[i, 0:j])
            if(j==0):
                new_hit_hist[i, j] = hit_hist[i, j]

    return(new_hit_hist)

fullR_fullL_hist_cml = cml_hit_hist(fullR_fullL_hist.values)
fullR_95L_hist_cml = cml_hit_hist(fullR_95L_hist.values)
fullR_50L_hist_cml = cml_hit_hist(fullR_50L_hist.values)
fullR_25L_hist_cml = cml_hit_hist(fullR_25L_hist.values)

fullL_50R_hist_cml = cml_hit_hist(fullL_50R_hist.values)
fullL_25R_hist_cml = cml_hit_hist(fullL_25R_hist.values)


# %%
# subtracting control from experimental

fullR_95L_hist_cml_norm = fullR_95L_hist_cml - fullR_fullL_hist_cml
fullR_50L_hist_cml_norm = fullR_50L_hist_cml - fullR_fullL_hist_cml
fullR_25L_hist_cml_norm = fullR_25L_hist_cml - fullR_fullL_hist_cml

fullL_50R_hist_cml_norm = fullL_50R_hist_cml - fullR_fullL_hist_cml
fullL_25R_hist_cml_norm = fullL_25R_hist_cml - fullR_fullL_hist_cml

# %%
# look at descending neurons; more or less active?

dVNC_left_ordered = []
dVNC_indices_left_ordered = []
for i in range(0, len(pairs.leftid)):
    for j in range(0, len(dVNC_left)):
        if(dVNC_left[j] == pairs.leftid[i]):
            dVNC_left_ordered.append(dVNC_left[j])
            dVNC_indices_left_ordered.append(dVNC_indices_left[j])

dVNC_right_ordered = []
dVNC_indices_right_ordered = []
for i in range(0, len(pairs.rightid)):
    for j in range(0, len(dVNC_right)):
        if(dVNC_right[j] == pairs.rightid[i]):
            dVNC_right_ordered.append(dVNC_right[j])
            dVNC_indices_right_ordered.append(dVNC_indices_right[j])

diff_pairs_fullR_25L = fullR_25L_hist_cml_norm[dVNC_indices_right_ordered, :]-fullR_25L_hist_cml_norm[dVNC_indices_left_ordered, :]
diff_pairs_fullR_25L = pd.DataFrame(diff_pairs_fullR_25L, index = dVNC_left_ordered)

# looks somewhat promising
# perhaps it would be useful to propagate the signal to VNC segments

dVNC_proj_left = projectome.loc[dVNC_left_ordered, :]
dVNC_proj_left = dVNC_proj_left.iloc[:, (len(dVNC_proj_left.iloc[0,:])-22):len(dVNC_proj_left.iloc[0,:])]

dVNC_proj_right = projectome.loc[dVNC_right_ordered, :]
dVNC_proj_right = dVNC_proj_right.iloc[:, (len(dVNC_proj_right.iloc[0,:])-22):len(dVNC_proj_right.iloc[0,:])]

# simply multiple hits by outputs to each segment for quick and dirty test
left_dVNC_hits = fullR_25L_hist_cml[dVNC_indices_left_ordered]
dVNC_project_left_hops = []
for i in range(0, len(left_dVNC_hits[0, :])):
    dVNC_project_left = []
    for j in range(0, len(left_dVNC_hits[:, 0])):
        dVNC_project_left.append(dVNC_proj_left.iloc[j, :] * left_dVNC_hits[j, i])

    dVNC_project_left = pd.DataFrame(dVNC_project_left).sum(axis = 0)
    dVNC_project_left_hops.append(dVNC_project_left)

right_dVNC_hits = fullR_25L_hist_cml[dVNC_indices_right_ordered]
dVNC_project_right_hops = []
for i in range(0, len(right_dVNC_hits[0, :])):
    dVNC_project_right = []
    for j in range(0, len(right_dVNC_hits[:, 0])):
        dVNC_project_right.append(dVNC_proj_right.iloc[j, :] * right_dVNC_hits[j, i])

    dVNC_project_right = pd.DataFrame(dVNC_project_right).sum(axis = 0)
    dVNC_project_right_hops.append(dVNC_project_right)


left_dVNC_hits = fullR_fullL_hist_cml[dVNC_indices_left_ordered]
dVNC_project_left_hops_control = []
for i in range(0, len(left_dVNC_hits[0, :])):
    dVNC_project_left = []
    for j in range(0, len(left_dVNC_hits[:, 0])):
        dVNC_project_left.append(dVNC_proj_left.iloc[j, :] * left_dVNC_hits[j, i])

    dVNC_project_left = pd.DataFrame(dVNC_project_left).sum(axis = 0)
    dVNC_project_left_hops_control.append(dVNC_project_left)

right_dVNC_hits = fullR_fullL_hist_cml[dVNC_indices_right_ordered]
dVNC_project_right_hops_control = []
for i in range(0, len(right_dVNC_hits[0, :])):
    dVNC_project_right = []
    for j in range(0, len(right_dVNC_hits[:, 0])):
        dVNC_project_right.append(dVNC_proj_right.iloc[j, :] * right_dVNC_hits[j, i])

    dVNC_project_right = pd.DataFrame(dVNC_project_right).sum(axis = 0)
    dVNC_project_right_hops_control.append(dVNC_project_right)

# %%
#

test = dVNC_project_right_hops[2] + dVNC_project_left_hops[2]
ratio = []
for i in np.arange(0, len(test), 2):
    ratio.append((test[i+1] - test[i])/((test[i+1]+test[i])/2))

test_control = dVNC_project_right_hops_control[2] + dVNC_project_left_hops_control[2]
ratio_control = []
for i in np.arange(0, len(test), 2):
    ratio_control.append((test_control[i+1] - test_control[i])/((test_control[i+1]+test_control[i])/2))


test = dVNC_project_right_hops[3] + dVNC_project_left_hops[3]
ratio = []
for i in np.arange(0, len(test), 2):
    ratio.append((test[i+1] - test[i])/((test[i+1]+test[i])/2))

test_control = dVNC_project_right_hops_control[3] + dVNC_project_left_hops_control[3]
ratio_control = []
for i in np.arange(0, len(test), 2):
    ratio_control.append((test_control[i+1] - test_control[i])/((test_control[i+1]+test_control[i])/2))


test4 = dVNC_project_right_hops[4] + dVNC_project_left_hops[4]
ratio = []
for i in np.arange(0, len(test), 2):
    ratio.append((test[i+1] - test[i])/((test[i+1]+test[i])/2))

test_control4 = dVNC_project_right_hops_control[4] + dVNC_project_left_hops_control[4]
ratio_control = []
for i in np.arange(0, len(test), 2):
    ratio_control.append((test_control[i+1] - test_control[i])/((test_control[i+1]+test_control[i])/2))

# %%
# comparing signal between descending partners; any asymmetry?
# nothing super convincing

import connectome_tools.process_matrix as promat

pairs = pd.read_csv('data/pairs-2020-05-08.csv', header = 0)

def compare_pairs_hops(skids_left, skids_right, indices_left, indices_right, pairList, hit_hist_list, right = True):
    
    contra = pymaid.get_skids_by_annotation('mw brain contralateral')
    skids = np.concatenate([skids_left, skids_right])
    diff_mat = []
    left_mat = []
    right_mat = []
    pair_leftid = []
    for i in skids:
        if(int(i) in pairList["leftid"].values):
            pair = promat.get_paired_skids(int(i), pairList)
            left_index = indices_left[skids_left==pair[0]][0]
            right_index = indices_right[skids_right==pair[1]][0]

            left_hits = hit_hist_list.iloc[left_index, :]
            right_hits = hit_hist_list.iloc[right_index, :]
            if(right == True): # depending on one's perspective; makes right bias positive
                pair_diff = right_hits - left_hits
                if(pair[0] in contra):
                    pair_diff = -pair_diff # switch direction if contralateral descending

            if(right == False): # depending on one's perspective; makes left bias positive
                pair_diff = left_hits - right_hits
                if(pair[0] in contra):
                    pair_diff = -pair_diff # switch direction if contralateral descending
                    
            diff_mat.append(pair_diff)
            left_mat.append(left_hits)
            right_mat.append(right_hits)
            pair_leftid.append(pair[0])

    return(pd.DataFrame(left_mat, index = pair_leftid),
            pd.DataFrame(right_mat, index = pair_leftid),
            pd.DataFrame(diff_mat, index = pair_leftid))

fullR_25L_hist_left, fullR_25L_hist_right, fullR_25L_hist_pairDiff = compare_pairs_hops(dVNC_left, dVNC_right, dVNC_indices_left, dVNC_indices_right, pairs, pd.DataFrame(fullR_25L_hist_cml))
fullL_25R_hist_left, fullL_25R_hist_right, fullL_25R_hist_pairDiff = compare_pairs_hops(dVNC_left, dVNC_right, dVNC_indices_left, dVNC_indices_right, pairs, pd.DataFrame(fullL_25R_hist_cml), right = True)

# %%
#
sns.heatmap(dVNC_projectome_pairs.loc[fullR_25L_hist_pairDiff.sort_values(by = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ascending = False).index].iloc[:, 1:len(dVNC_projectome_pairs)])

# %%
#
'''
# comparison between pairs

import connectome_tools.process_matrix as promat

pairs = pd.read_csv('data/pairs-2020-05-08.csv', header = 0)

def compare_pairs(skids_left, skids_right, indices_left, indices_right, pairList, hit_hist_list, right = True):
    
    skids = np.concatenate([skids_left, skids_right])
    pairs = []
    for i in skids:
        if(int(i) in pairList["leftid"].values):
            pair = promat.get_paired_skids(int(i), pairList)
            left_index = indices_left[skids_left==pair[0]][0]
            right_index = indices_right[skids_right==pair[1]][0]

            left_hits = hit_hist_list.iloc[left_index, :].sum(axis=0)
            right_hits = hit_hist_list.iloc[right_index, :].sum(axis=0)
            if(right == True): # depending on one's perspective; makes right bias positive
                diff = right_hits - left_hits
                percent_diff = ((right_hits - left_hits)/((right_hits + left_hits)/2))*100
            if(right == False): # depending on one's perspective; makes left bias positive
                diff = left_hits - right_hits
                percent_diff = ((left_hits - right_hits)/((right_hits + left_hits)/2))*100

            pairs.append({'leftid': pair[0], 'rightid': pair[1], 
                        'left_index': left_index, 'right_index': right_index, 
                        'left_hits': left_hits, 'right_hits': right_hits,
                        'diff': diff, 'percent_diff': percent_diff})

    return(pd.DataFrame(pairs))

fullR_95L_hist_pairs = compare_pairs(dVNC_left, dVNC_right, dVNC_indices_left, dVNC_indices_right, pairs, pd.DataFrame(fullR_95L_hist_cml_norm))
fullR_50L_hist_pairs = compare_pairs(dVNC_left, dVNC_right, dVNC_indices_left, dVNC_indices_right, pairs, pd.DataFrame(fullR_50L_hist_cml_norm))
fullR_25L_hist_pairs = compare_pairs(dVNC_left, dVNC_right, dVNC_indices_left, dVNC_indices_right, pairs, pd.DataFrame(fullR_25L_hist_cml_norm))

fullL_50R_hist_pairs = compare_pairs(dVNC_left, dVNC_right, dVNC_indices_left, dVNC_indices_right, pairs, pd.DataFrame(fullL_25R_hist_cml_norm))
fullL_25R_hist_pairs = compare_pairs(dVNC_left, dVNC_right, dVNC_indices_left, dVNC_indices_right, pairs, pd.DataFrame(fullL_50R_hist_cml_norm))

sns.heatmap(fullR_25L_hist_pairs.sort_values(by = 'diff', ascending = False).iloc[:,4:6])
sns.heatmap(fullR_50L_hist_pairs.sort_values(by = 'diff', ascending = False).iloc[:,4:6])

fig, axs = plt.subplots(
    3, 1, figsize=(6, 10)
)
threshold = 250
fig.tight_layout(pad=3.0)

test1 = fullR_25L_hist_pairs.sort_values(by = 'diff', ascending = False)

ax = axs[0]
ax.set_title('Descending neurons more active on right by >%i' %threshold)
sns.heatmap(dVNC_projectome_pairs.loc[test1.leftid[test1['diff']>threshold], ['T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']], ax = ax, rasterized = True)

ax = axs[1]
ax.set_title('Descending neurons more active on left by >%i' %threshold)
sns.heatmap(dVNC_projectome_pairs.loc[test1.leftid[test1['diff']<-threshold], ['T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']], ax = ax, rasterized = True)

ax = axs[2]
ax.axis("off")
caption = f"Figure: Output Projectome for Descending Neurons,\n downstream of ORN Cascade 100% right signal + 25% ORN left signal\n\n"
ax.text(0, 1, caption, va="top")

#plt.savefig('cascades/interhemisphere_plots/100_vs_25_ORNsignal_descendings.pdf', bbox_inches='tight')


fig, axs = plt.subplots(
    3, 1, figsize=(6, 10)
)
threshold = 150
fig.tight_layout(pad=3.0)

test1 = fullL_25R_hist_pairs.sort_values(by = 'diff', ascending = False)

ax = axs[0]
ax.set_title('Descending neurons more active on right by >%i' %threshold)
sns.heatmap(dVNC_projectome_pairs.loc[test1.leftid[test1['diff']>threshold], ['T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']], ax = ax, rasterized = True)

ax = axs[1]
ax.set_title('Descending neurons more active on left by >%i' %threshold)
sns.heatmap(dVNC_projectome_pairs.loc[test1.leftid[test1['diff']<-threshold], ['T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']], ax = ax, rasterized = True)

ax = axs[2]
ax.axis("off")
caption = f"Figure: Output Projectome for Descending Neurons,\n downstream of ORN Cascade 100% right signal + 25% ORN left signal\n\n"
ax.text(0, 1, caption, va="top")

#plt.savefig('cascades
'''
