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
fullR_75L_hist, fullR_75L_indices = static_random_subset_cascade(ORN_indices_right, ORN_indices_left, num_75L, its, cdispatch)
fullR_50L_hist, fullR_50L_indices = static_random_subset_cascade(ORN_indices_right, ORN_indices_left, num_50L, its, cdispatch)
fullR_25L_hist, fullR_25L_indices = static_random_subset_cascade(ORN_indices_right, ORN_indices_left, num_25L, its, cdispatch)
fullR_10L_hist, fullR_10L_indices = static_random_subset_cascade(ORN_indices_right, ORN_indices_left, num_10L, its, cdispatch)
fullR_5L_hist, fullR_5L_indices = static_random_subset_cascade(ORN_indices_right, ORN_indices_left, num_5L, its, cdispatch)

fullR_fullL_hist = pd.DataFrame(fullR_fullL_hist)
fullR_95L_hist = pd.DataFrame(fullR_95L_hist)
fullR_75L_hist = pd.DataFrame(fullR_95L_hist)
fullR_50L_hist = pd.DataFrame(fullR_50L_hist)
fullR_25L_hist = pd.DataFrame(fullR_25L_hist)
fullR_10L_hist = pd.DataFrame(fullR_10L_hist)


# opposite direction, stronger on left than right
fullL_50R_hist, fullL_50R_indices = static_random_subset_cascade(ORN_indices_left, ORN_indices_right, num_50L, its, cdispatch)
fullL_25R_hist, fullL_25R_indices = static_random_subset_cascade(ORN_indices_left, ORN_indices_right, num_25L, its, cdispatch)
fullL_10R_hist, fullL_10R_indices = static_random_subset_cascade(ORN_indices_left, ORN_indices_right, num_10L, its, cdispatch)

fullL_50R_hist = pd.DataFrame(fullL_25R_hist)
fullL_25R_hist = pd.DataFrame(fullL_25R_hist)
fullL_10R_hist = pd.DataFrame(fullL_10R_hist)


import os
os.system('say "code executed"')

# %%
# initial plots

fig, axs = plt.subplots(
    3, 1, figsize=(6, 20)
)

fig.tight_layout(pad=2.5)
ax = axs[0]
sns.heatmap(fullR_fullL_hist[dVNC_indices_left], ax = ax)

ax = axs[1]
sns.heatmap(fullR_25L_hist[dVNC_indices_left], ax = ax)

ax = axs[2]
sns.heatmap((fullR_fullL_hist.iloc[dVNC_indices_left, :] - fullR_25L_hist.iloc[dVNC_indices_left, :]), ax = ax)

#fig.savefig('cascades/interhemisphere_plots/assymetric_input_test.pdf', format='pdf', bbox_inches='tight')

sns.clustermap((fullR_fullL_hist[dVNC_indices_left] - fullR_25L_hist[dVNC_indices_left]), col_cluster = False)
sns.clustermap((fullR_fullL_hist[dVNC_indices_right] - fullR_25L_hist[dVNC_indices_right]), col_cluster = False)

# seems that descending neurons are hit earlier in strong signal than in weak signal
# %%
import connectome_tools.process_matrix as promat

# comparing signal between descending partners; any asymmetry?
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

fullR_fullL_hist_pairs = compare_pairs(dVNC_left, dVNC_right, dVNC_indices_left, dVNC_indices_right, pairs, fullR_fullL_hist)
fullR_95L_hist_pairs = compare_pairs(dVNC_left, dVNC_right, dVNC_indices_left, dVNC_indices_right, pairs, fullR_95L_hist)
fullR_75L_hist_pairs = compare_pairs(dVNC_left, dVNC_right, dVNC_indices_left, dVNC_indices_right, pairs, fullR_75L_hist)
fullR_50L_hist_pairs = compare_pairs(dVNC_left, dVNC_right, dVNC_indices_left, dVNC_indices_right, pairs, fullR_50L_hist)
fullR_25L_hist_pairs = compare_pairs(dVNC_left, dVNC_right, dVNC_indices_left, dVNC_indices_right, pairs, fullR_25L_hist)
fullR_10L_hist_pairs = compare_pairs(dVNC_left, dVNC_right, dVNC_indices_left, dVNC_indices_right, pairs, fullR_10L_hist)

control = fullR_95L_hist_pairs
test = fullR_75L_hist_pairs
test0 = fullR_50L_hist_pairs
test1 = fullR_25L_hist_pairs
test2 = fullR_10L_hist_pairs

control.iloc[:, 4:len(control)] = (control.iloc[:, 4:len(control)] - fullR_fullL_hist_pairs.iloc[:, 4:len(control)])
test.iloc[:, 4:len(test)] = (test.iloc[:, 4:len(test)] - fullR_fullL_hist_pairs.iloc[:, 4:len(test)])
test0.iloc[:, 4:len(test0)] = (test0.iloc[:, 4:len(test0)] - fullR_fullL_hist_pairs.iloc[:, 4:len(test0)])
test1.iloc[:, 4:len(test1)] = (test1.iloc[:, 4:len(test1)] - fullR_fullL_hist_pairs.iloc[:, 4:len(test1)])
test2.iloc[:, 4:len(test2)] = (test2.iloc[:, 4:len(test2)] - fullR_fullL_hist_pairs.iloc[:, 4:len(test2)])

control = control.sort_values(by = 'diff', ascending = False)
test = test.sort_values(by = 'diff', ascending = False)
test0 = test0.sort_values(by = 'diff', ascending = False)
test1 = test1.sort_values(by = 'diff', ascending = False)
test2 = test2.sort_values(by = 'diff', ascending = False)


# opposite side
fullL_fullR_hist_pairs = compare_pairs(dVNC_left, dVNC_right, dVNC_indices_left, dVNC_indices_right, pairs, fullR_fullL_hist, right = False)
fullL_25R_hist_pairs = compare_pairs(dVNC_left, dVNC_right, dVNC_indices_left, dVNC_indices_right, pairs, fullL_25R_hist, right = False)
fullL_10R_hist_pairs = compare_pairs(dVNC_left, dVNC_right, dVNC_indices_left, dVNC_indices_right, pairs, fullL_10R_hist, right = False)

test1R = fullL_25R_hist_pairs
test2R = fullL_10R_hist_pairs
test1R.iloc[:, 4:len(test1R)] = (test1R.iloc[:, 4:len(test1R)] - fullL_fullR_hist_pairs.iloc[:, 4:len(test1R)])
test2R.iloc[:, 4:len(test2R)] = (test2R.iloc[:, 4:len(test2R)] - fullL_fullR_hist_pairs.iloc[:, 4:len(test2R)])
test1R = test1R.sort_values(by = 'diff', ascending = False)
test2R = test2R.sort_values(by = 'diff', ascending = False)

# %%
# plotting difference in hit numbers left vs right
fig, axs = plt.subplots(
    1, 1, figsize=(4, 4)
)
vmax = 60

sns.heatmap(control.loc[:, ['left_hits', 'right_hits']], ax = axs, rasterized = True, vmax = vmax)
axs.set_ylabel('Descending neuron pairs')
axs.set_title('ORN Cascade\n100% right, 95% left')

plt.savefig('cascades/interhemisphere_plots/100_vs_95_ORNsignal_all_descending.pdf', bbox_inches='tight')


fig, axs = plt.subplots(
    1, 1, figsize=(4, 4)
)

sns.heatmap(test.loc[:, ['left_hits', 'right_hits']], ax = axs, rasterized = True, vmax = vmax)
axs.set_ylabel('Descending neuron pairs')
axs.set_title('ORN Cascade\n100% right, 75% left')

plt.savefig('cascades/interhemisphere_plots/100_vs_75_ORNsignal_all_descending.pdf', bbox_inches='tight')


fig, axs = plt.subplots(
    1, 1, figsize=(4, 4)
)

sns.heatmap(test0.loc[:, ['left_hits', 'right_hits']], ax = axs, rasterized = True, vmax = vmax)
axs.set_ylabel('Descending neuron pairs')
axs.set_title('ORN Cascade\n100% right, 50% left')

plt.savefig('cascades/interhemisphere_plots/100_vs_50_ORNsignal_all_descending.pdf', bbox_inches='tight')

fig, axs = plt.subplots(
    1, 1, figsize=(4, 4)
)
sns.heatmap(test1.loc[:, ['left_hits', 'right_hits']], ax = axs, rasterized = True, vmax = vmax)
axs.set_ylabel('Descending neuron pairs')
axs.set_title('ORN Cascade\n100% right, 25% left')

plt.savefig('cascades/interhemisphere_plots/100_vs_25_ORNsignal_all_descending.pdf', bbox_inches='tight')

fig, axs = plt.subplots(
    1, 1, figsize=(4, 4)
)
sns.heatmap(test2.loc[:, ['left_hits', 'right_hits']], ax = axs, rasterized = True, vmax = vmax)
axs.set_ylabel('Descending neuron pairs')
axs.set_title('ORN Cascade\n100% right, 10% left')

plt.savefig('cascades/interhemisphere_plots/100_vs_10_ORNsignal_all_descending.pdf', bbox_inches='tight')

# opposite side
fig, axs = plt.subplots(
    1, 1, figsize=(4, 4)
)
sns.heatmap(test2R.loc[:, ['left_hits', 'right_hits']], ax = axs, rasterized = True, vmax = vmax)
axs.set_ylabel('Descending neuron pairs')
axs.set_title('ORN Cascade\n25% right, 100% left')

plt.savefig('cascades/interhemisphere_plots/25_vs_100_ORNsignal_all_descending.pdf', bbox_inches='tight')

fig, axs = plt.subplots(
    1, 1, figsize=(4, 4)
)
sns.heatmap(test2R.loc[:, ['left_hits', 'right_hits']], ax = axs, rasterized = True, vmax = vmax)
axs.set_ylabel('Descending neuron pairs')
axs.set_title('ORN Cascade\n10% right, 100% left')

plt.savefig('cascades/interhemisphere_plots/10_vs_100_ORNsignal_all_descending.pdf', bbox_inches='tight')

# %%
# plotting dVNC projectome based on ordering of asymmetric signal

fig, axs = plt.subplots(
    3, 1, figsize=(6, 10)
)
threshold = 25
fig.tight_layout(pad=3.0)

ax = axs[0]
ax.set_title('Descending neurons more active on right by >%i' %threshold)
sns.heatmap(dVNC_projectome_pairs.loc[test0.leftid[test0['diff']>threshold], ['T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']], ax = ax, rasterized = True)

ax = axs[1]
ax.set_title('Descending neurons more active on left by >%i' %threshold)
sns.heatmap(dVNC_projectome_pairs.loc[test0.leftid[test0['diff']<-threshold], ['T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']], ax = ax, rasterized = True)

ax = axs[2]
ax.axis("off")
caption = f"Figure: Output Projectome for Descending Neurons,\n downstream of ORN Cascade 100% right signal + 50% ORN left signal\n\n"
ax.text(0, 1, caption, va="top")

plt.savefig('cascades/interhemisphere_plots/100_vs_50_ORNsignal_descendings.pdf', bbox_inches='tight')

# %%
# plotting dVNC projectome based on ordering of asymmetric signal

fig, axs = plt.subplots(
    3, 1, figsize=(6, 10)
)
threshold = 25
fig.tight_layout(pad=3.0)

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

plt.savefig('cascades/interhemisphere_plots/100_vs_25_ORNsignal_descendings.pdf', bbox_inches='tight')

# %%
# plotting dVNC projectome based on ordering of asymmetric signal

fig, axs = plt.subplots(
    3, 1, figsize=(6, 10)
)
threshold = 25
fig.tight_layout(pad=3.0)

ax = axs[0]
ax.set_title('Descending neurons more active on right by >%i' %threshold)
sns.heatmap(dVNC_projectome_pairs.loc[test2.leftid[test2['diff']>threshold], ['T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']], ax = ax, rasterized = True)

ax = axs[1]
ax.set_title('Descending neurons more active on left by >%i' %(threshold))
sns.heatmap(dVNC_projectome_pairs.loc[test2.leftid[test2['diff']<-threshold], ['T1', 'T2', 'T3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']], ax = ax, rasterized = True)

ax = axs[2]
ax.axis("off")
caption = f"Figure: Output Projectome for Descending Neurons,\n downstream of ORN Cascade 100% right signal + 10% ORN left signal\n\n"
ax.text(0, 1, caption, va="top")

plt.savefig('cascades/interhemisphere_plots/100_vs_10_ORNsignal_descendings.pdf', bbox_inches='tight')


# %%
# comparing hops 

def sort_pairs_hops(skids_left, skids_right, indices_left, indices_right, pairList, hit_hist_list, right = True):
    
    skids = np.concatenate([skids_left, skids_right])
    pairs = []
    for i in skids:
        if(int(i) in pairList["leftid"].values):
            pair = promat.get_paired_skids(int(i), pairList)
            left_index = indices_left[skids_left==pair[0]][0]
            right_index = indices_right[skids_right==pair[1]][0]

            left_hits = hit_hist_list.iloc[left_index, :]
            right_hits = hit_hist_list.iloc[right_index, :]

            pairs.append(left_hits)
            pairs.append(right_hits)


    return(pd.DataFrame(pairs))

fullR_25L_dVNC = sort_pairs_hops(dVNC_left, dVNC_right, dVNC_indices_left, dVNC_indices_right, pairs, fullR_25L_hist)
fullR_fullL_dVNC = sort_pairs_hops(dVNC_left, dVNC_right, dVNC_indices_left, dVNC_indices_right, pairs, fullR_fullL_hist)

diff_fullR_25L_dVNC = fullR_25L_dVNC - fullR_fullL_dVNC