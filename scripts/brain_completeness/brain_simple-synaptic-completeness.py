#%%

import numpy as np
import pandas as pd
from contools import Promat
from pymaid_creds import url, name, password, token
from data_settings import data_date, pairs_path
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

# datasets generated from https://github.com/clbarnes/brain_completion/blob/master/brain_completion/scripts/completeness/all_edges.py
all_brain_edges = pd.read_csv('data/completeness/all_brain_edges_2022-11-09.csv')
brain_related_skels = pd.read_csv('data/completeness/brain_related_skels_2022-11-09.csv')

# %%
# presynaptic site completion

pre_only = all_brain_edges.copy()
pre_only = pre_only.loc[:, ['pre_skeleton', 'connector']].set_index('pre_skeleton')
pre_only = pre_only.drop_duplicates()

is_complete = (brain_related_skels['n_soma_like'] + brain_related_skels['mw brain sensories']).astype(bool)
soma_skel = brain_related_skels[is_complete].skeleton
external_fragments = pymaid.get_skids_by_annotation('mw brain external fragments')
soma_skel = list(soma_skel) + external_fragments

complete_pre = len(pre_only.loc[np.intersect1d(soma_skel, pre_only.index)])
total_pre = len(pre_only)

pre_completeness = complete_pre/total_pre

print(f'Total Pre: {total_pre}')
print(f'Complete Pre: {complete_pre}')
print(f'Complete Presynaptic Sites: {pre_completeness*100:.1f}%')

# %%
# postsynaptic site completion

post_only = all_brain_edges.copy()
post_only = post_only.loc[:, ['connector', 'post_skeleton']].set_index('post_skeleton')

complete_post = len(post_only.loc[np.intersect1d(soma_skel, post_only.index)])
total_post = len(post_only)

post_completeness = complete_post/total_post

print(f'Total Post: {total_post}')
print(f'Complete Post: {complete_post}')
print(f'Complete Postsynaptic Sites: {post_completeness*100:.1f}%')
# %%
# all synaptic site completion

complete_all = complete_post + complete_pre
total_all = total_post + total_pre

all_completeness = complete_all/total_all
print(f'Total Synaptic Sites: {total_all}')
print(f'Complete Synaptic Sites: {complete_all}')
print(f'Complete Synaptic Sites: {all_completeness*100:.1f}%')

print(f'\nComplete Presynaptic Sites: {pre_completeness*100:.1f}%')
print(f'Complete Postsynaptic Sites: {post_completeness*100:.1f}%')
print(f'Complete Synaptic Sites: {all_completeness*100:.1f}%')

# %%
# number of differentiated brain neurons, completed on left and right

brain = pymaid.get_skids_by_annotation('mw brain neurons')
left = pymaid.get_skids_by_annotation('mw left')
right = pymaid.get_skids_by_annotation('mw right')

brain_right = len(np.intersect1d(right, brain))
brain_left = len(np.intersect1d(left, brain))

brain_right_incomplete = len(pymaid.get_skids_by_annotation('mw right incomplete'))
brain_left_incomplete = len(pymaid.get_skids_by_annotation('mw left incomplete'))

print(f'Total differentiated neurons: {len(brain)}')
print(f'Total completed neurons: {(brain_left - brain_left_incomplete) + (brain_right - brain_right_incomplete)}')
print(f'Percent completed neurons: {((brain_left - brain_left_incomplete)+ (brain_right - brain_right_incomplete))/len(brain)}')

print(f'\nTotal left neurons: {brain_left}')
print(f'Total completed left neurons: {brain_left - brain_left_incomplete}')
print(f'Percent completed left neurons: {((brain_left - brain_left_incomplete)/(brain_left))*100:.1f}%')

print(f'\nTotal right neurons: {brain_right}')
print(f'Total completed right neurons: {brain_right - brain_right_incomplete}')
print(f'Percent completed right neurons: {((brain_right - brain_right_incomplete)/(brain_right))*100:.1f}%')

# %%
# number of paired neurons and nonpaired

pairs = Promat.get_pairs(pairs_path=pairs_path)
paired_neurons = np.concatenate([pairs.leftid, pairs.rightid, pymaid.get_skids_by_annotation('mw duplicated neurons')]) # duplicated neurons considered paired
KC = pymaid.get_skids_by_annotation('mw KC')

paired_neurons_brain = np.intersect1d(brain, paired_neurons)
print(f'Paired brain neurons: {len(paired_neurons_brain)}')

nonpaired_neurons_brain = np.setdiff1d(brain, paired_neurons)
print(f'Nonpaired KCs: {len(np.intersect1d(nonpaired_neurons_brain, KC))}')
print(f'Nonpaired brain neurons: {len(np.setdiff1d(nonpaired_neurons_brain, KC))}')

# %%
# edge completion
from tqdm import tqdm

complete_edges = 0
for i in tqdm(range(0, len(all_brain_edges))):
    if((all_brain_edges.loc[i, 'pre_skeleton'] in soma_skel) & (all_brain_edges.loc[i, 'post_skeleton'] in soma_skel)):
        complete_edges+=1

total_edges = len(all_brain_edges)

edge_completeness = complete_edges/total_edges

print(f'Total Edges: {total_edges}')
print(f'Complete Edges: {complete_edges}')
print(f'Fraction Complete Edges: {edge_completeness}')

# %%
# projectome synapses

projectome = pd.read_csv('data/projectome.csv')

SEZ_inputs = sum((projectome.is_input==1) & ((projectome.SEZ_left==1) | (projectome.SEZ_right==1)))
SEZ_outputs = sum((projectome.is_input==0) & ((projectome.SEZ_left==1) | (projectome.SEZ_right==1)))

VNC_inputs = sum((projectome.is_input==1) & (projectome.loc[:, projectome.columns[10:]].sum(axis=1)==1))
VNC_outputs = sum((projectome.is_input==0) & (projectome.loc[:, projectome.columns[10:]].sum(axis=1)==1))

print(f'Total post in SEZ: {SEZ_inputs}')
print(f'Total pre in SEZ: {SEZ_outputs}')
print(f'Total post in VNC: {VNC_inputs}')
print(f'Total pre in VNC: {VNC_outputs}')

# %%
# identify fragments

not_complete = ((brain_related_skels['n_soma_like']==0) & (brain_related_skels['mw brain sensories']==0))
frag_skel = brain_related_skels[not_complete].skeleton

pre_only.loc[np.intersect1d(frag_skel, pre_only.index)]
post_only.loc[np.intersect1d(frag_skel, post_only.index)]

pre_frags = np.unique(pre_only.loc[np.intersect1d(frag_skel, pre_only.index)].index)
post_frags = np.unique(post_only.loc[np.intersect1d(frag_skel, post_only.index)].index)

#pd.DataFrame(pre_frags).to_csv('brain_completeness/data/pre_frags.csv')

post_frag_stats = brain_related_skels.set_index('skeleton').loc[post_frags]

#pd.DataFrame(post_frag_stats[post_frag_stats.n_nodes>200].index).to_csv('brain_completeness/data/post_frags.csv')
# %%
# number of non-brain neurons

char_tab = pd.read_csv('data/characterization_table_2021_03_05.csv')
other_cells = list(char_tab[char_tab.category=='other_cell'].skeleton_id)
brain = pymaid.get_skids_by_annotation('mw brain neurons')
brain_su = np.setdiff1d(pymaid.get_skids_by_annotation(f'mw potential SU'), brain)
other_cells = np.setdiff1d(other_cells, brain_su)

pymaid.add_annotations(other_cells, 'mw brain other cells')

#external = list(char_tab[char_tab.category=='external_object'].skeleton_id)

# %%
