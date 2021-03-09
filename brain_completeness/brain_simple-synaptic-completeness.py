#%%
import os
import sys
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
except:
    pass

import numpy as np
import pandas as pd
import pymaid as pymaid

all_brain_edges = pd.read_csv('brain_completeness/data/all_edges/all_brain_edges.csv')
brain_related_skels = pd.read_csv('brain_completeness/data/all_edges/brain_related_skels.csv')
# %%
# pre completion

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
print(f'Fraction Complete Pre: {pre_completeness}')

# %%
# post completion

post_only = all_brain_edges.copy()
post_only = post_only.loc[:, ['connector', 'post_skeleton']].set_index('post_skeleton')

complete_post = len(post_only.loc[np.intersect1d(soma_skel, post_only.index)])
total_post = len(post_only)

post_completeness = complete_post/total_post

print(f'Total Post: {total_post}')
print(f'Complete Post: {complete_post}')
print(f'Fraction Complete Post: {post_completeness}')
# %%
# summed post/pre completion

complete_all = complete_post + complete_pre
total_all = total_post + total_pre

all_completeness = complete_all/total_all
print(f'Total Post: {total_all}')
print(f'Complete Post: {complete_all}')
print(f'Fraction Complete Post: {all_completeness}')
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
