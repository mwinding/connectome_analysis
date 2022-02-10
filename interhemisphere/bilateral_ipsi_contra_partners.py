#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import connectome_tools.celltype as ct
import connectome_tools.process_graph as pg
import connectome_tools.process_matrix as pm
import navis

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

adj = pm.Promat.pull_adj(type_adj='ad', subgraph='brain')
inputs = pd.read_csv('data/graphs/inputs.csv', index_col=0)
adj_mat = pm.Adjacency_matrix(adj, inputs, 'ad')
pairs = pm.Promat.get_pairs()
# %%
# load bilaterals

bilateral = pymaid.get_skids_by_annotation('mw bilateral axon')
clustered = pymaid.get_skids_by_annotation('mw brain paper clustered neurons')
sens = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain inputs and ascending')
outputs = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain outputs')

bilateral = np.intersect1d(bilateral, clustered)
bilateral = np.setdiff1d(bilateral, sens + outputs)

bilateral_pairs, bilateral_unpaired, bilateral_nonpaired = pm.Promat.extract_pairs_from_list(bilateral, pairs)
recover_pairs = [pm.Promat.get_paired_skids(skid, pairs) for skid in list(bilateral_unpaired.unpaired)]

bilateral_all = [x for sublist in bilateral_pairs.values for x in sublist] + [x for sublist in recover_pairs for x in sublist] + list(bilateral_nonpaired.nonpaired)

# %%
# identify downstream partners ipsi/contra and upstream partners

bilateral_pairs_all, _, nonpaired = pm.Promat.extract_pairs_from_list(bilateral_all, pairs)
bilateral_pairids = list(bilateral_pairs_all.leftid) + list(nonpaired.nonpaired)

edge_list_ad = pd.read_csv('data/edges_threshold/ad_all-paired-edges.csv', index_col=0)
edge_list_ad = edge_list_ad.set_index('upstream_pair_id')

bilateral_pairids = list(np.intersect1d(bilateral_pairids, edge_list_ad.index))

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    calculation = dot / (norma * normb)

    return(calculation)

def dice(a, b):
    intersection = len(np.intersect1d(a, b))
    diff1 = len(np.setdiff1d(a, b))
    diff2 = len(np.setdiff1d(b, a))
    calculation = intersection*2/(intersection*2 + diff1 + diff2)

    return(calculation)

def iou(a, b):
    intersection = len(np.intersect1d(a, b))
    union = len(np.union1d(a, b))
    calculation = intersection/union

    return(calculation)


bilateral_data = []
for bilateral in bilateral_pairids:
    data = edge_list_ad.loc[bilateral, :]
    upstream_data = edge_list_ad[edge_list_ad.downstream_pair_id==bilateral]
    if((type(data)!=pd.Series) & (len(np.unique(data.type))>1)): # if is pd.Series indicates the bilateral only has one partner, therefore can't do the analysis; data.type makes sure there is ipsi and contra to compare

        ipsi = data[data.type=='ipsilateral'].loc[:, ['downstream_pair_id', 'left', 'right']].set_index('downstream_pair_id', drop=True)
        contra = data[data.type=='contralateral'].loc[:, ['downstream_pair_id', 'left', 'right']].set_index('downstream_pair_id', drop=True)
        ipsi.columns = ['ipsi-left', 'ipsi-right']
        contra.columns = ['contra-left', 'contra-right']

        data = pd.concat([ipsi, contra], axis=1).fillna(0)

        cos_left = cosine_similarity(list(data.loc[:, 'ipsi-left']), list(data.loc[:, 'contra-left']))
        cos_right = cosine_similarity(list(data.loc[:, 'ipsi-right']), list(data.loc[:, 'contra-right']))
        cos = np.mean([cos_left, cos_right])

        dice_data = dice(list(ipsi.index), list(contra.index))
        iou_data = iou(list(ipsi.index), list(contra.index))

        # collect all left/right partners from the pairids
        full_ipsi = [pm.Promat.get_paired_skids(skid, pairs) for skid in list(ipsi.index)]
        full_contra = [pm.Promat.get_paired_skids(skid, pairs) for skid in list(contra.index)]
        full_upstream = [pm.Promat.get_paired_skids(skid, pairs) for skid in list(upstream_data.index)]
        full_ipsi = [x for sublist in full_ipsi for x in sublist]
        full_contra = [x for sublist in full_contra for x in sublist]
        full_upstream = [x for sublist in full_upstream for x in sublist]

        bilateral_data.append([bilateral, cos, dice_data, iou_data, full_ipsi, full_contra, full_upstream])

bilateral_data = pd.DataFrame(bilateral_data, columns=['pairid', 'cosine', 'dice', 'iou', 'ipsi_partners', 'contra_partners', 'upstream_partners']).set_index('pairid')

fig,ax = plt.subplots(1,1, figsize=(1.25,1.75))
sns.swarmplot(y=bilateral_data.cosine, orient='v', color = sns.color_palette()[1], s=2, ax=ax)
plt.savefig('interhemisphere/plots/bilateral_ipsi-contra-partners_cosine-similarity.pdf', format='pdf', bbox_inches='tight')

# %%
# compare membership identities of ipsi vs contra partners for different bins of cosine similarity

# made three bins of bilateral neurons
bilateral_diff = bilateral_data[bilateral_data.cosine == 0]
bilateral_pdiff = bilateral_data[(bilateral_data.cosine > 0) & (bilateral_data.cosine <= 0.33)]
bilateral_same = bilateral_data[bilateral_data.cosine > 0.33]

# collect ipsi and contra partners for different bins
bilateral_diff_ipsi = np.unique([x for sublist in bilateral_diff.ipsi_partners for x in sublist])
bilateral_diff_contra = np.unique([x for sublist in bilateral_diff.contra_partners for x in sublist])

bilateral_pdiff_ipsi = np.unique([x for sublist in bilateral_pdiff.ipsi_partners for x in sublist])
bilateral_pdiff_contra = np.unique([x for sublist in bilateral_pdiff.contra_partners for x in sublist])

bilateral_same_ipsi = np.unique([x for sublist in bilateral_same.ipsi_partners for x in sublist])
bilateral_same_contra = np.unique([x for sublist in bilateral_same.contra_partners for x in sublist])

# generate Celltype objects for each cell class
bilateral_partners_ct = [bilateral_diff_ipsi, bilateral_diff_contra, 
                            bilateral_pdiff_ipsi, bilateral_pdiff_contra, 
                            bilateral_same_ipsi, bilateral_same_contra]

bilateral_partners_names = ['diff_ipsi', 'diff_contra', 'pdiff_ipsi', 'pdiff_contra', 'same_ipsi', 'same_contra']
bilateral_partners_ct = [ct.Celltype(bilateral_partners_names[i], skids) for i, skids in enumerate(bilateral_partners_ct)]

# generate Celltype_Analyzer to display which known celltypes comprise these groups
bilateral_partners = ct.Celltype_Analyzer(bilateral_partners_ct)
_, all_celltypes = ct.Celltype_Analyzer.default_celltypes()
bilateral_partners.set_known_types(all_celltypes)
memberships = bilateral_partners.memberships()

# plot memberships
bilateral_partners.plot_memberships('interhemisphere/plots/bilateral_ipsi-contra-partners_memberships.pdf', (1,1))

# %%
# examine partners of bilaterals with asymmetrical output and partially asymmetrical output

# asymmetrical bilaterals
bilateral_diff_cts = []
for skid in bilateral_diff.index:
    ipsi = bilateral_diff.loc[skid, :].ipsi_partners
    contra = bilateral_diff.loc[skid, :].contra_partners

    bilateral_diff_cts.append(ct.Celltype(f'{skid}-ipsi', ipsi))
    bilateral_diff_cts.append(ct.Celltype(f'{skid}-contra', contra))
    bilateral_diff_cts.append(ct.Celltype(f'{skid}-spacer', [])) # add these blank columns for formatting purposes only

bilateral_diff_cts = ct.Celltype_Analyzer(bilateral_diff_cts)
bilateral_diff_cts.set_known_types(all_celltypes)
bilateral_diff_cts_memberships = bilateral_diff_cts.memberships(raw_num=True)

for i in np.arange(0, int(len(bilateral_diff_cts_memberships.columns)), 3):
    total = sum(bilateral_diff_cts_memberships.iloc[:, i]) + sum(bilateral_diff_cts_memberships.iloc[:, i+1])
    bilateral_diff_cts_memberships.iloc[:, i] = bilateral_diff_cts_memberships.iloc[:, i]/total
    bilateral_diff_cts_memberships.iloc[:, i+1] = bilateral_diff_cts_memberships.iloc[:, i+1]/total

ylim = [0,1]
path = 'interhemisphere/plots/bilateral-asym_ipsi-contra-partners_memberships.pdf'
bilateral_diff_cts.plot_memberships(path=path, figsize=(0.1*len(bilateral_diff_cts.Celltypes),1), memberships=bilateral_diff_cts_memberships, ylim=ylim)

# asymmetrical bilaterals upstream
bilateral_diff_us_cts = []
for skid in bilateral_diff.index:
    us = bilateral_diff.loc[skid, :].upstream_partners

    bilateral_diff_us_cts.append(ct.Celltype(f'{skid}-upstream', us))
    bilateral_diff_us_cts.append(ct.Celltype(f'{skid}-spacer', [])) # add these blank columns for formatting purposes only
    bilateral_diff_us_cts.append(ct.Celltype(f'{skid}-spacer2', []))

bilateral_diff_us_cts = ct.Celltype_Analyzer(bilateral_diff_us_cts)
bilateral_diff_us_cts.set_known_types(all_celltypes)
bilateral_diff_us_cts_memberships = bilateral_diff_us_cts.memberships(raw_num=True)

ylim = [0,1]
path = 'interhemisphere/plots/bilateral-asym_upstream-partners_memberships.pdf'
bilateral_diff_us_cts.plot_memberships(path=path, figsize=(0.1*len(bilateral_diff_us_cts.Celltypes),1), ylim=ylim)

# partially asymmetrical bilaterals
bilateral_pdiff_cts = []
for skid in bilateral_pdiff.index:
    ipsi = bilateral_pdiff.loc[skid, :].ipsi_partners
    contra = bilateral_pdiff.loc[skid, :].contra_partners

    bilateral_pdiff_cts.append(ct.Celltype(f'{skid}-ipsi', ipsi))
    bilateral_pdiff_cts.append(ct.Celltype(f'{skid}-contra', contra))
    bilateral_pdiff_cts.append(ct.Celltype(f'{skid}-spacer', [])) # add these blank columns for formatting purposes only

bilateral_pdiff_cts = ct.Celltype_Analyzer(bilateral_pdiff_cts)
bilateral_pdiff_cts.set_known_types(all_celltypes)
bilateral_pdiff_cts_memberships = bilateral_pdiff_cts.memberships(raw_num=True)

for i in np.arange(0, int(len(bilateral_pdiff_cts_memberships.columns)), 3):
    total = sum(bilateral_pdiff_cts_memberships.iloc[:, i]) + sum(bilateral_pdiff_cts_memberships.iloc[:, i+1])
    bilateral_pdiff_cts_memberships.iloc[:, i] = bilateral_pdiff_cts_memberships.iloc[:, i]/total
    bilateral_pdiff_cts_memberships.iloc[:, i+1] = bilateral_pdiff_cts_memberships.iloc[:, i+1]/total

ylim = [0,1]
path = 'interhemisphere/plots/bilateral-partially-asym_ipsi-contra-partners_memberships.pdf'
bilateral_pdiff_cts.plot_memberships(path=path, figsize=(0.1*len(bilateral_pdiff_cts.Celltypes),1), memberships=bilateral_pdiff_cts_memberships, ylim=ylim)

# asymmetrical bilaterals upstream
bilateral_pdiff_us_cts = []
for skid in bilateral_pdiff.index:
    us = bilateral_pdiff.loc[skid, :].upstream_partners

    bilateral_pdiff_us_cts.append(ct.Celltype(f'{skid}-upstream', us))
    bilateral_pdiff_us_cts.append(ct.Celltype(f'{skid}-spacer', [])) # add these blank columns for formatting purposes only
    bilateral_pdiff_us_cts.append(ct.Celltype(f'{skid}-spacer2', []))

bilateral_pdiff_us_cts = ct.Celltype_Analyzer(bilateral_pdiff_us_cts)
bilateral_pdiff_us_cts.set_known_types(all_celltypes)
bilateral_pdiff_us_cts_memberships = bilateral_pdiff_us_cts.memberships(raw_num=True)

ylim = [0,1]
path = 'interhemisphere/plots/bilateral-partially-asym_upstream-partners_memberships.pdf'
bilateral_pdiff_us_cts.plot_memberships(path=path, figsize=(0.1*len(bilateral_pdiff_us_cts.Celltypes),1), ylim=ylim)
'''
# export partners to CATMAID for bilateral_diff and bilateral_pdiff
for skid in bilateral_diff.index:
    pymaid.add_annotations(bilateral_diff.loc[skid].ipsi_partners, f'mw {skid} ds ipsilateral partners')
    pymaid.add_annotations(bilateral_diff.loc[skid].contra_partners, f'mw {skid} ds contralateral partners')
    pymaid.add_annotations(bilateral_diff.loc[skid].upstream_partners, f'mw {skid} us partners')

    pymaid.add_meta_annotations(f'mw {skid} ds ipsilateral partners', 'mw bilateral axon asymmetrical')
    pymaid.add_meta_annotations(f'mw {skid} ds contralateral partners', 'mw bilateral axon asymmetrical')
    pymaid.add_meta_annotations(f'mw {skid} us partners', 'mw bilateral axon asymmetrical')

for skid in bilateral_pdiff.index:
    pymaid.add_annotations(bilateral_pdiff.loc[skid].ipsi_partners, f'mw {skid} ds ipsilateral partners')
    pymaid.add_annotations(bilateral_pdiff.loc[skid].contra_partners, f'mw {skid} ds contralateral partners')
    pymaid.add_annotations(bilateral_pdiff.loc[skid].upstream_partners, f'mw {skid} us partners')

    pymaid.add_meta_annotations(f'mw {skid} ds ipsilateral partners', 'mw bilateral axon partially asymmetrical')
    pymaid.add_meta_annotations(f'mw {skid} ds contralateral partners', 'mw bilateral axon partially asymmetrical')
    pymaid.add_meta_annotations(f'mw {skid} us partners', 'mw bilateral axon partially asymmetrical')
'''

bilateral_diff_pairs = pm.Promat.get_paired_skids(list(bilateral_diff.index), pairs)
bilateral_pdiff_pairs = pm.Promat.get_paired_skids(list(bilateral_pdiff.index), pairs)

pymaid.add_annotations(list(bilateral_diff_pairs.leftid) + list(bilateral_diff_pairs.rightid), 'mw bilateral axon asymmetrical neurons')
pymaid.add_annotations(list(bilateral_pdiff_pairs.leftid) + list(bilateral_pdiff_pairs.rightid), 'mw bilateral axon partially asymmetrical neurons')

# %%
