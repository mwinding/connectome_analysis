#%%
import sys
import os

os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory
os.chdir(os.path.dirname(os.getcwd()))

from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import cmasher as cmr

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

import connectome_tools.cascade_analysis as casc
import connectome_tools.celltype as ct
import connectome_tools.process_matrix as pm
import pickle 

n_init = 1000
pair_hist_list = pickle.load(open(f'data/cascades/all-brain-pairs_outputs-added_{n_init}-n_init.p', 'rb'))
ds_partners = pickle.load(open(f'data/cascades/all-brain-pairs_ds_partners_{n_init}-n_init.p', 'rb'))
ds_partners_df = pd.DataFrame(list(map(lambda x: [x[0], x[1]], zip([x.name for x in pair_hist_list], ds_partners))), columns=['skid', 'ds_partners'])
ds_partners_df.set_index('skid', inplace=True)

dVNC = pymaid.get_skids_by_annotation('mw dVNC')
dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')

# %%
# identify ds_partners with cascades and how many are in recurrent loops

threshold = n_init/2
hops = 8

dVNC_ds_partners_df = ds_partners_df.loc[np.intersect1d(dVNC, ds_partners_df.index)]

dVNC_feedback = []
dVNC_efference = []
for dVNC_pair in dVNC_ds_partners_df.index:
    ds_partners = dVNC_ds_partners_df.loc[dVNC_pair, 'ds_partners']

    feedback_partners = []
    efference_partners = []
    for ds in ds_partners:
        ds_partners2 = ds_partners_df.loc[ds, 'ds_partners']

        # recurrent partner, i.e. feedback
        if(dVNC_pair in ds_partners2):
            feedback_partners.append(ds)

        # non-recurrent partner, i.e. efference copy
        if(dVNC_pair not in ds_partners2):
            efference_partners.append(ds)

    dVNC_feedback.append(feedback_partners)
    dVNC_efference.append(efference_partners)

dVNC_ds_partners_df['feedback_partners'] = dVNC_feedback
dVNC_ds_partners_df['efference_partners'] = dVNC_efference

frac_feedback = [len(dVNC_ds_partners_df.loc[i, 'feedback_partners'])/len(dVNC_ds_partners_df.loc[i, 'ds_partners']) if len(dVNC_ds_partners_df.loc[i, 'ds_partners'])>0 else 0 for i in dVNC_ds_partners_df.index]
dVNC_ds_partners_df['fraction_feedback_partners'] = frac_feedback
frac_efference = [len(dVNC_ds_partners_df.loc[i, 'efference_partners'])/len(dVNC_ds_partners_df.loc[i, 'ds_partners']) if len(dVNC_ds_partners_df.loc[i, 'ds_partners'])>0 else 0 for i in dVNC_ds_partners_df.index]
dVNC_ds_partners_df['fraction_efference_partners'] = frac_efference

# plot fraction feedback vs. efference
data = dVNC_ds_partners_df.loc[:, ['fraction_feedback_partners', 'fraction_efference_partners']]
fig, ax = plt.subplots(1,1,figsize=(.5,.5))
sns.barplot(data=data.loc[(data!=0).any(axis=1)]) # plot only dVNCs that actually talk to brain neurons
ax.set(ylim=(0,1))
plt.savefig('cascades/feedback_through_brain/plots/feedback-vs-efference_dVNCs.pdf', format='pdf', bbox_inches='tight')


# same for dSEZs
dSEZ_ds_partners_df = ds_partners_df.loc[np.intersect1d(dSEZ, ds_partners_df.index)]

dSEZ_feedback = []
dSEZ_efference = []
for dSEZ_pair in dSEZ_ds_partners_df.index:
    ds_partners = dSEZ_ds_partners_df.loc[dSEZ_pair, 'ds_partners']

    feedback_partners = []
    efference_partners = []
    for ds in ds_partners:
        ds_partners2 = ds_partners_df.loc[ds, 'ds_partners']

        # recurrent partner, i.e. feedback
        if(dSEZ_pair in ds_partners2):
            feedback_partners.append(ds)

        # non-recurrent partner, i.e. efference copy
        if(dSEZ_pair not in ds_partners2):
            efference_partners.append(ds)

    dSEZ_feedback.append(feedback_partners)
    dSEZ_efference.append(efference_partners)

dSEZ_ds_partners_df['feedback_partners'] = dSEZ_feedback
dSEZ_ds_partners_df['efference_partners'] = dSEZ_efference

frac_feedback = [len(dSEZ_ds_partners_df.loc[i, 'feedback_partners'])/(len(dSEZ_ds_partners_df.loc[i, 'ds_partners'])) if len(dSEZ_ds_partners_df.loc[i, 'ds_partners'])>0 else 0 for i in dSEZ_ds_partners_df.index]
dSEZ_ds_partners_df['fraction_feedback_partners'] = frac_feedback
frac_efference = [len(dSEZ_ds_partners_df.loc[i, 'efference_partners'])/len(dSEZ_ds_partners_df.loc[i, 'ds_partners']) if len(dSEZ_ds_partners_df.loc[i, 'ds_partners'])>0 else 0 for i in dSEZ_ds_partners_df.index]
dSEZ_ds_partners_df['fraction_efference_partners'] = frac_efference

# plot fraction feedback vs. efference
data = dSEZ_ds_partners_df.loc[:, ['fraction_feedback_partners', 'fraction_efference_partners']]
fig, ax = plt.subplots(1,1,figsize=(.5,.5))
sns.barplot(data=data.loc[(data!=0).any(axis=1)]) # plot only dSEZs that actually talk to brain neurons
ax.set(ylim=(0,1))
plt.savefig('cascades/feedback_through_brain/plots/feedback-vs-efference_dSEZs.pdf', format='pdf', bbox_inches='tight')

# %%
# which celltypes receive feedback or efference signal?

_, celltypes = ct.Celltype_Analyzer.default_celltypes()

# build Celltype() objects for different feedback and efference types
dVNC_feedback = [dVNC_ds_partners_df.loc[ind, 'feedback_partners'] for ind in dVNC_ds_partners_df.index]
dVNC_feedback = list(np.unique([x for sublist in dVNC_feedback for x in sublist]))
dVNC_efference = [dVNC_ds_partners_df.loc[ind, 'efference_partners'] for ind in dVNC_ds_partners_df.index]
dVNC_efference = list(np.unique([x for sublist in dVNC_efference for x in sublist]))
dSEZ_feedback = [dSEZ_ds_partners_df.loc[ind, 'feedback_partners'] for ind in dSEZ_ds_partners_df.index]
dSEZ_feedback = list(np.unique([x for sublist in dSEZ_feedback for x in sublist]))
dSEZ_efference = [dSEZ_ds_partners_df.loc[ind, 'efference_partners'] for ind in dSEZ_ds_partners_df.index]
dSEZ_efference = list(np.unique([x for sublist in dSEZ_efference for x in sublist]))

ds_cts = [ct.Celltype('dVNC-feedback', dVNC_feedback), ct.Celltype('dVNC-efference', dVNC_efference), ct.Celltype('dSEZ-feedback', dSEZ_feedback), ct.Celltype('dSEZ-efference', dSEZ_efference)]
ds_cts = ct.Celltype_Analyzer(ds_cts)
ds_cts.set_known_types(celltypes)
ds_cts.memberships()

fig, ax = plt.subplots(1,1,figsize=(4,4))
sns.heatmap(data=ds_cts.memberships()) 
plt.savefig('cascades/feedback_through_brain/plots/feedback-efference_celltypes.pdf', format='pdf', bbox_inches='tight')

# %%
# identify directly 1-hop/2-hop downstream neurons: how many are feedback vs efference? (based on cascades)

# characterization of dVNC 1-/2-hop downstream partners
pairs = pm.Promat.get_pairs()

select_neurons = pymaid.get_skids_by_annotation('mw brain neurons')
select_neurons = select_neurons + [pymaid.get_skids_by_annotation('mw brain accessory neurons')]
ad_edges = pm.Promat.pull_edges(type_edges='ad', pairs_combined=False, select_neurons=select_neurons)

dVNC_pairs = pm.Promat.load_pairs_from_annotation('mw dVNC', pairs)
ds_dVNCs = [pm.Promat.downstream_multihop(edges=ad_edges, sources=dVNC_pair, hops=2) for dVNC_pair in dVNC_pairs.values]

ds_dVNCs_df = pd.DataFrame(ds_dVNCs, columns=['dVNC-ds-1hop', 'dVNC-ds-2hop'], index=dVNC_pairs.leftid)
ds_dVNCs_df = ds_dVNCs_df[[False if ds_dVNCs_df.loc[i, 'dVNC-ds-1hop']==[] else True for i in ds_dVNCs_df.index]]

feedback_1hop_list = []
feedback_2hop_list = []
efference_1hop_list = []
efference_2hop_list = []

for skid in ds_dVNCs_df.index:
    feedback_1hop = []
    feedback_2hop = []
    efference_1hop = []
    efference_2hop = []

    ds_1hop = ds_dVNCs_df.loc[skid, 'dVNC-ds-1hop']
    ds_2hop = ds_dVNCs_df.loc[skid, 'dVNC-ds-2hop']

    ds_1hop_pairs = pm.Promat.load_pairs_from_annotation('ds_1hop', pairs, skids=ds_1hop, use_skids=True).leftid
    ds_2hop_pairs = pm.Promat.load_pairs_from_annotation('ds_2hop', pairs, skids=ds_2hop, use_skids=True).leftid

    for pair in ds_1hop_pairs:
        ds_ds_pairs = ds_partners_df.loc[pair, 'ds_partners']
        if(skid in ds_ds_pairs):
            feedback_1hop.append(pair)
        if(skid not in ds_ds_pairs):
            efference_1hop.append(pair)

    for pair in ds_2hop_pairs:
        ds_ds_pairs = ds_partners_df.loc[pair, 'ds_partners']
        if(skid in ds_ds_pairs):
            feedback_2hop.append(pair)
        if(skid not in ds_ds_pairs):
            efference_2hop.append(pair) 

    feedback_1hop_list.append(feedback_1hop)  
    feedback_2hop_list.append(feedback_2hop)  
    efference_1hop_list.append(efference_1hop)  
    efference_2hop_list.append(efference_2hop)  

ds_dVNCs_df['dVNC-ds-1hop_feedback'] = feedback_1hop_list
ds_dVNCs_df['dVNC-ds-2hop_feedback'] = feedback_2hop_list
ds_dVNCs_df['dVNC-ds-1hop_efference'] = efference_1hop_list
ds_dVNCs_df['dVNC-ds-2hop_efference'] = efference_2hop_list


# characterization of dSEZ 1-/2-hop downstream partners
pairs = pm.Promat.get_pairs()

select_neurons = pymaid.get_skids_by_annotation('mw brain neurons')
select_neurons = select_neurons + [pymaid.get_skids_by_annotation('mw brain accessory neurons')]
ad_edges = pm.Promat.pull_edges(type_edges='ad', pairs_combined=False, select_neurons=select_neurons)

dSEZ_pairs = pm.Promat.load_pairs_from_annotation('mw dSEZ', pairs)
ds_dSEZs = [pm.Promat.downstream_multihop(edges=ad_edges, sources=dSEZ_pair, hops=2) for dSEZ_pair in dSEZ_pairs.values]

ds_dSEZs_df = pd.DataFrame(ds_dSEZs, columns=['dSEZ-ds-1hop', 'dSEZ-ds-2hop'], index=dSEZ_pairs.leftid)
ds_dSEZs_df = ds_dSEZs_df[[False if ds_dSEZs_df.loc[i, 'dSEZ-ds-1hop']==[] else True for i in ds_dSEZs_df.index]]

feedback_1hop_list = []
feedback_2hop_list = []
efference_1hop_list = []
efference_2hop_list = []

for skid in ds_dSEZs_df.index:
    feedback_1hop = []
    feedback_2hop = []
    efference_1hop = []
    efference_2hop = []

    ds_1hop = ds_dSEZs_df.loc[skid, 'dSEZ-ds-1hop']
    ds_2hop = ds_dSEZs_df.loc[skid, 'dSEZ-ds-2hop']

    ds_1hop_pairs = pm.Promat.load_pairs_from_annotation('ds_1hop', pairs, skids=ds_1hop, use_skids=True).leftid
    ds_2hop_pairs = pm.Promat.load_pairs_from_annotation('ds_2hop', pairs, skids=ds_2hop, use_skids=True).leftid

    for pair in ds_1hop_pairs:
        ds_ds_pairs = ds_partners_df.loc[pair, 'ds_partners']
        if(skid in ds_ds_pairs):
            feedback_1hop.append(pair)
        if(skid not in ds_ds_pairs):
            efference_1hop.append(pair)

    for pair in ds_2hop_pairs:
        ds_ds_pairs = ds_partners_df.loc[pair, 'ds_partners']
        if(skid in ds_ds_pairs):
            feedback_2hop.append(pair)
        if(skid not in ds_ds_pairs):
            efference_2hop.append(pair)

    feedback_1hop_list.append(feedback_1hop)  
    feedback_2hop_list.append(feedback_2hop)  
    efference_1hop_list.append(efference_1hop)  
    efference_2hop_list.append(efference_2hop)  

ds_dSEZs_df['dSEZ-ds-1hop_feedback'] = feedback_1hop_list
ds_dSEZs_df['dSEZ-ds-2hop_feedback'] = feedback_2hop_list
ds_dSEZs_df['dSEZ-ds-1hop_efference'] = efference_1hop_list
ds_dSEZs_df['dSEZ-ds-2hop_efference'] = efference_2hop_list

# %%
# fraction feedback vs efference

ds_dSEZs_df['fraction_feedback_1hop'] = [len(ds_dSEZs_df.loc[i, 'dSEZ-ds-1hop_feedback'])/(len(ds_dSEZs_df.loc[i, 'dSEZ-ds-1hop_feedback']) + len(ds_dSEZs_df.loc[i, 'dSEZ-ds-1hop_efference'])) for i in ds_dSEZs_df.index]
ds_dSEZs_df['fraction_feedback_2hop'] = [len(ds_dSEZs_df.loc[i, 'dSEZ-ds-2hop_feedback'])/(len(ds_dSEZs_df.loc[i, 'dSEZ-ds-2hop_feedback']) + len(ds_dSEZs_df.loc[i, 'dSEZ-ds-2hop_efference'])) if (len(ds_dSEZs_df.loc[i, 'dSEZ-ds-2hop_feedback']) + len(ds_dSEZs_df.loc[i, 'dSEZ-ds-2hop_efference']))>0 else 0 for i in ds_dSEZs_df.index]
ds_dSEZs_df['fraction_efference_1hop'] = [len(ds_dSEZs_df.loc[i, 'dSEZ-ds-1hop_efference'])/(len(ds_dSEZs_df.loc[i, 'dSEZ-ds-1hop_feedback']) + len(ds_dSEZs_df.loc[i, 'dSEZ-ds-1hop_efference'])) for i in ds_dSEZs_df.index]
ds_dSEZs_df['fraction_efference_2hop'] = [len(ds_dSEZs_df.loc[i, 'dSEZ-ds-2hop_efference'])/(len(ds_dSEZs_df.loc[i, 'dSEZ-ds-2hop_feedback']) + len(ds_dSEZs_df.loc[i, 'dSEZ-ds-2hop_efference'])) if (len(ds_dSEZs_df.loc[i, 'dSEZ-ds-2hop_feedback']) + len(ds_dSEZs_df.loc[i, 'dSEZ-ds-2hop_efference']))>0 else 0 for i in ds_dSEZs_df.index]

ds_dVNCs_df['fraction_feedback_1hop'] = [len(ds_dVNCs_df.loc[i, 'dVNC-ds-1hop_feedback'])/(len(ds_dVNCs_df.loc[i, 'dVNC-ds-1hop_feedback']) + len(ds_dVNCs_df.loc[i, 'dVNC-ds-1hop_efference'])) for i in ds_dVNCs_df.index]
ds_dVNCs_df['fraction_feedback_2hop'] = [len(ds_dVNCs_df.loc[i, 'dVNC-ds-2hop_feedback'])/(len(ds_dVNCs_df.loc[i, 'dVNC-ds-2hop_feedback']) + len(ds_dVNCs_df.loc[i, 'dVNC-ds-2hop_efference'])) if (len(ds_dVNCs_df.loc[i, 'dVNC-ds-2hop_feedback']) + len(ds_dVNCs_df.loc[i, 'dVNC-ds-2hop_efference']))>0 else 0 for i in ds_dVNCs_df.index]
ds_dVNCs_df['fraction_efference_1hop'] = [len(ds_dVNCs_df.loc[i, 'dVNC-ds-1hop_efference'])/(len(ds_dVNCs_df.loc[i, 'dVNC-ds-1hop_feedback']) + len(ds_dVNCs_df.loc[i, 'dVNC-ds-1hop_efference'])) for i in ds_dVNCs_df.index]
ds_dVNCs_df['fraction_efference_2hop'] = [len(ds_dVNCs_df.loc[i, 'dVNC-ds-2hop_efference'])/(len(ds_dVNCs_df.loc[i, 'dVNC-ds-2hop_feedback']) + len(ds_dVNCs_df.loc[i, 'dVNC-ds-2hop_efference'])) if (len(ds_dVNCs_df.loc[i, 'dVNC-ds-2hop_feedback']) + len(ds_dVNCs_df.loc[i, 'dVNC-ds-2hop_efference']))>0 else 0 for i in ds_dVNCs_df.index]

# plot
fig, ax = plt.subplots(1,1,figsize=(4,4))
data = ds_dVNCs_df.loc[:, ['fraction_feedback_1hop', 'fraction_efference_1hop', 'fraction_feedback_2hop', 'fraction_efference_2hop']]
sns.barplot(data=data)
plt.savefig('cascades/feedback_through_brain/plots/feedback-efference_ds-dVNCs.pdf', format='pdf', bbox_inches='tight')

# %%
# cell types within 1-hop/2-hop feedback/efference categories

_, celltypes = ct.Celltype_Analyzer.default_celltypes()

# build Celltype() objects for different feedback and efference types
dVNC_feedback_casc = [dVNC_ds_partners_df.loc[ind, 'feedback_partners'] for ind in dVNC_ds_partners_df.index]
dVNC_feedback_casc = list(np.unique([x for sublist in dVNC_feedback_casc for x in sublist]))
dVNC_efference_casc = [dVNC_ds_partners_df.loc[ind, 'efference_partners'] for ind in dVNC_ds_partners_df.index]
dVNC_efference_casc = list(np.unique([x for sublist in dVNC_efference_casc for x in sublist]))
dSEZ_feedback_casc = [dSEZ_ds_partners_df.loc[ind, 'feedback_partners'] for ind in dSEZ_ds_partners_df.index]
dSEZ_feedback_casc = list(np.unique([x for sublist in dSEZ_feedback_casc for x in sublist]))
dSEZ_efference_casc = [dSEZ_ds_partners_df.loc[ind, 'efference_partners'] for ind in dSEZ_ds_partners_df.index]
dSEZ_efference_casc = list(np.unique([x for sublist in dSEZ_efference_casc for x in sublist]))

# build Celltype() objects for different feedback and efference types (1-hop)
dVNC_feedback_1hop = [ds_dVNCs_df.loc[ind, 'dVNC-ds-1hop_feedback'] for ind in ds_dVNCs_df.index]
dVNC_feedback_1hop = list(np.unique([x for sublist in dVNC_feedback_1hop for x in sublist]))
dVNC_efference_1hop = [ds_dVNCs_df.loc[ind, 'dVNC-ds-1hop_efference'] for ind in ds_dVNCs_df.index]
dVNC_efference_1hop = list(np.unique([x for sublist in dVNC_efference_1hop for x in sublist]))
dSEZ_feedback_1hop = [ds_dSEZs_df.loc[ind, 'dSEZ-ds-1hop_feedback'] for ind in ds_dSEZs_df.index]
dSEZ_feedback_1hop = list(np.unique([x for sublist in dSEZ_feedback_1hop for x in sublist]))
dSEZ_efference_1hop = [ds_dSEZs_df.loc[ind, 'dSEZ-ds-1hop_efference'] for ind in ds_dSEZs_df.index]
dSEZ_efference_1hop = list(np.unique([x for sublist in dSEZ_efference_1hop for x in sublist]))

# build Celltype() objects for different feedback and efference types (2-hop)
dVNC_feedback_2hop = [ds_dVNCs_df.loc[ind, 'dVNC-ds-2hop_feedback'] for ind in ds_dVNCs_df.index]
dVNC_feedback_2hop = list(np.unique([x for sublist in dVNC_feedback_2hop for x in sublist]))
dVNC_efference_2hop = [ds_dVNCs_df.loc[ind, 'dVNC-ds-2hop_efference'] for ind in ds_dVNCs_df.index]
dVNC_efference_2hop = list(np.unique([x for sublist in dVNC_efference_2hop for x in sublist]))
dSEZ_feedback_2hop = [ds_dSEZs_df.loc[ind, 'dSEZ-ds-2hop_feedback'] for ind in ds_dSEZs_df.index]
dSEZ_feedback_2hop = list(np.unique([x for sublist in dSEZ_feedback_2hop for x in sublist]))
dSEZ_efference_2hop = [ds_dSEZs_df.loc[ind, 'dSEZ-ds-2hop_efference'] for ind in ds_dSEZs_df.index]
dSEZ_efference_2hop = list(np.unique([x for sublist in dSEZ_efference_2hop for x in sublist]))
'''
# combined plot
dVNCs_cts = [ct.Celltype('dVNC-feedback-1hop', dVNC_feedback_1hop), ct.Celltype('dVNC-efference-1hop', dVNC_efference_1hop), 
        ct.Celltype('dVNC-feedback-2hop', dVNC_feedback_2hop), ct.Celltype('dVNC-efference-2hop', dVNC_efference_2hop),
        ct.Celltype('dVNC-feedback-casc', dVNC_feedback_casc), ct.Celltype('dVNC-efference-casc', dVNC_efference_casc)]
dVNCs_cts = ct.Celltype_Analyzer(dVNCs_cts)
dVNCs_cts.set_known_types(celltypes)
dVNCs_cts.memberships()

fig, ax = plt.subplots(1,1,figsize=(4,4))
sns.heatmap(data=dVNCs_cts.memberships()) 
plt.savefig('cascades/feedback_through_brain/plots/feedback-efference_dVNC_celltypes.pdf', format='pdf', bbox_inches='tight')

dSEZs_cts = [ct.Celltype('dSEZ-feedback-1hop', dSEZ_feedback_1hop), ct.Celltype('dSEZ-efference-1hop', dSEZ_efference_1hop), 
        ct.Celltype('dSEZ-feedback-2hop', dSEZ_feedback_2hop), ct.Celltype('dSEZ-efference-2hop', dSEZ_efference_2hop),
        ct.Celltype('dSEZ-feedback-casc', dSEZ_feedback_casc), ct.Celltype('dSEZ-efference-casc', dSEZ_efference_casc)]
dSEZs_cts = ct.Celltype_Analyzer(dSEZs_cts)
dSEZs_cts.set_known_types(celltypes)
dSEZs_cts.memberships()

fig, ax = plt.subplots(1,1,figsize=(4,4))
sns.heatmap(data=dSEZs_cts.memberships()) 
plt.savefig('cascades/feedback_through_brain/plots/feedback-efference_dSEZ_celltypes.pdf', format='pdf', bbox_inches='tight')
'''
# split plots
import matplotlib as mpl

figsize=(.75,1.5)
vmax = 0.25

# modified cmaps for plots
cmap = plt.cm.get_cmap('Reds') # modify 'Reds' cmap to have a white background
red_cmap = cmap(np.linspace(0, 1, 20))
red_cmap[0] = np.array([1, 1, 1, 1])
red_cmap = mpl.colors.LinearSegmentedColormap.from_list(name='New_Reds', colors=red_cmap)

cmap = plt.cm.get_cmap('Oranges') # modify 'Oranges' cmap to have a white background
orange_cmap = cmap(np.linspace(0, 1, 20))
orange_cmap[0] = np.array([1, 1, 1, 1])
orange_cmap = mpl.colors.LinearSegmentedColormap.from_list(name='New_Oranges', colors=orange_cmap)

# dVNCs
dVNCs_cts = [ct.Celltype('dVNC-feedback-1hop', dVNC_feedback_1hop),  
        ct.Celltype('dVNC-feedback-2hop', dVNC_feedback_2hop), 
        ct.Celltype('dVNC-feedback-casc', dVNC_feedback_casc)]
dVNCs_cts = ct.Celltype_Analyzer(dVNCs_cts)
dVNCs_cts.set_known_types(celltypes)

fig, ax = plt.subplots(1,1,figsize=figsize)
sns.heatmap(data=dVNCs_cts.memberships().drop(labels=['sensories', 'ascendings'], axis=0), annot=True, fmt='.0%', cmap=orange_cmap, vmax=vmax) 
plt.savefig('cascades/feedback_through_brain/plots/feedback_dVNC_celltypes.pdf', format='pdf', bbox_inches='tight')

dVNCs_cts = [ct.Celltype('dVNC-feedback-1hop', dVNC_efference_1hop),  
        ct.Celltype('dVNC-feedback-2hop', dVNC_efference_2hop), 
        ct.Celltype('dVNC-feedback-casc', dVNC_efference_casc)]
dVNCs_cts = ct.Celltype_Analyzer(dVNCs_cts)
dVNCs_cts.set_known_types(celltypes)

fig, ax = plt.subplots(1,1,figsize=figsize)
sns.heatmap(data=dVNCs_cts.memberships().drop(labels=['sensories', 'ascendings'], axis=0), annot=True, fmt='.0%', cmap=red_cmap, vmax=vmax) 
plt.savefig('cascades/feedback_through_brain/plots/efference_dVNC_celltypes.pdf', format='pdf', bbox_inches='tight')

# dSEZs
dSEZs_cts = [ct.Celltype('dSEZ-feedback-1hop', dSEZ_feedback_1hop),  
        ct.Celltype('dSEZ-feedback-2hop', dSEZ_feedback_2hop), 
        ct.Celltype('dSEZ-feedback-casc', dSEZ_feedback_casc)]
dSEZs_cts = ct.Celltype_Analyzer(dSEZs_cts)
dSEZs_cts.set_known_types(celltypes)

fig, ax = plt.subplots(1,1,figsize=figsize)
sns.heatmap(data=dSEZs_cts.memberships().drop(labels=['sensories', 'ascendings'], axis=0), annot=True, fmt='.0%', cmap=orange_cmap, vmax=vmax) 
plt.savefig('cascades/feedback_through_brain/plots/feedback_dSEZ_celltypes.pdf', format='pdf', bbox_inches='tight')

dSEZs_cts = [ct.Celltype('dSEZ-feedback-1hop', dSEZ_efference_1hop),  
        ct.Celltype('dSEZ-feedback-2hop', dSEZ_efference_2hop), 
        ct.Celltype('dSEZ-feedback-casc', dSEZ_efference_casc)]
dSEZs_cts = ct.Celltype_Analyzer(dSEZs_cts)
dSEZs_cts.set_known_types(celltypes)

fig, ax = plt.subplots(1,1,figsize=figsize)
sns.heatmap(data=dSEZs_cts.memberships().drop(labels=['sensories', 'ascendings'], axis=0), annot=True, fmt='.0%', cmap=red_cmap, vmax=vmax) 
plt.savefig('cascades/feedback_through_brain/plots/efference_dSEZ_celltypes.pdf', format='pdf', bbox_inches='tight')

# %%
# plot memberships

dSEZs_cts = [ct.Celltype('dSEZ-feedback-1hop', dSEZ_feedback_1hop),
            ct.Celltype('dSEZ-feedback-2hop', dSEZ_feedback_2hop),
            ct.Celltype('dSEZ-feedback-casc', dSEZ_feedback_casc),
            ct.Celltype('dSEZ-efference-1hop', dSEZ_efference_1hop),
            ct.Celltype('dSEZ-efference-2hop', dSEZ_efference_2hop),
            ct.Celltype('dSEZ-efference-casc', dSEZ_efference_casc)]
dSEZs_cts = ct.Celltype_Analyzer(dSEZs_cts)
dSEZs_cts.set_known_types(celltypes)
dSEZs_cts.plot_memberships('cascades/feedback_through_brain/plots/feedback-efference_dSEZ_celltypes.pdf', figsize=(1,1))


dVNCs_cts = [ct.Celltype('dVNC-feedback-1hop', dVNC_feedback_1hop),
            ct.Celltype('dVNC-feedback-2hop', dVNC_feedback_2hop),
            ct.Celltype('dVNC-feedback-casc', dVNC_feedback_casc),
            ct.Celltype('dVNC-efference-1hop', dVNC_efference_1hop),
            ct.Celltype('dVNC-efference-2hop', dVNC_efference_2hop),
            ct.Celltype('dVNC-efference-casc', dVNC_efference_casc)]
dVNCs_cts = ct.Celltype_Analyzer(dVNCs_cts)
dVNCs_cts.set_known_types(celltypes)
dVNCs_cts.plot_memberships('cascades/feedback_through_brain/plots/feedback-efference_dVNC_celltypes.pdf', figsize=(1,1))

# %%
