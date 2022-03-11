#%%

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

# add left/right neurons
pairs = pm.Promat.get_pairs()
pairs.set_index('leftid', inplace=True)

dVNC_ds_partners_all = []
for skid_list in dVNC_ds_partners_df.ds_partners:
    skids_leftid = list(np.intersect1d(skid_list, pairs.index))
    skids_nonpaired = list(np.setdiff1d(skid_list, pairs.index))
    skids_rightid = list(pairs.loc[skids_leftid].rightid)

    dVNC_ds_partners_all.append(skids_leftid + skids_rightid + skids_nonpaired)

dVNC_feedback_all = []
for skid_list in dVNC_feedback:
    skids_leftid = list(np.intersect1d(skid_list, pairs.index))
    skids_nonpaired = list(np.setdiff1d(skid_list, pairs.index))
    skids_rightid = list(pairs.loc[skids_leftid].rightid)

    dVNC_feedback_all.append(skids_leftid + skids_rightid + skids_nonpaired)

dVNC_efference_all = []
for skid_list in dVNC_efference:
    skids_leftid = list(np.intersect1d(skid_list, pairs.index))
    skids_nonpaired = list(np.setdiff1d(skid_list, pairs.index))
    skids_rightid = list(pairs.loc[skids_leftid].rightid)

    dVNC_efference_all.append(skids_leftid + skids_rightid + skids_nonpaired)

dVNC_ds_partners_df['ds_partners'] = dVNC_ds_partners_all
dVNC_ds_partners_df['feedback_partners'] = dVNC_feedback_all
dVNC_ds_partners_df['efference_partners'] = dVNC_efference_all

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

# add left/right neurons
pairs = pm.Promat.get_pairs()
pairs.set_index('leftid', inplace=True)

dSEZ_ds_partners_all = []
for skid_list in dSEZ_ds_partners_df.ds_partners:
    skids_leftid = list(np.intersect1d(skid_list, pairs.index))
    skids_nonpaired = list(np.setdiff1d(skid_list, pairs.index))
    skids_rightid = list(pairs.loc[skids_leftid].rightid)

    dSEZ_ds_partners_all.append(skids_leftid + skids_rightid + skids_nonpaired)

dSEZ_feedback_all = []
for skid_list in dSEZ_feedback:
    skids_leftid = list(np.intersect1d(skid_list, pairs.index))
    skids_nonpaired = list(np.setdiff1d(skid_list, pairs.index))
    skids_rightid = list(pairs.loc[skids_leftid].rightid)

    dSEZ_feedback_all.append(skids_leftid + skids_rightid + skids_nonpaired)

dSEZ_efference_all = []
for skid_list in dSEZ_efference:
    skids_leftid = list(np.intersect1d(skid_list, pairs.index))
    skids_nonpaired = list(np.setdiff1d(skid_list, pairs.index))
    skids_rightid = list(pairs.loc[skids_leftid].rightid)

    dSEZ_efference_all.append(skids_leftid + skids_rightid + skids_nonpaired)

dSEZ_ds_partners_df['ds_partners'] = dSEZ_ds_partners_all
dSEZ_ds_partners_df['feedback_partners'] = dSEZ_feedback_all
dSEZ_ds_partners_df['efference_partners'] = dSEZ_efference_all

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
# in text data
average_upstream_feedback = np.mean([len(x) for x in dVNC_ds_partners_df.feedback_partners if len(x)>0])
print(f'dVNCs provide feedback to an average of {average_upstream_feedback:.0f} neurons using 8-hop cascades')

# how many path lengths (mean+/-std) are between dVNCs and their feedback partners?
feedback_dVNCs = list(dVNC_ds_partners_df[dVNC_ds_partners_df.fraction_feedback_partners!=0].index)

cascade_df = pd.DataFrame(pair_hist_list, columns = ['cascade'], index = [x.name for x in pair_hist_list])
dVNC_cascade_df = cascade_df.loc[feedback_dVNCs]

n_init=1000
hit_thres = n_init/10

lengths = []
for dVNC in dVNC_cascade_df.index:
    skid_hit_hist = dVNC_cascade_df.loc[dVNC, 'cascade'].skid_hit_hist
    feedback_partners = dVNC_ds_partners_df.loc[dVNC, 'feedback_partners']

    # identify length of each path
    df_boolen = (skid_hit_hist>hit_thres)

    lengths = []
    for skid in feedback_partners:
        lengths.append(list(df_boolen.columns[df_boolen.loc[skid]]))

dVNC_rpath_mean = np.mean([len(x) for x in lengths])
dVNC_rpath_std = np.std([len(x) for x in lengths])

print(f'There are {dVNC_rpath_mean:.1f}+/-{dVNC_rpath_std:.1f} paths of different lengths between dVNC to recurrent partners')

# %%
# export dVNC/dSEZ feedback/efference cascade partners

dVNC_ds_feedback_partners = list(np.unique([x for sublist in dVNC_ds_partners_df.feedback_partners for x in sublist]))
dVNC_ds_efference_partners = list(np.unique([x for sublist in dVNC_ds_partners_df.efference_partners for x in sublist]))
dSEZ_ds_feedback_partners = list(np.unique([x for sublist in dSEZ_ds_partners_df.feedback_partners for x in sublist]))
dSEZ_ds_efference_partners = list(np.unique([x for sublist in dSEZ_ds_partners_df.efference_partners for x in sublist]))

pymaid.add_annotations(dVNC_ds_feedback_partners, 'mw dVNC ds-cascade_FB 2022-03-10')
pymaid.add_annotations(dVNC_ds_efference_partners, 'mw dVNC ds-cascade_EC 2022-03-10')
pymaid.add_annotations(dSEZ_ds_feedback_partners, 'mw dSEZ ds-cascade_FB 2022-03-10')
pymaid.add_annotations(dSEZ_ds_efference_partners, 'mw dSEZ ds-cascade_EC 2022-03-10')

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


# add left/right neurons
pairs = pm.Promat.get_pairs()
pairs.set_index('leftid', inplace=True)

feedback_1hop_list_all = []
for skid_list in feedback_1hop_list:
    skids_leftid = list(np.intersect1d(skid_list, pairs.index))
    skids_nonpaired = list(np.setdiff1d(skid_list, pairs.index))
    skids_rightid = list(pairs.loc[skids_leftid].rightid)

    feedback_1hop_list_all.append(skids_leftid + skids_rightid + skids_nonpaired)

feedback_2hop_list_all = []
for skid_list in feedback_2hop_list:
    skids_leftid = list(np.intersect1d(skid_list, pairs.index))
    skids_nonpaired = list(np.setdiff1d(skid_list, pairs.index))
    skids_rightid = list(pairs.loc[skids_leftid].rightid)

    feedback_2hop_list_all.append(skids_leftid + skids_rightid + skids_nonpaired)

efference_1hop_list_all = []
for skid_list in efference_1hop_list:
    skids_leftid = list(np.intersect1d(skid_list, pairs.index))
    skids_nonpaired = list(np.setdiff1d(skid_list, pairs.index))
    skids_rightid = list(pairs.loc[skids_leftid].rightid)

    efference_1hop_list_all.append(skids_leftid + skids_rightid + skids_nonpaired)

efference_2hop_list_all = []
for skid_list in efference_2hop_list:
    skids_leftid = list(np.intersect1d(skid_list, pairs.index))
    skids_nonpaired = list(np.setdiff1d(skid_list, pairs.index))
    skids_rightid = list(pairs.loc[skids_leftid].rightid)

    efference_2hop_list_all.append(skids_leftid + skids_rightid + skids_nonpaired)

ds_dVNCs_df['dVNC-ds-1hop_feedback'] = feedback_1hop_list_all
ds_dVNCs_df['dVNC-ds-2hop_feedback'] = feedback_2hop_list_all
ds_dVNCs_df['dVNC-ds-1hop_efference'] = efference_1hop_list_all
ds_dVNCs_df['dVNC-ds-2hop_efference'] = efference_2hop_list_all


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

# add left/right neurons
pairs = pm.Promat.get_pairs()
pairs.set_index('leftid', inplace=True)

feedback_1hop_list_all = []
for skid_list in feedback_1hop_list:
    skids_leftid = list(np.intersect1d(skid_list, pairs.index))
    skids_nonpaired = list(np.setdiff1d(skid_list, pairs.index))
    skids_rightid = list(pairs.loc[skids_leftid].rightid)

    feedback_1hop_list_all.append(skids_leftid + skids_rightid + skids_nonpaired)

feedback_2hop_list_all = []
for skid_list in feedback_2hop_list:
    skids_leftid = list(np.intersect1d(skid_list, pairs.index))
    skids_nonpaired = list(np.setdiff1d(skid_list, pairs.index))
    skids_rightid = list(pairs.loc[skids_leftid].rightid)

    feedback_2hop_list_all.append(skids_leftid + skids_rightid + skids_nonpaired)

efference_1hop_list_all = []
for skid_list in efference_1hop_list:
    skids_leftid = list(np.intersect1d(skid_list, pairs.index))
    skids_nonpaired = list(np.setdiff1d(skid_list, pairs.index))
    skids_rightid = list(pairs.loc[skids_leftid].rightid)

    efference_1hop_list_all.append(skids_leftid + skids_rightid + skids_nonpaired)

efference_2hop_list_all = []
for skid_list in efference_2hop_list:
    skids_leftid = list(np.intersect1d(skid_list, pairs.index))
    skids_nonpaired = list(np.setdiff1d(skid_list, pairs.index))
    skids_rightid = list(pairs.loc[skids_leftid].rightid)

    efference_2hop_list_all.append(skids_leftid + skids_rightid + skids_nonpaired)

ds_dSEZs_df['dSEZ-ds-1hop_feedback'] = feedback_1hop_list_all
ds_dSEZs_df['dSEZ-ds-2hop_feedback'] = feedback_2hop_list_all
ds_dSEZs_df['dSEZ-ds-1hop_efference'] = efference_1hop_list_all
ds_dSEZs_df['dSEZ-ds-2hop_efference'] = efference_2hop_list_all

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
# export cell types

annots = ['mw ' + x.name + ' 2022-03-10' for x in dVNCs_cts.Celltypes]
[pymaid.add_annotations(dVNCs_cts.Celltypes[i].skids, annots[i]) for i in range(len(dVNCs_cts.Celltypes))]

annots = ['mw ' + x.name + ' 2022-03-10' for x in dSEZs_cts.Celltypes]
[pymaid.add_annotations(dSEZs_cts.Celltypes[i].skids, annots[i]) for i in range(len(dSEZs_cts.Celltypes))]

# %%
