#%%

from pymaid_creds import url, name, password, token
from data_settings import pairs_path, data_date
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

from contools import Celltype, Celltype_Analyzer, Promat, Cascade_Analyzer
import pickle 

n_init = 1000
cascades_df = pickle.load(open(f'data/cascades/all-brain-pairs-nonpaired_inputs-interneurons-outputs_{n_init}-n_init_{data_date}.p', 'rb'))
partners_df = cascades_df.loc[:, ['ds_partners_8hop', 'ds_partners_5hop']]

dVNC = pymaid.get_skids_by_annotation('mw dVNC')
dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')

pairs = Promat.get_pairs(pairs_path=pairs_path)

# %%
# identify ds_partners with cascades and how many are in recurrent loops

# identify downstream neurons receiving feedback or efference copy signal
def identify_fb_ec(partners_df, all_partners_df, threshold, hops, pairs, name, plot=True):

    feedback = []
    efference = []
    for pair_id in partners_df.index:
        ds_partners = partners_df.loc[pair_id, f'ds_partners_{hops}hop']
        ds_partners = np.intersect1d(ds_partners, all_partners_df.index) # collect only pair_ids

        feedback_partners = []
        efference_partners = []
        for ds in ds_partners:
            ds_partners2 = all_partners_df.loc[ds, f'ds_partners_{hops}hop']

            # recurrent partner, i.e. feedback
            if(pair_id in ds_partners2):
                feedback_partners.append(ds)

            # non-recurrent partner, i.e. efference copy
            if(pair_id not in ds_partners2):
                efference_partners.append(ds)

        feedback.append(feedback_partners)
        efference.append(efference_partners)

    # add left/right neurons
    pairs = pairs.copy()
    pairs.set_index('leftid', inplace=True)
    '''
    ds_partners_all = []
    for skid_list in partners_df.loc[:, f'ds_partners_{hops}hop']:
        skids_leftid = list(np.intersect1d(skid_list, pairs.index))
        skids_nonpaired = list(np.setdiff1d(skid_list, pairs.index))
        skids_rightid = list(pairs.loc[skids_leftid].rightid)

        ds_partners_all.append(np.unique(skids_leftid + skids_rightid + skids_nonpaired))
    '''
    feedback_all = []
    for skid_list in feedback:
        skids_leftid = list(np.intersect1d(skid_list, pairs.index))
        skids_nonpaired = list(np.setdiff1d(skid_list, pairs.index))
        skids_rightid = list(pairs.loc[skids_leftid].rightid)

        feedback_all.append(list(np.unique(skids_leftid + skids_rightid + skids_nonpaired)))

    efference_all = []
    for skid_list in efference:
        skids_leftid = list(np.intersect1d(skid_list, pairs.index))
        skids_nonpaired = list(np.setdiff1d(skid_list, pairs.index))
        skids_rightid = list(pairs.loc[skids_leftid].rightid)

        efference_all.append(list(np.unique(skids_leftid + skids_rightid + skids_nonpaired)))

    #partners_df[f'ds_partners_{hops}hop'] = ds_partners_all
    partners_df[f'feedback_partners_{hops}hop'] = feedback_all
    partners_df[f'efference_partners_{hops}hop'] = efference_all

    frac_feedback = [len(partners_df.loc[i, f'feedback_partners_{hops}hop'])/len(partners_df.loc[i, f'ds_partners_{hops}hop']) if len(partners_df.loc[i, f'ds_partners_{hops}hop'])>0 else 0 for i in partners_df.index]
    partners_df[f'fraction_feedback_partners_{hops}hop'] = frac_feedback
    frac_efference = [len(partners_df.loc[i, f'efference_partners_{hops}hop'])/len(partners_df.loc[i, f'ds_partners_{hops}hop']) if len(partners_df.loc[i, f'ds_partners_{hops}hop'])>0 else 0 for i in partners_df.index]
    partners_df[f'fraction_efference_partners_{hops}hop'] = frac_efference

    if(plot):
        # plot fraction feedback vs. efference
        data = partners_df.loc[:, [f'fraction_feedback_partners_{hops}hop', f'fraction_efference_partners_{hops}hop']]
        fig, ax = plt.subplots(1,1,figsize=(.5,.5))
        sns.barplot(data=data.loc[(data!=0).any(axis=1)]) # plot only dVNCs that actually talk to brain neurons
        ax.set(ylim=(0,1))
        plt.savefig(f'plots/feedback-vs-efference_ds-{name}_{hops}hop.pdf', format='pdf', bbox_inches='tight')

    return(partners_df)

n_init = 1000
threshold = n_init/2

hops = 8
dVNC_partners_df = partners_df.loc[np.intersect1d(dVNC, partners_df.index)]
dVNC_partners_df = identify_fb_ec(dVNC_partners_df, partners_df, threshold, hops, pairs, 'dVNC')

hops = 5
dVNC_partners_df = identify_fb_ec(dVNC_partners_df, partners_df, threshold, hops, pairs, 'dVNC')

hops = 8
dSEZ_partners_df = partners_df.loc[np.intersect1d(dSEZ, partners_df.index)]
dSEZ_partners_df = identify_fb_ec(dSEZ_partners_df, partners_df, threshold, hops, pairs, 'dSEZ')

hops = 5
dSEZ_partners_df = identify_fb_ec(dSEZ_partners_df, partners_df, threshold, hops, pairs, 'dSEZ')

# %%
# in text data

# for dVNCs
hops=5
average_upstream_feedback = np.mean([len(x) for x in dVNC_partners_df.loc[:, f'feedback_partners_{hops}hop'] if len(x)>0])
print(f'dVNCs provide feedback to an average of {average_upstream_feedback:.0f} neurons using {hops}-hop cascades')

# how many path lengths (mean+/-std) are between dVNCs and their feedback partners?
feedback_dVNCs = list(dVNC_partners_df[dVNC_partners_df.loc[:, f'fraction_feedback_partners_{hops}hop']!=0].index)

dVNC_cascade_df = cascades_df.loc[feedback_dVNCs]

n_init=1000
hit_thres = n_init/10

lengths = []
for dVNC in dVNC_cascade_df.index:
    hh_pairwise = dVNC_cascade_df.loc[dVNC, 'cascade_objs'].hh_pairwise
    feedback_partners = dVNC_partners_df.loc[dVNC, f'feedback_partners_{hops}hop']
    feedback_partners = list(np.intersect1d(feedback_partners, [x[1] for x in hh_pairwise.index])) # convert to pair_id

    # identify length of each path
    df_boolean = (hh_pairwise>hit_thres)

    lengths = []
    for skid in feedback_partners:
        lengths.append(list(df_boolean.columns[df_boolean.loc[(slice(None), skid), :].values[0]]))

dVNC_rpath_mean = np.mean([len(x) for x in lengths])
dVNC_rpath_std = np.std([len(x) for x in lengths])

print(f'There are {dVNC_rpath_mean:.1f}+/-{dVNC_rpath_std:.1f} paths of different lengths between dVNC to recurrent partners')


# for dSEZs
hops=5
average_upstream_feedback = np.mean([len(x) for x in dSEZ_partners_df.loc[:, f'feedback_partners_{hops}hop'] if len(x)>0])
print(f'dSEZs provide feedback to an average of {average_upstream_feedback:.0f} neurons using {hops}-hop cascades')

# how many path lengths (mean+/-std) are between dVNCs and their feedback partners?
feedback_dSEZs = list(dSEZ_partners_df[dSEZ_partners_df.loc[:, f'fraction_feedback_partners_{hops}hop']!=0].index)

dSEZ_cascade_df = cascades_df.loc[feedback_dSEZs]

n_init=1000
hit_thres = n_init/10

lengths = []
for dSEZ in dSEZ_cascade_df.index:
    hh_pairwise = dSEZ_cascade_df.loc[dSEZ, 'cascade_objs'].hh_pairwise
    feedback_partners = dSEZ_partners_df.loc[dSEZ, f'feedback_partners_{hops}hop']
    feedback_partners = list(np.intersect1d(feedback_partners, [x[1] for x in hh_pairwise.index])) # convert to pair_id

    # identify length of each path
    df_boolean = (hh_pairwise>hit_thres)

    lengths = []
    for skid in feedback_partners:
        lengths.append(list(df_boolean.columns[df_boolean.loc[(slice(None), skid), :].values[0]]))

dSEZ_rpath_mean = np.mean([len(x) for x in lengths])
dSEZ_rpath_std = np.std([len(x) for x in lengths])

print(f'There are {dSEZ_rpath_mean:.1f}+/-{dSEZ_rpath_std:.1f} paths of different lengths between dSEZ to recurrent partners')

# %%
# export dVNC/dSEZ feedback/efference cascade partners

dVNC_ds_feedback_partners = list(np.unique([x for sublist in dVNC_partners_df.feedback_partners_8hop for x in sublist]))
dVNC_ds_efference_partners = list(np.unique([x for sublist in dVNC_partners_df.efference_partners_8hop for x in sublist]))
dSEZ_ds_feedback_partners = list(np.unique([x for sublist in dVNC_partners_df.feedback_partners_8hop for x in sublist]))
dSEZ_ds_efference_partners = list(np.unique([x for sublist in dVNC_partners_df.efference_partners_8hop for x in sublist]))

pymaid.add_annotations(dVNC_ds_feedback_partners, 'mw dVNC ds-cascade_FB_8hop 2022-03-15')
pymaid.add_annotations(dVNC_ds_efference_partners, 'mw dVNC ds-cascade_EC_8hop 2022-03-15')
pymaid.add_annotations(dSEZ_ds_feedback_partners, 'mw dSEZ ds-cascade_FB_8hop 2022-03-15')
pymaid.add_annotations(dSEZ_ds_efference_partners, 'mw dSEZ ds-cascade_EC_8hop 2022-03-15')

# %%
# which celltypes receive feedback or efference signal?

_, celltypes = Celltype_Analyzer.default_celltypes()
hops = 8

# build Celltype() objects for different feedback and efference types
dVNC_feedback = [dVNC_partners_df.loc[ind, f'feedback_partners_{hops}hop'] for ind in dVNC_partners_df.index]
dVNC_feedback = list(np.unique([x for sublist in dVNC_feedback for x in sublist]))
dVNC_efference = [dVNC_partners_df.loc[ind, f'efference_partners_{hops}hop'] for ind in dVNC_partners_df.index]
dVNC_efference = list(np.unique([x for sublist in dVNC_efference for x in sublist]))
dSEZ_feedback = [dSEZ_partners_df.loc[ind, f'feedback_partners_{hops}hop'] for ind in dSEZ_partners_df.index]
dSEZ_feedback = list(np.unique([x for sublist in dSEZ_feedback for x in sublist]))
dSEZ_efference = [dSEZ_partners_df.loc[ind, f'efference_partners_{hops}hop'] for ind in dSEZ_partners_df.index]
dSEZ_efference = list(np.unique([x for sublist in dSEZ_efference for x in sublist]))

ds_cts = [Celltype('dVNC-feedback', dVNC_feedback), Celltype('dVNC-efference', dVNC_efference), Celltype('dSEZ-feedback', dSEZ_feedback), Celltype('dSEZ-efference', dSEZ_efference)]
ds_cts = Celltype_Analyzer(ds_cts)
ds_cts.set_known_types(celltypes)
ds_cts.memberships()

fig, ax = plt.subplots(1,1,figsize=(4,4))
sns.heatmap(data=ds_cts.memberships())
plt.savefig(f'plots/feedback-efference_{hops}hop_celltypes.pdf', format='pdf', bbox_inches='tight')

# %%
# identify directly 1-hop/2-hop downstream neurons: how many are feedback vs efference? (based on cascades)

# characterization of dVNC multihop downstream partners (using %input thresholds)
pairs = Promat.get_pairs(pairs_path=pairs_path)
hops = 8

select_neurons = pymaid.get_skids_by_annotation(['mw brain neurons', 'mw brain accessory neurons'])
ad_edges = Promat.pull_edges(type_edges='ad', threshold=0.01, data_date=data_date, pairs_combined=False, select_neurons=select_neurons)

dVNC_pairs = Promat.load_pairs_from_annotation('mw dVNC', pairs)
ds_dVNCs = [Promat.downstream_multihop(edges=ad_edges, sources=dVNC_pair, hops=5, pairs=pairs) for dVNC_pair in dVNC_pairs.values]

ds_dVNCs_df = pd.DataFrame(ds_dVNCs, columns=[f'dVNC-ds-{i}hop' for i in range(1,6)], index=dVNC_pairs.leftid)
ds_dVNCs_df = ds_dVNCs_df[[False if ds_dVNCs_df.loc[i, 'dVNC-ds-1hop']==[] else True for i in ds_dVNCs_df.index]]

# hop_level refers to %input thresholds, hops refers to cascade max hops
def split_fb_ec(ds_df, name, hop_level, hops, pairs_path):

    pairs = Promat.get_pairs(pairs_path=pairs_path)

    # identify neurons that are feedback (in feedforward cascade of max {hops} hops to {skid} DN)
    # identify neurons that are efference copy (not in feedforward cascade to {skid} DN)
    feedback_list = []
    efference_list = []
    for skid in ds_df.index:
        feedback = []
        efference = []

        ds_hop = ds_df.loc[skid, f'{name}-ds-{hop_level}hop']
        ds_hop_pairs = Promat.load_pairs_from_annotation(f'ds_{hop_level}hop', pairs, skids=ds_hop, use_skids=True).leftid
        
        for pair in ds_hop_pairs:
            ds_ds_pairs = partners_df.loc[pair, f'ds_partners_{hops}hop']
            if(skid in ds_ds_pairs):
                feedback.append(pair)
            if(skid not in ds_ds_pairs):
                efference.append(pair)

        feedback_list.append(feedback)
        efference_list.append(efference)

    # include both left and right skids from each pair in feedback/efference lists
    pairs.set_index('leftid', inplace=True)
    feedback_list_all = []
    for skid_list in feedback_list:
        skids_leftid = list(np.intersect1d(skid_list, pairs.index))
        skids_nonpaired = list(np.setdiff1d(skid_list, pairs.index))
        skids_rightid = list(pairs.loc[skids_leftid].rightid)

        feedback_list_all.append(skids_leftid + skids_rightid + skids_nonpaired)

    efference_list_all = []
    for skid_list in efference_list:
        skids_leftid = list(np.intersect1d(skid_list, pairs.index))
        skids_nonpaired = list(np.setdiff1d(skid_list, pairs.index))
        skids_rightid = list(pairs.loc[skids_leftid].rightid)

        efference_list_all.append(skids_leftid + skids_rightid + skids_nonpaired)

    # add skid pairs to appropriate column categories in ds_df
    ds_df[f'{name}-ds-{hop_level}hop_feedback'] = feedback_list_all
    ds_df[f'{name}-ds-{hop_level}hop_efference'] = efference_list_all
    return(ds_df)

ds_dVNCs_df = split_fb_ec(ds_dVNCs_df, 'dVNC', hop_level=1, hops=8, pairs_path=pairs_path)
ds_dVNCs_df = split_fb_ec(ds_dVNCs_df, 'dVNC', hop_level=2, hops=8, pairs_path=pairs_path)
ds_dVNCs_df = split_fb_ec(ds_dVNCs_df, 'dVNC', hop_level=3, hops=8, pairs_path=pairs_path)
ds_dVNCs_df = split_fb_ec(ds_dVNCs_df, 'dVNC', hop_level=4, hops=8, pairs_path=pairs_path)
ds_dVNCs_df = split_fb_ec(ds_dVNCs_df, 'dVNC', hop_level=5, hops=8, pairs_path=pairs_path)

# characterization of dSEZ multihop downstream partners
pairs = Promat.get_pairs(pairs_path=pairs_path)

select_neurons = pymaid.get_skids_by_annotation(['mw brain neurons', 'mw brain accessory neurons'])
ad_edges = Promat.pull_edges(type_edges='ad', threshold=0.01, data_date=data_date, pairs_combined=False, select_neurons=select_neurons)

dSEZ_pairs = Promat.load_pairs_from_annotation('mw dSEZ', pairs)
ds_dSEZs = [Promat.downstream_multihop(edges=ad_edges, sources=dSEZ_pair, hops=5, pairs=pairs) for dSEZ_pair in dSEZ_pairs.values]

ds_dSEZs_df = pd.DataFrame(ds_dSEZs, columns=[f'dSEZ-ds-{i}hop' for i in range(1,6)], index=dSEZ_pairs.leftid)
ds_dSEZs_df = ds_dSEZs_df[[False if ds_dSEZs_df.loc[i, 'dSEZ-ds-1hop']==[] else True for i in ds_dSEZs_df.index]]

ds_dSEZs_df = split_fb_ec(ds_dSEZs_df, 'dSEZ', hop_level=1, hops=8, pairs_path=pairs_path)
ds_dSEZs_df = split_fb_ec(ds_dSEZs_df, 'dSEZ', hop_level=2, hops=8, pairs_path=pairs_path)
ds_dSEZs_df = split_fb_ec(ds_dSEZs_df, 'dSEZ', hop_level=3, hops=8, pairs_path=pairs_path)
ds_dSEZs_df = split_fb_ec(ds_dSEZs_df, 'dSEZ', hop_level=4, hops=8, pairs_path=pairs_path)
ds_dSEZs_df = split_fb_ec(ds_dSEZs_df, 'dSEZ', hop_level=5, hops=8, pairs_path=pairs_path)

# %%
# fraction feedback vs efference
# needs to be updated

ds_dSEZs_df['fraction_feedback_1hop'] = [len(ds_dSEZs_df.loc[i, 'dSEZ-ds-1hop_feedback'])/(len(ds_dSEZs_df.loc[i, 'dSEZ-ds-1hop'])) if (len(ds_dSEZs_df.loc[i, 'dSEZ-ds-1hop']))>0 else 0 for i in ds_dSEZs_df.index]
ds_dSEZs_df['fraction_feedback_2hop'] = [len(ds_dSEZs_df.loc[i, 'dSEZ-ds-2hop_feedback'])/(len(ds_dSEZs_df.loc[i, 'dSEZ-ds-2hop'])) if (len(ds_dSEZs_df.loc[i, 'dSEZ-ds-2hop']))>0 else 0 for i in ds_dSEZs_df.index]
ds_dSEZs_df['fraction_efference_1hop'] = [len(ds_dSEZs_df.loc[i, 'dSEZ-ds-1hop_efference'])/(len(ds_dSEZs_df.loc[i, 'dSEZ-ds-1hop'])) if (len(ds_dSEZs_df.loc[i, 'dSEZ-ds-1hop']))>0 else 0 for i in ds_dSEZs_df.index]
ds_dSEZs_df['fraction_efference_2hop'] = [len(ds_dSEZs_df.loc[i, 'dSEZ-ds-2hop_efference'])/(len(ds_dSEZs_df.loc[i, 'dSEZ-ds-2hop'])) if (len(ds_dSEZs_df.loc[i, 'dSEZ-ds-2hop']))>0 else 0 for i in ds_dSEZs_df.index]

ds_dVNCs_df['fraction_feedback_1hop'] = [len(ds_dVNCs_df.loc[i, 'dVNC-ds-1hop_feedback'])/(len(ds_dVNCs_df.loc[i, 'dVNC-ds-1hop'])) if (len(ds_dVNCs_df.loc[i, 'dVNC-ds-1hop']))>0 else 0 for i in ds_dVNCs_df.index]
ds_dVNCs_df['fraction_feedback_2hop'] = [len(ds_dVNCs_df.loc[i, 'dVNC-ds-2hop_feedback'])/(len(ds_dVNCs_df.loc[i, 'dVNC-ds-2hop'])) if (len(ds_dVNCs_df.loc[i, 'dVNC-ds-2hop']))>0 else 0 for i in ds_dVNCs_df.index]
ds_dVNCs_df['fraction_efference_1hop'] = [len(ds_dVNCs_df.loc[i, 'dVNC-ds-1hop_efference'])/(len(ds_dVNCs_df.loc[i, 'dVNC-ds-1hop'])) if (len(ds_dVNCs_df.loc[i, 'dVNC-ds-1hop']))>0 else 0 for i in ds_dVNCs_df.index]
ds_dVNCs_df['fraction_efference_2hop'] = [len(ds_dVNCs_df.loc[i, 'dVNC-ds-2hop_efference'])/(len(ds_dVNCs_df.loc[i, 'dVNC-ds-2hop'])) if (len(ds_dVNCs_df.loc[i, 'dVNC-ds-2hop']))>0 else 0 for i in ds_dVNCs_df.index]

# plot
fig, ax = plt.subplots(1,1,figsize=(4,4))
data = ds_dVNCs_df.loc[:, ['fraction_feedback_1hop', 'fraction_efference_1hop', 'fraction_feedback_2hop', 'fraction_efference_2hop']]
sns.barplot(data=data)
plt.savefig('plots/feedback-efference_ds-dVNCs.pdf', format='pdf', bbox_inches='tight', ylim=(0,1))

fig, ax = plt.subplots(1,1,figsize=(4,4))
data = ds_dSEZs_df.loc[:, ['fraction_feedback_1hop', 'fraction_efference_1hop', 'fraction_feedback_2hop', 'fraction_efference_2hop']]
sns.barplot(data=data)
plt.savefig('plots/feedback-efference_ds-dSEZs.pdf', format='pdf', bbox_inches='tight', ylim=(0,1))

# %%
# in text data

# for dVNCs
hops=1
average_upstream_feedback = np.mean([len(x) for x in ds_dVNCs_df.loc[:, f'dVNC-ds-{hops}hop'] if len(x)>0])
print(f'dVNCs provide feedback to an average of {average_upstream_feedback:.0f} neurons in {hops}-hops')

hops=2
average_upstream_feedback = np.mean([len(x) for x in ds_dVNCs_df.loc[:, f'dVNC-ds-{hops}hop'] if len(x)>0])
print(f'dVNCs provide feedback to an average of {average_upstream_feedback:.0f} neurons in {hops}-hops')

# for dSEZs
hops=1
average_upstream_feedback = np.mean([len(x) for x in ds_dSEZs_df.loc[:, f'dSEZ-ds-{hops}hop'] if len(x)>0])
print(f'dVNCs provide feedback to an average of {average_upstream_feedback:.0f} neurons in {hops}-hops')

hops=2
average_upstream_feedback = np.mean([len(x) for x in ds_dSEZs_df.loc[:, f'dSEZ-ds-{hops}hop'] if len(x)>0])
print(f'dVNCs provide feedback to an average of {average_upstream_feedback:.0f} neurons in {hops}-hops')


# %%
# cell types within 1-hop/2-hop feedback/efference categories

_, celltypes = Celltype_Analyzer.default_celltypes()
hops = 8

# build Celltype() objects for different feedback and efference types
dVNC_feedback_casc = [dVNC_partners_df.loc[ind, f'feedback_partners_{hops}hop'] for ind in dVNC_partners_df.index]
dVNC_feedback_casc = list(np.unique([x for sublist in dVNC_feedback_casc for x in sublist]))
dVNC_efference_casc = [dVNC_partners_df.loc[ind, f'efference_partners_{hops}hop'] for ind in dVNC_partners_df.index]
dVNC_efference_casc = list(np.unique([x for sublist in dVNC_efference_casc for x in sublist]))
dSEZ_feedback_casc = [dSEZ_partners_df.loc[ind, f'feedback_partners_{hops}hop'] for ind in dSEZ_partners_df.index]
dSEZ_feedback_casc = list(np.unique([x for sublist in dSEZ_feedback_casc for x in sublist]))
dSEZ_efference_casc = [dSEZ_partners_df.loc[ind, f'efference_partners_{hops}hop'] for ind in dSEZ_partners_df.index]
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
dVNCs_cts = [Celltype('dVNC-feedback-1hop', dVNC_feedback_1hop), Celltype('dVNC-efference-1hop', dVNC_efference_1hop), 
        Celltype('dVNC-feedback-2hop', dVNC_feedback_2hop), Celltype('dVNC-efference-2hop', dVNC_efference_2hop),
        Celltype('dVNC-feedback-casc', dVNC_feedback_casc), Celltype('dVNC-efference-casc', dVNC_efference_casc)]
dVNCs_cts = Celltype_Analyzer(dVNCs_cts)
dVNCs_cts.set_known_types(celltypes)
dVNCs_cts.memberships()

fig, ax = plt.subplots(1,1,figsize=(4,4))
sns.heatmap(data=dVNCs_cts.memberships()) 
plt.savefig('cascades/feedback_through_brain/plots/feedback-efference_dVNC_celltypes.pdf', format='pdf', bbox_inches='tight')

dSEZs_cts = [Celltype('dSEZ-feedback-1hop', dSEZ_feedback_1hop), Celltype('dSEZ-efference-1hop', dSEZ_efference_1hop), 
        Celltype('dSEZ-feedback-2hop', dSEZ_feedback_2hop), Celltype('dSEZ-efference-2hop', dSEZ_efference_2hop),
        Celltype('dSEZ-feedback-casc', dSEZ_feedback_casc), Celltype('dSEZ-efference-casc', dSEZ_efference_casc)]
dSEZs_cts = Celltype_Analyzer(dSEZs_cts)
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
dVNCs_cts = [Celltype('dVNC-feedback-1hop', dVNC_feedback_1hop),  
        Celltype('dVNC-feedback-2hop', dVNC_feedback_2hop), 
        Celltype('dVNC-feedback-casc', dVNC_feedback_casc)]
dVNCs_cts = Celltype_Analyzer(dVNCs_cts)
dVNCs_cts.set_known_types(celltypes)

fig, ax = plt.subplots(1,1,figsize=figsize)
sns.heatmap(data=dVNCs_cts.memberships().drop(labels=['sensories', 'ascendings'], axis=0), annot=True, fmt='.0%', cmap=orange_cmap, vmax=vmax) 
plt.savefig('plots/feedback_dVNC_celltypes.pdf', format='pdf', bbox_inches='tight')

dVNCs_cts = [Celltype('dVNC-feedback-1hop', dVNC_efference_1hop),  
        Celltype('dVNC-feedback-2hop', dVNC_efference_2hop), 
        Celltype('dVNC-feedback-casc', dVNC_efference_casc)]
dVNCs_cts = Celltype_Analyzer(dVNCs_cts)
dVNCs_cts.set_known_types(celltypes)

fig, ax = plt.subplots(1,1,figsize=figsize)
sns.heatmap(data=dVNCs_cts.memberships().drop(labels=['sensories', 'ascendings'], axis=0), annot=True, fmt='.0%', cmap=red_cmap, vmax=vmax) 
plt.savefig('plots/efference_dVNC_celltypes.pdf', format='pdf', bbox_inches='tight')

# dSEZs
dSEZs_cts = [Celltype('dSEZ-feedback-1hop', dSEZ_feedback_1hop),  
        Celltype('dSEZ-feedback-2hop', dSEZ_feedback_2hop), 
        Celltype('dSEZ-feedback-casc', dSEZ_feedback_casc)]
dSEZs_cts = Celltype_Analyzer(dSEZs_cts)
dSEZs_cts.set_known_types(celltypes)

fig, ax = plt.subplots(1,1,figsize=figsize)
sns.heatmap(data=dSEZs_cts.memberships().drop(labels=['sensories', 'ascendings'], axis=0), annot=True, fmt='.0%', cmap=orange_cmap, vmax=vmax) 
plt.savefig('plots/feedback_dSEZ_celltypes.pdf', format='pdf', bbox_inches='tight')

dSEZs_cts = [Celltype('dSEZ-feedback-1hop', dSEZ_efference_1hop),  
        Celltype('dSEZ-feedback-2hop', dSEZ_efference_2hop), 
        Celltype('dSEZ-feedback-casc', dSEZ_efference_casc)]
dSEZs_cts = Celltype_Analyzer(dSEZs_cts)
dSEZs_cts.set_known_types(celltypes)

fig, ax = plt.subplots(1,1,figsize=figsize)
sns.heatmap(data=dSEZs_cts.memberships().drop(labels=['sensories', 'ascendings'], axis=0), annot=True, fmt='.0%', cmap=red_cmap, vmax=vmax) 
plt.savefig('plots/efference_dSEZ_celltypes.pdf', format='pdf', bbox_inches='tight')

# %%
# 1-5 hops downstream of dVNCs or dSEZs in brain (feedback vs efference)

_, celltypes = Celltype_Analyzer.default_celltypes()

# build Celltype() objects for different feedback and efference types (1-hop)

dVNC_feedback_all_hops = []
dVNC_efference_all_hops = []
for hop_level in range(1,6):
    dVNC_feedback_hop = [ds_dVNCs_df.loc[ind, f'dVNC-ds-{hop_level}hop_feedback'] for ind in ds_dVNCs_df.index]
    dVNC_feedback_hop = list(np.unique([x for sublist in dVNC_feedback_hop for x in sublist]))
    dVNC_efference_hop = [ds_dVNCs_df.loc[ind, f'dVNC-ds-{hop_level}hop_efference'] for ind in ds_dVNCs_df.index]
    dVNC_efference_hop = list(np.unique([x for sublist in dVNC_efference_hop for x in sublist]))

    dVNC_feedback_all_hops.append(dVNC_feedback_hop)
    dVNC_efference_all_hops.append(dVNC_efference_hop)

dSEZ_feedback_all_hops = []
dSEZ_efference_all_hops = []
for hop_level in range(1,6):
    dSEZ_feedback_hop = [ds_dSEZs_df.loc[ind, f'dSEZ-ds-{hop_level}hop_feedback'] for ind in ds_dSEZs_df.index]
    dSEZ_feedback_hop = list(np.unique([x for sublist in dSEZ_feedback_hop for x in sublist]))
    dSEZ_efference_hop = [ds_dSEZs_df.loc[ind, f'dSEZ-ds-{hop_level}hop_efference'] for ind in ds_dSEZs_df.index]
    dSEZ_efference_hop = list(np.unique([x for sublist in dSEZ_efference_hop for x in sublist]))

    dSEZ_feedback_all_hops.append(dSEZ_feedback_hop)
    dSEZ_efference_all_hops.append(dSEZ_efference_hop)

# plot celltypes at each hop_level
import matplotlib as mpl

figsize=(1.25,1.5)
vmax = 0.25
# dVNCs
dVNCs_cts = [Celltype(f'dVNC-feedback-{hop_level+1}hop', dVNC_feedback_all_hops[hop_level]) for hop_level in range(0,5)]
dVNCs_cts = Celltype_Analyzer(dVNCs_cts)
dVNCs_cts.set_known_types(celltypes)

fig, ax = plt.subplots(1,1,figsize=figsize)
sns.heatmap(data=dVNCs_cts.memberships().drop(labels=['sensories', 'ascendings'], axis=0), annot=True, fmt='.0%', cmap=orange_cmap, vmax=vmax) 
plt.savefig('plots/feedback_hops1-5_dVNC_celltypes.pdf', format='pdf', bbox_inches='tight')

dVNCs_cts = [Celltype(f'dVNC-efference-{hop_level+1}hop', dVNC_efference_all_hops[hop_level]) for hop_level in range(0,5)]
dVNCs_cts = Celltype_Analyzer(dVNCs_cts)
dVNCs_cts.set_known_types(celltypes)

fig, ax = plt.subplots(1,1,figsize=figsize)
sns.heatmap(data=dVNCs_cts.memberships().drop(labels=['sensories', 'ascendings'], axis=0), annot=True, fmt='.0%', cmap=red_cmap, vmax=vmax) 
plt.savefig('plots/efference_hops1-5_dVNC_celltypes.pdf', format='pdf', bbox_inches='tight')

# dSEZs
dSEZs_cts = [Celltype(f'dSEZ-feedback-{hop_level+1}hop', dSEZ_feedback_all_hops[hop_level]) for hop_level in range(0,5)]
dSEZs_cts = Celltype_Analyzer(dSEZs_cts)
dSEZs_cts.set_known_types(celltypes)

fig, ax = plt.subplots(1,1,figsize=figsize)
sns.heatmap(data=dSEZs_cts.memberships().drop(labels=['sensories', 'ascendings'], axis=0), annot=True, fmt='.0%', cmap=orange_cmap, vmax=vmax) 
plt.savefig('plots/feedback_hops1-5_dSEZ_celltypes.pdf', format='pdf', bbox_inches='tight')

dSEZs_cts = [Celltype(f'dSEZ-efference-{hop_level+1}hop', dSEZ_efference_all_hops[hop_level]) for hop_level in range(0,5)]
dSEZs_cts = Celltype_Analyzer(dSEZs_cts)
dSEZs_cts.set_known_types(celltypes)

fig, ax = plt.subplots(1,1,figsize=figsize)
sns.heatmap(data=dSEZs_cts.memberships().drop(labels=['sensories', 'ascendings'], axis=0), annot=True, fmt='.0%', cmap=red_cmap, vmax=vmax) 
plt.savefig('plots/efference_hops1-5_dSEZ_celltypes.pdf', format='pdf', bbox_inches='tight')


# %%
# plot memberships

dSEZs_cts = [Celltype('dSEZ-feedback-1hop', dSEZ_feedback_all_hops[0]),
            Celltype('dSEZ-feedback-2hop', dSEZ_feedback_all_hops[1]),
            Celltype('dSEZ-feedback-3hop', dSEZ_feedback_all_hops[2]),
            Celltype('dSEZ-feedback-4hop', dSEZ_feedback_all_hops[3]),
            Celltype('dSEZ-feedback-5hop', dSEZ_feedback_all_hops[4]),
            Celltype('dSEZ-feedback-casc', dSEZ_feedback_casc),
            Celltype('dSEZ-efference-1hop', dSEZ_efference_all_hops[0]),
            Celltype('dSEZ-efference-2hop', dSEZ_efference_all_hops[1]),
            Celltype('dSEZ-efference-3hop', dSEZ_efference_all_hops[2]),
            Celltype('dSEZ-efference-4hop', dSEZ_efference_all_hops[3]),
            Celltype('dSEZ-efference-5hop', dSEZ_efference_all_hops[4]),
            Celltype('dSEZ-efference-casc', dSEZ_efference_casc)]
dSEZs_cts = Celltype_Analyzer(dSEZs_cts)
dSEZs_cts.set_known_types(celltypes)
dSEZs_cts.plot_memberships('plots/feedback-efference_dSEZ_celltypes.pdf', figsize=(1,1), ylim=(0,1))
dSEZs_cts.plot_memberships('plots/feedback-efference_dSEZ_celltypes_raw.pdf', figsize=(1,1), raw_num=True)


dVNCs_cts = [Celltype('dVNC-feedback-1hop', dVNC_feedback_all_hops[0]),
            Celltype('dVNC-feedback-2hop', dVNC_feedback_all_hops[1]),
            Celltype('dVNC-feedback-3hop', dVNC_feedback_all_hops[2]),
            Celltype('dVNC-feedback-4hop', dVNC_feedback_all_hops[3]),
            Celltype('dVNC-feedback-5hop', dVNC_feedback_all_hops[4]),
            Celltype('dVNC-feedback-casc', dVNC_feedback_casc),
            Celltype('dVNC-efference-1hop', dVNC_efference_all_hops[0]),
            Celltype('dVNC-efference-2hop', dVNC_efference_all_hops[1]),
            Celltype('dVNC-efference-3hop', dVNC_efference_all_hops[2]),
            Celltype('dVNC-efference-4hop', dVNC_efference_all_hops[3]),
            Celltype('dVNC-efference-5hop', dVNC_efference_all_hops[4]),
            Celltype('dVNC-efference-casc', dVNC_efference_casc)]
dVNCs_cts = Celltype_Analyzer(dVNCs_cts)
dVNCs_cts.set_known_types(celltypes)
dVNCs_cts.plot_memberships('plots/feedback-efference_dVNC_celltypes.pdf', figsize=(1,1), ylim=(0,1))
dVNCs_cts.plot_memberships('plots/feedback-efference_dVNC_celltypes_raw.pdf', figsize=(1,1), raw_num=True)

# %%
# export cell types

annots = ['mw ' + x.name + ' 2022-03-15' for x in dVNCs_cts.Celltypes]
[pymaid.add_annotations(dVNCs_cts.Celltypes[i].skids, annots[i]) for i in range(len(dVNCs_cts.Celltypes))]

annots = ['mw ' + x.name + ' 2022-03-15' for x in dSEZs_cts.Celltypes]
[pymaid.add_annotations(dSEZs_cts.Celltypes[i].skids, annots[i]) for i in range(len(dSEZs_cts.Celltypes))]

# %%
