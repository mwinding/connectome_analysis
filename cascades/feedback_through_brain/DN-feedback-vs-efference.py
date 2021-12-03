#%%
import sys
import os

os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory

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
fig, ax = plt.subplots(1,1,figsize=(4,4))
sns.barplot(data=data.loc[(data!=0).any(axis=1)]) # plot only dVNCs that actually take to brain neurons
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

frac_feedback = [len(dSEZ_ds_partners_df.loc[i, 'feedback_partners'])/len(dSEZ_ds_partners_df.loc[i, 'ds_partners']) if len(dSEZ_ds_partners_df.loc[i, 'ds_partners'])>0 else 0 for i in dSEZ_ds_partners_df.index]
dSEZ_ds_partners_df['fraction_feedback_partners'] = frac_feedback
frac_efference = [len(dSEZ_ds_partners_df.loc[i, 'efference_partners'])/len(dSEZ_ds_partners_df.loc[i, 'ds_partners']) if len(dSEZ_ds_partners_df.loc[i, 'ds_partners'])>0 else 0 for i in dSEZ_ds_partners_df.index]
dSEZ_ds_partners_df['fraction_efference_partners'] = frac_efference

# plot fraction feedback vs. efference
data = dSEZ_ds_partners_df.loc[:, ['fraction_feedback_partners', 'fraction_efference_partners']]
fig, ax = plt.subplots(1,1,figsize=(4,4))
sns.barplot(data=data.loc[(data!=0).any(axis=1)]) # plot only dSEZs that actually take to brain neurons
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
# identify directly downstream neurons: how many are feedback vs efference? (based on cascades)
adj_ad = pm.Promat.pull_adj(type_adj='ad', subgraph='brain and accesssory')

ds_dVNC = 