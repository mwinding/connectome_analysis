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
import connectome_tools.cluster_analysis as clust

import pickle 

n_init = 1000
pair_hist_list = pickle.load(open(f'data/cascades/all-brain-pairs_outputs-added_{n_init}-n_init.p', 'rb'))

# %%
# identify ds_partners
from tqdm import tqdm
from joblib import Parallel, delayed

threshold = n_init/2
hops = 8
pairs = pm.Promat.get_pairs()

'''
def find_ds_partner_ids(pair_hist, pair_hist_list, pairs, hops):
    ds_partners = list(pair_hist.pairwise_threshold(threshold=threshold, hops=hops))
    pair_ids = pm.Promat.load_pairs_from_annotation('ds', pairs, skids=ds_partners, use_skids=True)
    pair_ids = list(pair_ids.leftid)
    pair_ids = list(np.intersect1d(pair_ids, [x.name for x in pair_hist_list]))
    return(pair_ids)

ds_partners = Parallel(n_jobs=-1)(delayed(find_ds_partner_ids)(pair_hist_list[i], pair_hist_list, pairs, hops) for i in tqdm(range(len(pair_hist_list))))
pickle.dump(ds_partners, open(f'data/cascades/all-brain-pairs_ds_partners_{n_init}-n_init.p', 'wb'))
'''

ds_partners = pickle.load(open(f'data/cascades/all-brain-pairs_ds_partners_{n_init}-n_init.p', 'rb'))
ds_partners_df = pd.DataFrame(list(map(lambda x: [x[0], x[1]], zip([x.name for x in pair_hist_list], ds_partners))), columns=['skid', 'ds_partners'])
ds_partners_df.set_index('skid', inplace=True)
# %%
# how many partners are in recurrent loops?

# collect all recurrent skids
recurrent_partners_col = []
for skid in ds_partners_df.index:
    recurrent_partners = []
    ds_partners = ds_partners_df.loc[skid, 'ds_partners']
    for partner in ds_partners:
        ds_ds_partners = ds_partners_df.loc[partner, 'ds_partners']
        if(skid in ds_ds_partners):
            recurrent_partners.append(partner)

    recurrent_partners_col.append(recurrent_partners)

# fraction of recurrent vs. nonrecurrent parents
ds_partners_df['recurrent_partners'] = recurrent_partners_col
frac_recurrent = [len(ds_partners_df.loc[i, 'recurrent_partners'])/len(ds_partners_df.loc[i, 'ds_partners']) if len(ds_partners_df.loc[i, 'ds_partners'])>0 else 0 for i in ds_partners_df.index]
ds_partners_df['fraction_recurrent_partners'] = frac_recurrent
ds_partners_df['fraction_nonrecurrent_partners'] = 1-ds_partners_df.fraction_recurrent_partners

#ds_partners_df = ds_partners_df[[False if x==[] else True for x in ds_partners_df.ds_partners]]

# plot total number of recurrent neurons
fig, ax = plt.subplots(1,1,figsize=(.5,1))
data = [sum(ds_partners_df.fraction_recurrent_partners==0)/len(ds_partners_df.index), sum(ds_partners_df.fraction_recurrent_partners!=0)/len(ds_partners_df.index)]
sns.barplot(x=['Non-recurrent Neurons', 'Recurrent Neurons'] , y=data, ax=ax)
ax.set(ylim=(0,1))
plt.savefig('cascades/feedback_through_brain/plots/recurrent-vs-nonrecurrent_fractions.pdf', format='pdf', bbox_inches='tight')

# boxplot of data with points
fig, ax = plt.subplots(1,1,figsize=(2,4))
data = ds_partners_df[~(ds_partners_df.fraction_recurrent_partners==0)].fraction_recurrent_partners
sns.boxplot(y=data, ax=ax, color=sns.color_palette()[1])
sns.stripplot(y=data, ax=ax, s=2, alpha=0.5, color='black', jitter=0.15)
plt.savefig('cascades/feedback_through_brain/plots/recurrent-boxplot-points.pdf', format='pdf', bbox_inches='tight')

# catplot of data
data = ds_partners_df.copy()
data['celltype'] = ['nonrecurrent' if x==0 else 'recurrent' for x in ds_partners_df.fraction_recurrent_partners]
fig, ax = plt.subplots(1,1,figsize=(2,4))
sns.catplot(data = data, x='celltype', y='fraction_recurrent_partners', order=['nonrecurrent', 'recurrent'], kind='boxen')
plt.savefig('cascades/feedback_through_brain/plots/recurrent-boxplot.pdf', format='pdf', bbox_inches='tight')

# stripplot of data
fig, ax = plt.subplots(1,1,figsize=(4,4))
sns.stripplot(y=ds_partners_df.fraction_recurrent_partners, ax=ax, s=3, alpha=0.5, color=sns.color_palette()[1])
plt.savefig('cascades/feedback_through_brain/plots/recurrent-partner-fractions.pdf', format='pdf', bbox_inches='tight')

# distribution plot
fig, ax = plt.subplots(1,1,figsize=(4,4))
sns.histplot(x=ds_partners_df.fraction_recurrent_partners, binwidth=0.05, ax=ax, color='tab:gray', stat='probability')
plt.savefig('cascades/feedback_through_brain/plots/recurrent-partner-fractions_hist.pdf', format='pdf', bbox_inches='tight')

# %%
# recurrence by cell type

_, celltypes = ct.Celltype_Analyzer.default_celltypes()
all_celltypes = [x.skids for x in celltypes]
all_celltypes = [x for sublist in all_celltypes for x in sublist]
other_ct = ct.Celltype('Other', np.setdiff1d(pymaid.get_skids_by_annotation('mw brain neurons'), all_celltypes), 'tab:gray')
celltypes = celltypes + [other_ct]

celltype_annotation = []
for skid in ds_partners_df.index:
    for celltype in celltypes:
        if(skid in celltype.skids):
            celltype_annotation.append(celltype.name)

ds_partners_df['celltype'] = celltype_annotation
#ds_partners_ct_df = ds_partners_df.copy()
#ds_partners_ct_df = ds_partners_ct_df[[False if x==[] else True for x in ds_partners_ct_df.ds_partners]] # remove neurons with no partners


# plot results as barplot with points, barplot, or violinplot
fig, ax = plt.subplots(1,1,figsize=(4,4))
sns.barplot(x=ds_partners_df.celltype, y=ds_partners_df.fraction_recurrent_partners, order=['PNs', 'PNs-somato', 'LNs', 'LHNs', 'FFNs', 'MBINs', 'KCs' ,'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'dSEZs', 'dVNCs', 'Other'])
sns.stripplot(x=ds_partners_df.celltype, y=ds_partners_df.fraction_recurrent_partners, s=1, alpha=0.5, color='black', order=['PNs', 'PNs-somato', 'LNs', 'LHNs', 'FFNs', 'MBINs', 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'dSEZs', 'dVNCs', 'Other'])
plt.xticks(rotation=45, ha='right')
ax.set(ylim=(-0.05, 1))
plt.savefig('cascades/feedback_through_brain/plots/recurrent-partner-fractions_by-celltype_barplot-with-points.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1,figsize=(2,1))
sns.barplot(x=ds_partners_df.celltype, y=ds_partners_df.fraction_recurrent_partners, order=['PNs', 'PNs-somato', 'LNs', 'LHNs', 'FFNs', 'MBINs', 'KCs' , 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'dSEZs', 'dVNCs', 'Other'])
plt.xticks(rotation=45, ha='right')
ax.set(ylim=(0, 1))
plt.savefig('cascades/feedback_through_brain/plots/recurrent-partner-fractions_by-celltype_barplot.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1,figsize=(8,4))
sns.violinplot(x=ds_partners_df.celltype, y=ds_partners_df.fraction_recurrent_partners, scale='width', order=['PNs', 'PNs-somato', 'LNs', 'LHNs', 'FFNs', 'MBINs', 'KCs' , 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'dSEZs', 'dVNCs', 'Other'])
plt.xticks(rotation=45, ha='right')
ax.set(ylim=(0, 1))
plt.savefig('cascades/feedback_through_brain/plots/recurrent-partner-fractions_by-celltype_violinplot.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1,figsize=(8,4))
sns.boxplot(x=ds_partners_df.celltype, y=ds_partners_df.fraction_recurrent_partners, whis=[0, 100], order=['PNs', 'PNs-somato', 'LNs', 'LHNs', 'FFNs', 'MBINs', 'KCs' , 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'dSEZs', 'dVNCs', 'Other'])
plt.xticks(rotation=45, ha='right')
ax.set(ylim=(0, 1))
plt.savefig('cascades/feedback_through_brain/plots/recurrent-partner-fractions_by-celltype_boxplot.pdf', format='pdf', bbox_inches='tight')

# %%
# recurrence by cluster

clusters = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain clusters level 7', split=True)
clusters_ct = list(map(lambda x: ct.Celltype(*x), zip(clusters[1], clusters[0])))

cluster_annotation = []
for skid in ds_partners_df.index:
    i=0
    for celltype in clusters_ct:
        if(skid in celltype.skids):
            cluster_annotation.append(celltype.name)
        if(skid not in celltype.skids):
            i+=1
        if(i==90):
            cluster_annotation.append('None')


ds_partners_df['cluster'] = cluster_annotation


# plot results as barplot with points, barplot, or violinplot
fig, ax = plt.subplots(1,1,figsize=(8,4))
sns.barplot(x=ds_partners_df.cluster, y=ds_partners_df.fraction_recurrent_partners, order=[x.name for x in clusters_ct])
sns.stripplot(x=ds_partners_df.cluster, y=ds_partners_df.fraction_recurrent_partners, s=1, alpha=0.5, color='black', order=[x.name for x in clusters_ct])
plt.xticks(rotation=45, ha='right')
ax.set(ylim=(-0.05, 1))
plt.savefig('cascades/feedback_through_brain/plots/recurrent-partner-fractions_by-cluster_barplot-with-points.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1,figsize=(8,4))
sns.barplot(x=ds_partners_df.cluster, y=ds_partners_df.fraction_recurrent_partners, order=[x.name for x in clusters_ct])
plt.xticks(rotation=45, ha='right')
ax.set(ylim=(0, 1))
plt.savefig('cascades/feedback_through_brain/plots/recurrent-partner-fractions_by-cluster_barplot.pdf', format='pdf', bbox_inches='tight')

# %%
