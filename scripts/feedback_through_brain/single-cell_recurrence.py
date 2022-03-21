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
from tqdm import tqdm
from joblib import Parallel, delayed

n_init = 1000
cascades_df = pickle.load(open(f'data/cascades/all-brain-pairs-nonpaired_inputs-interneurons-outputs_{n_init}-n_init_{data_date}.p', 'rb'))

# exclude input neurons, include only brain and accessory neurons
brain = pymaid.get_skids_by_annotation(['mw brain neurons', 'mw brain accessory neurons'])
brain = list(np.intersect1d(brain, cascades_df.index))
cascades_df = cascades_df.loc[brain, :]

# %%
# how many partners are in recurrent loops?

# collect all recurrent skids

def recurrent_partners(skid, ds_partners_df, hops, pairs):
    recurrent_partners = []
    ds_partners = ds_partners_df.loc[skid, f'ds_partners_{hops}hop']
    #ds_partners = Promat.extract_pairs_from_list(ds_partners, pairs, return_pair_ids=True)[3] # convert to pair_ids, very slow
    ds_partners = list(np.intersect1d(ds_partners, ds_partners_df.index))  # effectively converts to pair_ids

    for partner in ds_partners:
        if(partner in ds_partners_df.index):
            ds_ds_partners = ds_partners_df.loc[partner, f'ds_partners_{hops}hop']
            if(skid in ds_ds_partners):
                recurrent_partners.append(partner)

        if(partner not in ds_partners_df.index):
            print(f'{partner} not in skid list!')

        # expand to both left/right neurons
        #recurrent_partners = [Promat.get_paired_skids(skid, pairs) for skid in recurrent_partners]
        #recurrent_partners = [x for sublist in recurrent_partners for x in sublist]
        #recurrent_partners = list(np.unique(recurrent_partners))
    
    return(recurrent_partners)

ds_partners_df = cascades_df.loc[:, ['ds_partners_8hop', 'ds_partners_5hop']]

# convert to pair_ids; did this as a stop-gap because expanding the recurrent partners took too long on my current compute
ds_partners_df['ds_partners_8hop'] = [list(np.intersect1d(x, ds_partners_df.index)) for x in ds_partners_df.ds_partners_8hop]
ds_partners_df['ds_partners_5hop'] = [list(np.intersect1d(x, ds_partners_df.index)) for x in ds_partners_df.ds_partners_5hop]

pairs = Promat.get_pairs(pairs_path=pairs_path)

hops = 8
ds_partners_df[f'recurrent_partners_{hops}hop'] = Parallel(n_jobs=-1)(delayed(recurrent_partners)(ds_partners_df.index[i], ds_partners_df, hops, pairs) for i in tqdm(range(len(ds_partners_df.index))))
hops = 5
ds_partners_df[f'recurrent_partners_{hops}hop'] = Parallel(n_jobs=-1)(delayed(recurrent_partners)(ds_partners_df.index[i], ds_partners_df, hops, pairs) for i in tqdm(range(len(ds_partners_df.index))))

import pickle
pickle.dump(ds_partners_df, open(f'data/cascades/all-recurrent-partners_brain-interneurons-outputs_{n_init}-n_init_{data_date}.p', 'wb'))

# %%
# analysis of recurrent partners at 5- and 8-hops

def recurrent_plots(ds_partners_df, hops, celltypes):
    # fraction of recurrent vs. nonrecurrent parents at 8-hop
    frac_recurrent = [len(ds_partners_df.loc[i, f'recurrent_partners_{hops}hop'])/len(ds_partners_df.loc[i, f'ds_partners_{hops}hop']) if len(ds_partners_df.loc[i, f'ds_partners_{hops}hop'])>0 else 0 for i in ds_partners_df.index]
    ds_partners_df[f'fraction_recurrent_{hops}hop'] = frac_recurrent
    ds_partners_df[f'fraction_nonrecurrent_{hops}hop'] = 1-ds_partners_df.loc[:, f'fraction_recurrent_{hops}hop']

    # plot total number of recurrent neurons
    fig, ax = plt.subplots(1,1,figsize=(.5,1))

    # duplicate values for pairs, values for nonpaired neurons remain the same

    # below probably doesn't apply
    #_, unpaired, nonpaired = pm.Promat.extract_pairs_from_list(ds_partners_df.index, pm.Promat.get_pairs())
    #data = list(ds_partners_df.loc[unpaired.unpaired, f'fraction_recurrent_{hops}hop']) + list(ds_partners_df.loc[unpaired.unpaired, :].fraction_recurrent_partners) + list(ds_partners_df.loc[nonpaired.nonpaired, :].fraction_recurrent_partners)
    data = ds_partners_df.loc[:, f'fraction_recurrent_{hops}hop']
    data = [sum(np.array(data)==0)/len(data), sum(np.array(data)!=0)/len(data)]
    sns.barplot(x=[f'Non-recurrent Neurons ({hops}-Hops)', f'Recurrent Neurons ({hops}-Hops)'] , y=data, ax=ax)
    ax.set(ylim=(0,1))
    plt.savefig(f'plots/recurrent-vs-nonrecurrent_fractions_{hops}hops.pdf', format='pdf', bbox_inches='tight')

    # boxplot of data with points
    fig, ax = plt.subplots(1,1,figsize=(2,4))
    data = ds_partners_df[~(ds_partners_df.loc[:, f'fraction_recurrent_{hops}hop']==0)].loc[:, f'fraction_recurrent_{hops}hop']
    sns.boxplot(y=data, ax=ax, color=sns.color_palette()[1])
    sns.stripplot(y=data, ax=ax, s=2, alpha=0.5, color='black', jitter=0.15)
    plt.savefig(f'plots/recurrent-boxplot-points_{hops}hops.pdf', format='pdf', bbox_inches='tight')

    # catplot of data
    data = ds_partners_df.copy()
    data['celltype'] = ['nonrecurrent' if x==0 else 'recurrent' for x in ds_partners_df.loc[:, f'fraction_recurrent_{hops}hop']]
    fig, ax = plt.subplots(1,1,figsize=(2,4))
    sns.catplot(data = data, x='celltype', y=f'fraction_recurrent_{hops}hop', order=['nonrecurrent', 'recurrent'], kind='boxen')
    plt.savefig(f'plots/recurrent-boxplot_{hops}hops.pdf', format='pdf', bbox_inches='tight')

    # stripplot of data
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    sns.stripplot(y=ds_partners_df.loc[:, f'fraction_recurrent_{hops}hop'], ax=ax, s=3, alpha=0.5, color=sns.color_palette()[1])
    plt.savefig(f'plots/recurrent-partner-fractions_{hops}hops.pdf', format='pdf', bbox_inches='tight')

    # distribution plot
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    sns.histplot(x=ds_partners_df.loc[:, f'fraction_recurrent_{hops}hop'], binwidth=0.05, ax=ax, color='tab:gray', stat='probability')
    plt.savefig(f'plots/recurrent-partner-fractions_hist_{hops}hops.pdf', format='pdf', bbox_inches='tight')


    ##########
    # celltype plots

    celltype_annotation = []
    for skid in ds_partners_df.index:
        for celltype in celltypes:
            if(skid in celltype.skids):
                celltype_annotation.append(celltype.name)

    ds_partners_df['celltype'] = celltype_annotation

    # plot results as barplot with points, barplot, or violinplot
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    sns.barplot(x=ds_partners_df.celltype, y=ds_partners_df.loc[:, f'fraction_recurrent_{hops}hop'], order=['PNs', 'PNs-somato', 'LNs', 'LHNs', 'FFNs', 'MBINs', 'KCs' ,'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'dSEZs', 'dVNCs', 'Other'])
    sns.stripplot(x=ds_partners_df.celltype, y=ds_partners_df.loc[:, f'fraction_recurrent_{hops}hop'], s=1, alpha=0.5, color='black', order=['PNs', 'PNs-somato', 'LNs', 'LHNs', 'FFNs', 'MBINs', 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'dSEZs', 'dVNCs', 'Other'])
    plt.xticks(rotation=45, ha='right')
    ax.set(ylim=(-0.05, 1))
    plt.savefig(f'plots/recurrent-partner-fractions_{hops}hops_by-celltype_barplot-with-points.pdf', format='pdf', bbox_inches='tight')

    fig, ax = plt.subplots(1,1,figsize=(2,1))
    sns.barplot(x=ds_partners_df.celltype, y=ds_partners_df.loc[:, f'fraction_recurrent_{hops}hop'], order=['PNs', 'PNs-somato', 'LNs', 'LHNs', 'FFNs', 'MBINs', 'KCs' , 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'dSEZs', 'dVNCs', 'Other'])
    plt.xticks(rotation=45, ha='right')
    ax.set(ylim=(0, 1))
    plt.savefig(f'plots/recurrent-partner-fractions_{hops}hops_by-celltype_barplot.pdf', format='pdf', bbox_inches='tight')

    fig, ax = plt.subplots(1,1,figsize=(8,4))
    sns.violinplot(x=ds_partners_df.celltype, y=ds_partners_df.loc[:, f'fraction_recurrent_{hops}hop'], scale='width', order=['PNs', 'PNs-somato', 'LNs', 'LHNs', 'FFNs', 'MBINs', 'KCs' , 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'dSEZs', 'dVNCs', 'Other'])
    plt.xticks(rotation=45, ha='right')
    ax.set(ylim=(0, 1))
    plt.savefig(f'plots/recurrent-partner-fractions_{hops}hops_by-celltype_violinplot.pdf', format='pdf', bbox_inches='tight')

    fig, ax = plt.subplots(1,1,figsize=(8,4))
    sns.boxplot(x=ds_partners_df.celltype, y=ds_partners_df.loc[:, f'fraction_recurrent_{hops}hop'], whis=[0, 100], order=['PNs', 'PNs-somato', 'LNs', 'LHNs', 'FFNs', 'MBINs', 'KCs' , 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'dSEZs', 'dVNCs', 'Other'])
    plt.xticks(rotation=45, ha='right')
    ax.set(ylim=(0, 1))
    plt.savefig(f'plots/recurrent-partner-fractions_{hops}hops_by-celltype_boxplot.pdf', format='pdf', bbox_inches='tight')

    return(ds_partners_df)

_, celltypes = Celltype_Analyzer.default_celltypes()
all_celltypes = [x.skids for x in celltypes]
all_celltypes = [x for sublist in all_celltypes for x in sublist]
other_ct = Celltype('Other', np.setdiff1d(ds_partners_df.index, all_celltypes), 'tab:gray')
celltypes = celltypes + [other_ct]

ds_partners_df2 = recurrent_plots(ds_partners_df=ds_partners_df, hops=8, celltypes=celltypes)
ds_partners_df3 = recurrent_plots(ds_partners_df=ds_partners_df, hops=5, celltypes=celltypes)

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
# checking neurons upstream of KCs, do they receive cascade signal?
# the concern is whether the result that KC have low recurrence real or a result of cascade thresholding

ds_KC = np.unique([x for sublist in ds_partners_df.loc[np.intersect1d(pymaid.get_skids_by_annotation('mw KC'), ds_partners_df.index), :].ds_partners for x in sublist])
us_KC = pymaid.get_skids_by_annotation('mw upstream KCs')

len(ds_KC)
print(f'there are {len(np.intersect1d(ds_KC, us_KC))} neurons in KC cascades that are directly upstream')
print(f'note that there are {len(ds_KC)} total neurons downstream of KCs by 8-hop cascade')

# %%
# recurrence by cluster
# NOT USED IN PAPER

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
# flip cascades to make an us_partners_df

skids = ds_partners_df.index

us_partners_df = []
for source in skids:
    us_partners = []
    for skid in skids:
        ds = list(ds_partners_df.loc[skid, 'ds_partners'])
        if(source in ds):
            us_partners.append(skid)

    us_partners_df.append([source, us_partners])

us_partners_df = pd.DataFrame(us_partners_df, columns = ['skid', 'us_partners'])
us_partners_df.set_index('skid', inplace=True)

# %%
# look at recurrence from the perspective of downstream neuron
# turns out it's exactly the same by definition (seemed like that would be likely, but I checked just in case)

# collect all recurrent skids
recurrent_partners_col = []
for skid in us_partners_df.index:
    recurrent_partners = []
    us_partners = us_partners_df.loc[skid, 'us_partners']
    for partner in us_partners:
        if(partner in us_partners_df.index):
            us_us_partners = us_partners_df.loc[partner, 'us_partners']
            if(skid in us_us_partners):
                recurrent_partners.append(partner)

        if(partner not in us_partners_df.index):
            print(f'{partner} not in skid list!')
            
    recurrent_partners_col.append(recurrent_partners)

# fraction of recurrent vs. nonrecurrent parents
us_partners_df['recurrent_partners'] = recurrent_partners_col
frac_recurrent = [len(us_partners_df.loc[i, 'recurrent_partners'])/len(us_partners_df.loc[i, 'us_partners']) if len(us_partners_df.loc[i, 'us_partners'])>0 else 0 for i in us_partners_df.index]
us_partners_df['fraction_recurrent_partners'] = frac_recurrent
us_partners_df['fraction_nonrecurrent_partners'] = 1-us_partners_df.fraction_recurrent_partners

# plot total number of recurrent neurons
fig, ax = plt.subplots(1,1,figsize=(.5,1))

# duplicate values for pairs, values for nonpaired neurons remain the same
_, unpaired, nonpaired = pm.Promat.extract_pairs_from_list(us_partners_df.index, pm.Promat.get_pairs())
data = list(us_partners_df.loc[unpaired.unpaired, :].fraction_recurrent_partners) + list(us_partners_df.loc[unpaired.unpaired, :].fraction_recurrent_partners) + list(us_partners_df.loc[nonpaired.nonpaired, :].fraction_recurrent_partners)
data = [sum(np.array(data)==0)/len(data), sum(np.array(data)!=0)/len(data)]
sns.barplot(x=['Non-recurrent Neurons', 'Recurrent Neurons'] , y=data, ax=ax)
ax.set(ylim=(0,1))
plt.savefig('cascades/feedback_through_brain/plots/recurrent-vs-nonrecurrent_fractions_upstream-perspective.pdf', format='pdf', bbox_inches='tight')

# %%
# multilength recurrence onto upstream neurons?

hit_thres = n_init/10

multilayered_df = []
for i, hit_hist in enumerate(pair_hist_list):
    skid_hit_hist = hit_hist.skid_hit_hist
    number_layers = (skid_hit_hist>hit_thres).sum(axis=1)

    # identify length of each path
    df_boolen = (skid_hit_hist>hit_thres)
    ds = list(df_boolen.index[(df_boolen).sum(axis=1)>0])

    lengths = []
    for skid in ds:
        lengths.append(list(df_boolen.columns[df_boolen.loc[skid]]))

    df = list(zip(skid_hit_hist.index, number_layers, lengths))
    df = pd.DataFrame(df, columns=['ds_partner', 'layers', 'lengths'])
    df.set_index('ds_partner', inplace=True)
    multilayered_df.append(df)

# plotting layer counts and length stats
layer_counts = []
lengths = []
for i, source in enumerate([x.name for x in pair_hist_list]):
    source_layers = []
    us_partners = list(us_partners_df.loc[source, 'us_partners'])

    # expand to include pairs if relevant
    indices = list(np.intersect1d(us_partners, multilayered_df[i].index))
    counts = list(multilayered_df[i].loc[indices].layers)
    layer_counts.append([source, counts, np.mean(counts), np.std(counts)])

    lengths_data = list(multilayered_df[i].loc[indices].lengths)
    #lengths_data = [x for sublist in lengths_data for x in sublist]
    lengths.append([source, lengths_data])#, np.mean(lengths_data), np.std(lengths_data)])

all_counts = [x[1] for x in layer_counts]
all_counts = [x for sublist in all_counts for x in sublist]

binwidth = 1
bins = np.arange(min(all_counts), max(all_counts) + binwidth*1.5) - binwidth*0.5

figsize = (.75, 0.35)

fig, ax = plt.subplots(1,1,figsize=figsize)
sns.histplot(x=all_counts, bins=bins, stat='density')
plt.xticks(rotation=45, ha='right')
ax.set(xlim=(-0.75, 6.75), ylim=(0,0.55))
plt.savefig(f'cascades/feedback_through_brain/plots/brain-single-cell_recurrent-layers.pdf', format='pdf', bbox_inches='tight')

paths_mean = np.mean([x for x in all_counts if x!=0])
paths_std = np.std([x for x in all_counts if x!=0])
print(f'Recurrent pathways were multilength with {paths_mean:.2f} +/- {paths_std:.2f} different lengths (mean+/-std)')

# plot distribution of lengths
lengths = pd.DataFrame(lengths, columns=['source', 'lengths'])

all_lengths = [x for sublist in lengths.lengths for x in sublist]

binwidth = 1
bins = np.arange(min(all_lengths), max(all_lengths) + binwidth*1.5) - binwidth*0.5

figsize = (.75, 0.35)

fig, ax = plt.subplots(1,1,figsize=figsize)
sns.histplot(x=all_lengths, bins=bins, stat='density')
plt.xticks(rotation=45, ha='right')
ax.set(xlim=(-0.75, 7.75), ylim=(0,0.55))
plt.savefig(f'cascades/feedback_through_brain/plots/brain-single-cell_recurrent-layers_path-length-distribution.pdf', format='pdf', bbox_inches='tight')

# %%
# investigate lengths a bit more

all_lengths = [x for sublist in lengths.lengths for x in sublist if len(x)>1]

length_mean = np.mean([np.mean(x) for x in all_lengths])
length_std = np.std([np.mean(x) for x in all_lengths])

length_min = np.mean([np.min(x) for x in all_lengths])
length_min_std = np.std([np.min(x) for x in all_lengths])

length_max = np.mean([np.max(x) for x in all_lengths])
length_max_std = np.std([np.max(x) for x in all_lengths])

print(f'The mean recurrent path length was {length_mean:.1f}+/-{length_std:.1f}')
print(f'with on average the shortest path at {length_min:.1f}+/-{length_min_std:.1f}')
print(f'and longest path at {length_max:.1f}+/-{length_max_std:.1f}')