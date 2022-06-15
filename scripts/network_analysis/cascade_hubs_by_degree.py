# %%

from pymaid_creds import url, name, password, token
from data_settings import pairs_path, data_date
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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
cascades_df_with_inputs = pickle.load(open(f'data/cascades/all-brain-pairs-nonpaired_inputs-interneurons-outputs_{n_init}-n_init_{data_date}.p', 'rb'))

# exclude input neurons, include only brain and accessory neurons
brain = pymaid.get_skids_by_annotation(['mw brain neurons', 'mw brain accessory neurons'])
brain = list(np.intersect1d(brain, cascades_df.index))
cascades_df = cascades_df.loc[brain, :]

pairs = Promat.get_pairs(pairs_path=pairs_path)

# %%
# general analysis of cascade signal

# how many 5-hop and 8-hop partners do individual neurons have? what fraction of all brain neurons?
inputs = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain inputs')
brain = pymaid.get_skids_by_annotation('mw brain neurons')

fract_brain_5hop = [len(x)/len(brain) for x in cascades_df.ds_partners_5hop]
fract_brain_8hop = [len(x)/len(brain) for x in cascades_df.ds_partners_8hop]

fract_brain_5hop_input = [len(x)/len(brain) for x in cascades_df_with_inputs.loc[np.intersect1d(inputs, cascades_df_with_inputs.index)].ds_partners_5hop]
fract_brain_8hop_input = [len(x)/len(brain) for x in cascades_df_with_inputs.loc[np.intersect1d(inputs, cascades_df_with_inputs.index)].ds_partners_8hop]

print(f'From individual 5-hop cascades from sensory pairs, {np.mean(fract_brain_5hop_input)*100:.1f}% +/- {np.std(fract_brain_5hop_input)*100:.1f}% of brain neurons are encountered')
print(f'From individual 8-hop cascades from sensory pairs, {np.mean(fract_brain_8hop_input)*100:.1f}% +/- {np.std(fract_brain_8hop_input)*100:.1f}% of brain neurons are encountered')

print(f'From individual 5-hop cascades from individual brain pairs, {np.mean(fract_brain_5hop)*100:.1f}% +/- {np.std(fract_brain_5hop)*100:.1f}% of brain neurons are encountered')
print(f'From individual 8-hop cascades from individual brain pairs, {np.mean(fract_brain_8hop)*100:.1f}% +/- {np.std(fract_brain_8hop)*100:.1f}% of brain neurons are encountered')

# plot data as boxplots
fig,axs = plt.subplots(1, 2, figsize=(2,1), sharey=True)
ax = axs[0]
df = pd.DataFrame([['5hop' for i in range(0, len(fract_brain_5hop_input))] + ['8hop' for i in range(0, len(fract_brain_8hop_input))], fract_brain_5hop_input+fract_brain_8hop_input], index=['hops','fraction_ds'])
df = df.T
sns.barplot(x=df.hops, y=df.fraction_ds, ax=ax, ci='sd', capsize=0.1)
#sns.boxenplot(x=df.hops, y=df.fraction_ds, ax=ax, whis=[0,100], showmeans=True)
ax.set(ylim=(0,1), title='Sens pairs -> all brain')
ax = axs[1]
df = pd.DataFrame([['5hop' for i in range(0, len(fract_brain_5hop))] + ['8hop' for i in range(0, len(fract_brain_8hop))], fract_brain_5hop+fract_brain_8hop], index=['hops','fraction_ds'])
df = df.T
#sns.boxenplot(x=df.hops, y=df.fraction_ds, ax=ax, whis=[0,100], showmeans=True)
sns.barplot(x=df.hops, y=df.fraction_ds, ax=ax, ci='sd', capsize=0.1)
ax.set(ylim=(0,1), title='Brain pairs -> all brain')
plt.savefig('plots/cascades_individual-pairs_throughout-brain.pdf', format='pdf', bbox_inches='tight')

# plot data as histograms
fig,axs = plt.subplots(1,2,figsize=(3,1))
ax=axs[0]
sns.histplot(fract_brain_5hop, ax=ax, bins=40)
ax.set(xlim=(0,1))

ax=axs[1]
sns.histplot(fract_brain_8hop, ax=ax, bins=40)
ax.set(xlim=(0,1))
plt.savefig('plots/cascades_individual-pairs_throughout-brain_hist.pdf', format='pdf', bbox_inches='tight')

# how many neurons receive signal from single pairs of neurons?
skids_brain_5hop = [x for x in cascades_df.ds_partners_5hop]
skids_brain_8hop = [x for x in cascades_df.ds_partners_8hop]
skids_brain_5hop = list(np.unique([x for sublist in skids_brain_5hop for x in sublist]))
skids_brain_8hop = list(np.unique([x for sublist in skids_brain_8hop for x in sublist]))

skids_brain_5hop_input = [x for x in cascades_df_with_inputs.loc[np.intersect1d(inputs, cascades_df_with_inputs.index)].ds_partners_5hop]
skids_brain_8hop_input = [x for x in cascades_df_with_inputs.loc[np.intersect1d(inputs, cascades_df_with_inputs.index)].ds_partners_8hop]
skids_brain_5hop_input = list(np.unique([x for sublist in skids_brain_5hop_input for x in sublist]))
skids_brain_8hop_input = list(np.unique([x for sublist in skids_brain_8hop_input for x in sublist]))

print(f'From all 5-hop cascades from sensory pairs, {len(skids_brain_5hop_input)/len(brain)*100:.1f}% of brain neurons are encountered')
print(f'From all 8-hop cascades from sensory pairs, {len(skids_brain_8hop_input)/len(brain)*100:.1f}% of brain neurons are encountered')

print(f'From all 5-hop cascades from individual brain pairs, {len(skids_brain_5hop)/len(brain)*100:.1f}% of brain neurons are encountered')
print(f'From all 8-hop cascades from individual brain pairs, {len(skids_brain_8hop)/len(brain)*100:.1f}% of brain neurons are encountered')

# %%
# identify upstream partners via cascade

def generate_us_df(partners_df, hops):
    skids = partners_df.index

    us_partners_list = []
    for source in skids:
        us_partners = []
        for skid in skids:
            ds = list(partners_df.loc[skid, f'ds_partners_{hops}hop'])
            if(source in ds):
                us_partners.append(skid)

        us_partners_list.append(us_partners)

    return(us_partners_list)

partners_df = cascades_df.loc[:, ['ds_partners_8hop', 'ds_partners_5hop']]

# convert to pair_ids; did this as a stop-gap because expanding the upstream partners took too long on my current compute
partners_df['ds_partners_8hop'] = [list(np.intersect1d(x, partners_df.index)) for x in partners_df.ds_partners_8hop]
partners_df['ds_partners_5hop'] = [list(np.intersect1d(x, partners_df.index)) for x in partners_df.ds_partners_5hop]

# identify upstream neurons, takes a few minutes
partners_df['us_partners_8hop'] = generate_us_df(partners_df, hops=8)
partners_df['us_partners_5hop'] = generate_us_df(partners_df, hops=5)

'''
# expand out pair_ids in ds/us partners, takes unknown amount of time >20 minutes; suggests Promat.get_paired_skids() is inefficient
partners_df['ds_partners_8hop'] = [Promat.get_paired_skids(x, pairs, unlist=True) for x in partners_df.ds_partners_8hop]
partners_df['ds_partners_5hop'] = [Promat.get_paired_skids(x, pairs, unlist=True) for x in partners_df.ds_partners_5hop]
partners_df['us_partners_8hop'] = [Promat.get_paired_skids(x, pairs, unlist=True) for x in partners_df.us_partners_8hop]
partners_df['us_partners_5hop'] = [Promat.get_paired_skids(x, pairs, unlist=True) for x in partners_df.us_partners_5hop]
'''
# %%
# identify cascade hub neurons (in, in-out, and out) using 'us_partners_[8|5]hop' and 'ds_partners_[8|5]hop'

partners_df['out_degree_5hop'] = [len(x) for x in partners_df.ds_partners_5hop]
partners_df['out_degree_8hop'] = [len(x) for x in partners_df.ds_partners_8hop]
partners_df['in_degree_5hop'] = [len(x) for x in partners_df.us_partners_5hop]
partners_df['in_degree_8hop'] = [len(x) for x in partners_df.us_partners_8hop]

partners_df['type_5hop'] = ['' for i in range(0, len(partners_df.index))]
partners_df['type_8hop'] = ['' for i in range(0, len(partners_df.index))]

threshold_5hop = np.round((partners_df.in_degree_5hop.mean() + partners_df.out_degree_5hop.mean())/2 + 1.5*(partners_df.in_degree_5hop.std() + partners_df.out_degree_5hop.std())/2)
threshold_8hop = np.round((partners_df.in_degree_8hop.mean() + partners_df.out_degree_8hop.mean())/2 + 1.5*(partners_df.in_degree_8hop.std() + partners_df.out_degree_8hop.std())/2)

for ind in partners_df.index:
    test = partners_df.loc[ind, :]

    # 5-hop hubs
    if(test.in_degree_5hop>=threshold_5hop) & (test.out_degree_5hop<threshold_5hop):
        partners_df.loc[ind, 'type_5hop'] = 'in_hub'
    if(test.in_degree_5hop<threshold_5hop) & (test.out_degree_5hop>=threshold_5hop):
        partners_df.loc[ind, 'type_5hop'] = 'out_hub'
    if(test.in_degree_5hop>=threshold_5hop) & (test.out_degree_5hop>=threshold_5hop):
        partners_df.loc[ind, 'type_5hop'] = 'in_out_hub'
    if(test.in_degree_5hop<threshold_5hop) & (test.out_degree_5hop<threshold_5hop):
        partners_df.loc[ind, 'type_5hop'] = 'non_hub'
    
    # 8-hop hubs
    if(test.in_degree_8hop>=threshold_8hop) & (test.out_degree_8hop<threshold_8hop):
        partners_df.loc[ind, 'type_8hop'] = 'in_hub'
    if(test.in_degree_8hop<threshold_8hop) & (test.out_degree_8hop>=threshold_8hop):
        partners_df.loc[ind, 'type_8hop'] = 'out_hub'
    if(test.in_degree_8hop>=threshold_8hop) & (test.out_degree_8hop>=threshold_8hop):
        partners_df.loc[ind, 'type_8hop'] = 'in_out_hub'
    if(test.in_degree_8hop<threshold_8hop) & (test.out_degree_8hop<threshold_8hop):
        partners_df.loc[ind, 'type_8hop'] = 'non_hub'

# plot degree distributions with hub coloring
thresholds = [threshold_5hop, threshold_8hop]
for i, hop in enumerate([5,8]):
    hubs_plot = partners_df.groupby([f'in_degree_{hop}hop', f'out_degree_{hop}hop']).count().iloc[:, 0].reset_index()
    hubs_plot.columns = ['in_degree', 'out_degree', 'count']
    hub_type = []
    for index in range(0, len(hubs_plot)):
        if((hubs_plot.iloc[index, :].in_degree>=thresholds[i]) & (hubs_plot.iloc[index, :].out_degree<thresholds[i])):
            hub_type.append('in_hub')
        if((hubs_plot.iloc[index, :].out_degree>=thresholds[i]) & (hubs_plot.iloc[index, :].in_degree<thresholds[i])):
            hub_type.append('out_hub')
        if((hubs_plot.iloc[index, :].in_degree>=thresholds[i]) & (hubs_plot.iloc[index, :].out_degree>=thresholds[i])):
            hub_type.append('in_out_hub')
        if((hubs_plot.iloc[index, :].in_degree<thresholds[i]) & (hubs_plot.iloc[index, :].out_degree<thresholds[i])):
            hub_type.append('non-hub')

    hubs_plot['type']=hub_type

    fig, ax = plt.subplots(1,1, figsize=(1.5,1.5))
    sns.scatterplot(data=hubs_plot.iloc[1:, :], x='in_degree', y='out_degree', hue='type', size='count', ax=ax, 
                    sizes=(1, 15), edgecolor='none', alpha=0.8)
    ax.set(ylim=(0, 1250), xlim=(0, 1200))
    ax.axvline(x=thresholds[i], color='grey', linewidth=0.25, alpha=0.5)
    ax.axhline(y=thresholds[i], color='grey', linewidth=0.25, alpha=0.5)
    ax.legend().set_visible(False)
    plt.savefig(f'plots/cascade-{hop}hop_hubs_scatterplot.pdf', format='pdf', bbox_inches='tight')

    fig, ax = plt.subplots(1,1, figsize=(2,2))
    sns.scatterplot(data=hubs_plot.iloc[1:, :], x='in_degree', y='out_degree', hue='type', size='count', ax=ax, 
                    sizes=(1, 15), edgecolor='none', alpha=0.8)
    ax.set(ylim=(0, 1250), xlim=(0, 1200))
    ax.axvline(x=thresholds[i], color='grey', linewidth=0.25, alpha=0.5)
    ax.axhline(y=thresholds[i], color='grey', linewidth=0.25, alpha=0.5)
    plt.savefig(f'plots/cascade-{hop}hop_hubs_scatterplot_legend.pdf', format='pdf', bbox_inches='tight')
# %%
# compare to connectivity-based axo-dendritic hubs
in_hub_ad = pymaid.get_skids_by_annotation('mw ad hubs_in')
out_hub_ad = pymaid.get_skids_by_annotation('mw ad hubs_out')
in_out_hub_ad = pymaid.get_skids_by_annotation('mw ad hubs_in_out')

in_hub_5hop = list(partners_df[partners_df.type_5hop=='in_hub'].index)
out_hub_5hop = list(partners_df[partners_df.type_5hop=='out_hub'].index)
in_out_hub_5hop = list(partners_df[partners_df.type_5hop=='in_out_hub'].index)
non_hub_5hop = list(partners_df[partners_df.type_5hop=='non_hub'].index)

in_hub_5hop_all = Promat.get_paired_skids(in_hub_5hop, pairs, unlist=True)
out_hub_5hop_all = Promat.get_paired_skids(out_hub_5hop, pairs, unlist=True)
in_out_hub_5hop_all = Promat.get_paired_skids(in_out_hub_5hop, pairs, unlist=True)
non_hub_5hop_all = Promat.get_paired_skids(non_hub_5hop, pairs, unlist=True)

pymaid.add_annotations(in_hub_5hop_all, 'mw cascade hubs_in')
pymaid.add_annotations(out_hub_5hop_all, 'mw cascade hubs_out')
pymaid.add_annotations(in_out_hub_5hop_all, 'mw cascade hubs_in_out')

print(f'{len(np.intersect1d(in_hub_ad, in_hub_5hop_all))/len(in_hub_ad)*100:.1f}% of a-d in hubs are also cascade in hubs')
print(f'{len(np.intersect1d(out_hub_ad, out_hub_5hop_all))/len(out_hub_ad)*100:.1f}% of a-d in hubs are also cascade out hubs')
print(f'{len(np.intersect1d(in_out_hub_ad, in_out_hub_5hop_all))/len(in_out_hub_ad)*100:.1f}% of a-d in hubs are also cascade in-out hubs')
print(f'{len(np.intersect1d(in_out_hub_ad + out_hub_ad + in_hub_ad, in_hub_5hop_all + out_hub_5hop_all + in_out_hub_5hop_all))/len(in_out_hub_ad + out_hub_ad + in_hub_ad)*100:.1f}% of a-d hubs are also cascade hubs')

# %%
# celltypes in cascade hubs

_, celltypes = Celltype_Analyzer.default_celltypes()

cascade_hubs_cta = [Celltype('casc-out-hub', out_hub_5hop_all),
                    Celltype('casc-in-out-hubs', in_out_hub_5hop_all),
                    Celltype('casc-in-hubs', in_hub_5hop_all)]
cascade_hubs_cta = Celltype_Analyzer(cascade_hubs_cta)
cascade_hubs_cta.set_known_types(celltypes)
cascade_hubs_cta.plot_memberships('plots/cascade-hubs_celltypes', (1,2))

# %%
# quantify % of the brain that receives input/output from hubs
brain_pairids = Promat.load_pairs_from_annotation('', pairs, return_type='all_pair_ids', skids=brain, use_skids=True)

mean = np.mean(partners_df.loc[in_hub_5hop, 'in_degree_5hop'])/len(brain_pairids)*100
std = np.std(partners_df.loc[in_hub_5hop, 'in_degree_5hop'])/len(brain_pairids)*100
print(f'Cascade in-hubs receive from {mean:.1f}% +/- {std:.1f}% of the brain')

mean = np.mean(partners_df.loc[in_out_hub_5hop, 'in_degree_5hop'])/len(brain_pairids)*100
std = np.std(partners_df.loc[in_out_hub_5hop, 'in_degree_5hop'])/len(brain_pairids)*100
print(f'Cascade in-hubs receive from {mean:.1f}% +/- {std:.1f}% of the brain')

mean = np.mean(partners_df.loc[in_out_hub_5hop, 'out_degree_5hop'])/len(brain_pairids)*100
std = np.std(partners_df.loc[in_out_hub_5hop, 'out_degree_5hop'])/len(brain_pairids)*100
print(f'Cascade in-hubs output to {mean:.1f}% +/- {std:.1f}% of the brain')

mean = np.mean(partners_df.loc[out_hub_5hop, 'out_degree_5hop'])/len(brain_pairids)*100
std = np.std(partners_df.loc[out_hub_5hop, 'out_degree_5hop'])/len(brain_pairids)*100
print(f'Cascade out-hubs output to {mean:.1f}% +/- {std:.1f}% of the brain')

np.mean(partners_df.loc[in_hub_5hop, 'in_degree_5hop'])/len(brain_pairids)
np.mean(partners_df.loc[in_out_hub_5hop, 'in_degree_5hop'])/len(brain_pairids)
np.mean(partners_df.loc[out_hub_5hop, 'out_degree_5hop'])/len(brain_pairids)
