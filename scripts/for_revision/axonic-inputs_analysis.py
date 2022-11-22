# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
from data_settings import data_date, pairs_path
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

from contools import Celltype, Celltype_Analyzer, Promat

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

pairs = Promat.get_pairs(pairs_path=pairs_path)

# load previously generated edge list with % threshold
threshold = 2
synapse_threshold = True
ad_edges = Promat.pull_edges(
    type_edges='aa',
    threshold=threshold,
    data_date=data_date,
    pairs_combined=False,
    synapse_threshold=synapse_threshold
)

# load axonic inputs and output CSVs

axon_inputs = pd.read_csv('data/adj/inputs_' + data_date + '.csv', index_col=0)
axon_outputs = pd.read_csv('data/adj/outputs_' + data_date + '.csv', index_col=0)
input_output = pd.concat([axon_inputs, axon_outputs], axis=1)

edges_aa = Promat.pull_edges(type_edges='aa', threshold=2, data_date=data_date, pairs_combined=True, synapse_threshold=True)
edges_da = Promat.pull_edges(type_edges='da', threshold=2, data_date=data_date, pairs_combined=True, synapse_threshold=True)
edges_axonic = pd.concat([edges_aa, edges_da])
edges_axonic = edges_axonic.reset_index(drop=True)

adj_aa = Promat.pull_adj(type_adj='aa', data_date=data_date)
adj_da = Promat.pull_adj(type_adj='da', data_date=data_date)

adj_axonic = adj_aa + adj_da

# %%
# addressing comments about axonic inputs and outputs

# first, find which neurons have intact axons (heuristic: contains an output site)
brain_neurons = pymaid.get_skids_by_annotation('mw brain neurons')
remove_neurons = pymaid.get_skids_by_annotation(['mw brain very incomplete', 'mw partially differentiated'])
brain_neurons = list(np.setdiff1d(brain_neurons, remove_neurons)) # remove neurons that are incomplete or partially differentiated

skids_with_axon = input_output[input_output.axon_output>0].index
skids_with_axon = list(np.intersect1d(brain_neurons, skids_with_axon))

# fraction of neurons with >=2 inputs to axon
frac_2syn_axon_input = sum(input_output.loc[skids_with_axon].axon_input>=2)/len(input_output.loc[skids_with_axon])
print(f'{frac_2syn_axon_input*100:.1f}% of neurons have >=2 inputs onto their axons')

# fraction of neurons with symmetrical >=2 inputs
pairs_sym_axon_input = np.unique(edges_axonic[(edges_axonic.upstream_status=='paired') & (edges_axonic.downstream_status=='paired')].downstream_pair_id)
pairs_sym_axon_input = list(np.intersect1d(brain_neurons, pairs_sym_axon_input))
pairs_sym_axon_input = len(pairs_sym_axon_input)*2
print(f'{pairs_sym_axon_input/len(skids_with_axon)*100:.1f}% of neurons have symmetrical axonic inputs')

# fraction of neurons that output to an axon with >=2 inputs
frac_output_axonic = sum((adj_axonic.loc[skids_with_axon, :]>=2).sum(axis=1)>0)/len(skids_with_axon)
print(f'{frac_output_axonic*100:.1f}% of neurons have >=2 synaptic outputs onto other axons')

# fraction of neurons with symmetrical >=2 synaptic outputs
pairs_sym_axon_output = np.unique(edges_axonic[(edges_axonic.upstream_status=='paired') & (edges_axonic.downstream_status=='paired')].upstream_pair_id)
pairs_sym_axon_output = list(np.intersect1d(brain_neurons, pairs_sym_axon_output))
pairs_sym_axon_output = len(pairs_sym_axon_output)*2
print(f'{pairs_sym_axon_output/len(skids_with_axon)*100:.1f}% of neurons have symmetrical outputs to other axons')

# %%
# addressing comments about dendritic outputs

# fraction of neurons with dendritic outputs of >=2 to another neuron
adj_dd = Promat.pull_adj(type_adj='dd', data_date=data_date)
adj_da = Promat.pull_adj(type_adj='da', data_date=data_date)

adj_den_outputs = adj_dd + adj_da

frac_output_dendritic = sum((adj_den_outputs.loc[brain_neurons, :]>=2).sum(axis=1)>0)/len(brain_neurons)
print(f'{frac_output_dendritic*100:.1f}% of neurons have >=2 dendritic outputs')

# fraction of neurons with symmetrical dendritic outputs
edges_dd = Promat.pull_edges(type_edges='dd', threshold=2, data_date=data_date, pairs_combined=True, synapse_threshold=True)
edges_da = Promat.pull_edges(type_edges='da', threshold=2, data_date=data_date, pairs_combined=True, synapse_threshold=True)
edges_dendritic = pd.concat([edges_dd, edges_da])
edges_dendritic = edges_dendritic.reset_index(drop=True)
edges_dendritic = edges_dendritic[[x in brain_neurons for x in edges_dendritic.upstream_pair_id]]

pairs_sym_den_output = np.unique(edges_dendritic[(edges_dendritic.upstream_status=='paired') & (edges_dendritic.downstream_status=='paired')].upstream_pair_id)
pairs_sym_den_output = list(np.intersect1d(brain_neurons, pairs_sym_den_output))
pairs_sym_den_output = len(pairs_sym_den_output)*2
print(f'{pairs_sym_den_output/len(brain_neurons)*100:.1f}% of neurons have symmetrical dendritic outputs to other neurons')

# %%
# ratio of axonic input / output

axon_input_output = input_output[input_output.axon_output>0]
axon_input_output = axon_input_output.axon_input/axon_input_output.axon_output
axon_input_output = axon_input_output[~np.isnan(axon_input_output)]
axon_input_output = axon_input_output.loc[np.intersect1d(brain_neurons, axon_input_output.index)]

axon_input_output = axon_input_output.sort_values(ascending=True)

mean = np.mean(axon_input_output)
median = np.median(axon_input_output)
std = np.std(axon_input_output)
std_low = mean - std
std_high = mean + std
std2_high = mean + std*2
std3_high = mean + std*3
std4_high = mean + std*4

print(f'The median axonic I/O ratio is {median:.3f}')
print(f'The mean axonic I/O ratio is {mean:.3f}')

fig, ax = plt.subplots(1,1, figsize=(4,3))
sns.scatterplot(x=range(len(axon_input_output)), y=axon_input_output, ec='gray', fc='none', alpha=0.75, ax=ax)
plt.axhline(y=mean, color='gray', linewidth = 0.5)
plt.axhline(y=std2_high, color='gray', linewidth = 0.5)
plt.axhline(y=std4_high, color='gray', linewidth = 0.5)
plt.savefig('plots/all-cell_ratio-axonic-IO.pdf', format='pdf', bbox_inches='tight')


# %%
# which neurons have highest and lowest IO ratio

_, celltypes = Celltype_Analyzer.default_celltypes()

std2_high_skids = axon_input_output[(axon_input_output>=std2_high) & (axon_input_output<std4_high)].index
std4_high_skids = axon_input_output[axon_input_output>=std4_high].index

cta = Celltype_Analyzer([Celltype('>=4*std ratioIO', std4_high_skids), Celltype('>=2*std ratioIO', std2_high_skids)])
cta.set_known_types(celltypes)
cta.plot_memberships(path='plots/high-axonic-IO-ratio.pdf', figsize=(1,1))

print(f'There are {len(std4_high_skids)} neurons >=4*std axonic IO ratio')
print(f'There are {len(std2_high_skids)} neurons 2-4*std axonic IO ratio')

# %%
# where are these neurons in the clusters?

def plot_marginal_cell_type_cluster(size, particular_cell_type, particular_color, cluster_level, path, all_celltypes=None, ylim=(0,1), yticks=([])):

    # all cell types plot data
    if(all_celltypes==None):
        _, all_celltypes = Celltype_Analyzer.default_celltypes()
        
    clusters = Celltype_Analyzer.get_skids_from_meta_annotation(f'mw brain clusters level {cluster_level}', split=True, return_celltypes=True)
    cluster_analyze = Celltype_Analyzer(clusters)

    cluster_analyze.set_known_types(all_celltypes)
    celltype_colors = [x.get_color() for x in cluster_analyze.get_known_types()]
    all_memberships = cluster_analyze.memberships()
    all_memberships = all_memberships.iloc[[0,1,2,3,4,5,6,7,8,9,10,11,12,17,13,14,15,16], :] # switching order so unknown is not above outputs and RGNs before pre-outputs
    celltype_colors = [celltype_colors[i] for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,17,13,14,15,16]] # switching order so unknown is not above outputs and RGNs before pre-outputs
    
    # particular cell type data
    cluster_analyze.set_known_types([particular_cell_type])
    membership = cluster_analyze.memberships()

    # plot
    fig = plt.figure(figsize=size) 
    fig.subplots_adjust(hspace=0.1)
    gs = plt.GridSpec(4, 1)

    ax = fig.add_subplot(gs[0:3, 0])
    ind = np.arange(0, len(cluster_analyze.Celltypes))
    ax.bar(ind, membership.iloc[0, :], color=particular_color)
    ax.set(xlim = (-1, len(ind)), ylim=ylim, xticks=([]), yticks=yticks, title=particular_cell_type.get_name())

    ax = fig.add_subplot(gs[3, 0])
    ind = np.arange(0, len(cluster_analyze.Celltypes))
    ax.bar(ind, all_memberships.iloc[0, :], color=celltype_colors[0])
    bottom = all_memberships.iloc[0, :]
    for i in range(1, len(all_memberships.index)):
        plt.bar(ind, all_memberships.iloc[i, :], bottom = bottom, color=celltype_colors[i])
        bottom = bottom + all_memberships.iloc[i, :]
    ax.set(xlim = (-1, len(ind)), ylim=(0,1), xticks=([]), yticks=([]))
    ax.axis('off')
    ax.axis('off')

    plt.savefig(path, format='pdf', bbox_inches='tight')

cluster_level = 7
size = (2,0.5)
adj_names = ['ad', 'aa', 'dd', 'da']
_, celltypes = Celltype_Analyzer.default_celltypes()
all_high_skids = np.concatenate([std2_high_skids, std4_high_skids])

plot_marginal_cell_type_cluster(size, Celltype(f'>=2*SD Axonic Input/Output Ratio', all_high_skids), 'gray', cluster_level, f'plots/high-axonicIO_clusters{cluster_level}.pdf', all_celltypes = celltypes)

# %%
# ratio of dendritic output / input

den_output_input = input_output[input_output.dendrite_input>0]
den_output_input = den_output_input.dendrite_output/den_output_input.dendrite_input
den_output_input = den_output_input[~np.isnan(den_output_input)]
den_output_input = den_output_input.loc[np.intersect1d(brain_neurons, den_output_input.index)]

den_output_input.loc[865151] = 0 # fixed issue with axon split point
den_output_input = den_output_input.sort_values(ascending=True)

mean = np.mean(den_output_input)
median = np.median(den_output_input)
std = np.std(den_output_input)
std_low = mean - std
std_high = mean + std
std2_high = mean + std*2
std3_high = mean + std*3
std4_high = mean + std*4

print(f'The median axonic I/O ratio is {median:.3f}')
print(f'The mean axonic I/O ratio is {mean:.3f}')

fig, ax = plt.subplots(1,1, figsize=(4,3))
sns.scatterplot(x=range(len(den_output_input)), y=den_output_input, ec='gray', fc='none', alpha=0.75, ax=ax)
plt.axhline(y=mean, color='gray', linewidth = 0.5)
plt.axhline(y=std2_high, color='gray', linewidth = 0.5)
plt.axhline(y=std4_high, color='gray', linewidth = 0.5)
plt.savefig('plots/all-cell_ratio-dendritic-OI.pdf', format='pdf', bbox_inches='tight')

# %%
# which neurons have highest and lowest IO ratio

_, celltypes = Celltype_Analyzer.default_celltypes()

std2_high_skids = den_output_input[(den_output_input>=std2_high) & (den_output_input<std4_high)].index
std4_high_skids = den_output_input[den_output_input>=std4_high].index

cta = Celltype_Analyzer([Celltype('>=4*std ratioOI', std4_high_skids), Celltype('>=2*std ratioOI', std2_high_skids)])
cta.set_known_types(celltypes)
cta.plot_memberships(path='plots/high-dendritic-OI-ratio.pdf', figsize=(1,1))

print(f'There are {len(std4_high_skids)} neurons >=4*std dendritic OI ratio')
print(f'There are {len(std2_high_skids)} neurons 2-4*std dendritic OI ratio')
print(f'There are {len(np.setdiff1d(den_output_input.index, np.r_[std2_high_skids, std4_high_skids]))} neurons <2*std dendritic OI ratio')

# %%
# where are these neurons in the clusters?

def plot_marginal_cell_type_cluster(size, particular_cell_type, particular_color, cluster_level, path, all_celltypes=None, ylim=(0,1), yticks=([])):

    # all cell types plot data
    if(all_celltypes==None):
        _, all_celltypes = Celltype_Analyzer.default_celltypes()
        
    clusters = Celltype_Analyzer.get_skids_from_meta_annotation(f'mw brain clusters level {cluster_level}', split=True, return_celltypes=True)
    cluster_analyze = Celltype_Analyzer(clusters)

    cluster_analyze.set_known_types(all_celltypes)
    celltype_colors = [x.get_color() for x in cluster_analyze.get_known_types()]
    all_memberships = cluster_analyze.memberships()
    all_memberships = all_memberships.iloc[[0,1,2,3,4,5,6,7,8,9,10,11,12,17,13,14,15,16], :] # switching order so unknown is not above outputs and RGNs before pre-outputs
    celltype_colors = [celltype_colors[i] for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,17,13,14,15,16]] # switching order so unknown is not above outputs and RGNs before pre-outputs
    
    # particular cell type data
    cluster_analyze.set_known_types([particular_cell_type])
    membership = cluster_analyze.memberships()

    # plot
    fig = plt.figure(figsize=size) 
    fig.subplots_adjust(hspace=0.1)
    gs = plt.GridSpec(4, 1)

    ax = fig.add_subplot(gs[0:3, 0])
    ind = np.arange(0, len(cluster_analyze.Celltypes))
    ax.bar(ind, membership.iloc[0, :], color=particular_color)
    ax.set(xlim = (-1, len(ind)), ylim=ylim, xticks=([]), yticks=yticks, title=particular_cell_type.get_name())

    ax = fig.add_subplot(gs[3, 0])
    ind = np.arange(0, len(cluster_analyze.Celltypes))
    ax.bar(ind, all_memberships.iloc[0, :], color=celltype_colors[0])
    bottom = all_memberships.iloc[0, :]
    for i in range(1, len(all_memberships.index)):
        plt.bar(ind, all_memberships.iloc[i, :], bottom = bottom, color=celltype_colors[i])
        bottom = bottom + all_memberships.iloc[i, :]
    ax.set(xlim = (-1, len(ind)), ylim=(0,1), xticks=([]), yticks=([]))
    ax.axis('off')
    ax.axis('off')

    plt.savefig(path, format='pdf', bbox_inches='tight')

cluster_level = 7
size = (2,0.5)
_, celltypes = Celltype_Analyzer.default_celltypes()
all_high_skids = np.concatenate([std2_high_skids, std4_high_skids])

plot_marginal_cell_type_cluster(size, Celltype(f'>=2*SD Dendritic Output/Input Ratio', all_high_skids), 'gray', cluster_level, f'plots/high-dendriticOI_clusters{cluster_level}.pdf', all_celltypes = celltypes)

# %%
# DNs for different behaviours within clusters

candidate_behaviours_cts = Celltype_Analyzer.get_skids_from_meta_annotation('mw dVNC candidate behaviors', split=True, return_celltypes=True)

for celltype in candidate_behaviours_cts:
    plot_marginal_cell_type_cluster(size, celltype, 'gray', cluster_level, f'plots/DN-behaviours_{celltype.name.replace("mw ", "")}_clusters{cluster_level}.pdf', all_celltypes = celltypes)


# %%
# test role of APL/MBIN/KC
to_remove = pymaid.get_skids_by_annotation(['mw APL', 'mw MBIN', 'mw KC'])

fig, ax = plt.subplots(1,1, figsize=(1,1))
sns.barplot(y=axon_input_output.loc[np.setdiff1d(axon_input_output.index, to_remove)], errorbar='sd', ax=ax)
ax.set(ylim=(0,.5))
plt.savefig('plots/mean-std_axonicIOratio_noKCs-MBINs-APL.pdf', format='pdf', bbox_inches='tight')

# %%
# ratio of axon input / output per celltypes

axon_input_output = pd.DataFrame(axon_input_output, columns=['ratioIO'])
_, celltypes = Celltype_Analyzer.default_celltypes()

skid_ct = []
for skid in axon_input_output.index:
    for celltype in celltypes:
        if(skid in celltype.skids):
            skid_ct.append([skid, axon_input_output.loc[skid, 'ratioIO'], celltype.name])
            
df_io = pd.DataFrame(skid_ct, columns=['skid', 'ratioIO', 'celltype'])

plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'arial'

fig, ax = plt.subplots(1,1,figsize=(8,4))
sns.barplot(data=df_io, x='celltype', y='ratioIO', ax=ax)
plt.xticks(rotation=45, ha='right')
plt.savefig('plots/axonic-input-out-ratio_per_celltypes.png', bbox_inches='tight', format='png')

# prep to save as CSV
df_io_csv = df_io.sort_values('ratioIO', ascending=False, ignore_index=True).loc[:, ['skid', 'celltype', 'ratioIO']]
df_io_csv.columns = ['skid', 'celltype', 'axonic_input-output_ratio']

# replace some outdated names, dVNCs -> DN-VNCs; dSEZs -> DN-SEZs; FFNs -> MB-FFNs
# save as CSV
to_replace = {
  'dVNCs': 'DN-VNCs',
  'dSEZs': 'DN-SEZs',
  'FFNs': 'MB-FFNs',
  'pre-dSEZs': 'pre-DN-SEZs',
  'pre-dVNCs': 'pre-DN-VNCs'
}

for i, name in enumerate(df_io_csv.celltype):
    if(name in to_replace.keys()):
        df_io_csv.loc[i, 'celltype'] = to_replace[df_io_csv.loc[i, 'celltype']]

df_io_csv.to_csv('plots/single-cell_axonic-input-output-ratio.csv', index=False)


# %%
# do KCs have most aa connections per output count?

aa_output_frac = adj_aa.sum(axis=1)
aa_output_frac = aa_output_frac/axon_outputs.axon_output.loc[aa_output_frac.index]
aa_output_frac = aa_output_frac.dropna()

skid_ct = []
for skid in aa_output_frac.index:
    for celltype in celltypes:
        if(skid in celltype.skids):
            skid_ct.append([skid, aa_output_frac.loc[skid], celltype.name])
            
df_aa_frac = pd.DataFrame(skid_ct, columns=['skid', 'aa_frac_output', 'celltype'])
sns.barplot(data = df_aa_frac, x='celltype', y='aa_frac_output')