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
print(f'The median axonic I/O ratio is {np.median(axon_input_output):.3f}')

fig, ax = plt.subplots(1,1)
sns.scatterplot(x=range(len(axon_input_output)), y=axon_input_output.ratioIO, ec='gray', fc='none', alpha=0.75, ax=ax)

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