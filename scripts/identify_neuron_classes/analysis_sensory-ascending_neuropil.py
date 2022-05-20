#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
from data_settings import data_date, pairs_path
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

from contools import Celltype, Celltype_Analyzer, Promat
import navis

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

# %%
# load neuron types based on annotations

# warning: hardcoded names to get order correct
# could be problem if more are added
order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-warm', 'thermo-cold', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']
order2_names = [f'mw {celltype} 2nd_order' for celltype in order]
order3_names = [f'mw {celltype} 3rd_order' for celltype in order]
order4_names = [f'mw {celltype} 4th_order' for celltype in order]

order2 = [pymaid.get_skids_by_annotation(annot) for annot in order2_names]
order3 = [pymaid.get_skids_by_annotation(annot) for annot in order3_names]
order4 = [pymaid.get_skids_by_annotation(annot) for annot in order4_names]

order2_ct = Celltype_Analyzer([Celltype(order2_names[i].replace('mw ', ''), skids) for i, skids in enumerate(order2)])
order3_ct = Celltype_Analyzer([Celltype(order3_names[i].replace('mw ', ''), skids) for i, skids in enumerate(order3)])
order4_ct = Celltype_Analyzer([Celltype(order4_names[i].replace('mw ', ''), skids) for i, skids in enumerate(order4)])

# upset plots of 2nd/3rd order centers
#   returned values are Celltypes of upset plot partitions 
order2_cats = order2_ct.upset_members(path='identify_neuron_classes/plots/2nd-order_upset-plot', plot_upset=True, threshold=10, exclude_singletons_from_threshold=True)
order3_cats = order3_ct.upset_members(path='identify_neuron_classes/plots/3rd-order_upset-plot', plot_upset=True, threshold=10, exclude_singletons_from_threshold=True)
order4_cats = order4_ct.upset_members(path='identify_neuron_classes/plots/4th-order_upset-plot', plot_upset=True, threshold=10, exclude_singletons_from_threshold=True)

# %%
# manually generate Sankey-like plots
# use numbers extracted here for sizes of each bar
order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-warm', 'thermo-cold', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']
sens = [Celltype(name, Celltype_Analyzer.get_skids_from_meta_annotation(f'mw {name}')) for name in order]
order2 = [Celltype(f'{name} 2nd_order', pymaid.get_skids_by_annotation(f'mw {name} 2nd_order')) for name in order]
order3 = [Celltype(f'{name} 3rd_order', pymaid.get_skids_by_annotation(f'mw {name} 3rd_order')) for name in order]
order4 = [Celltype(f'{name} 4th_order', pymaid.get_skids_by_annotation(f'mw {name} 4th_order')) for name in order]

LNs = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain LNs')
LNs_o = list(np.setdiff1d(pymaid.get_skids_by_annotation('mw LNs_cohort'), pymaid.get_skids_by_annotation('mw LNs_noncohort')))
LNs_io = list(np.setdiff1d(pymaid.get_skids_by_annotation('mw LNs_noncohort'), pymaid.get_skids_by_annotation('mw LNs_cohort')))
LNs_both = list(np.intersect1d(pymaid.get_skids_by_annotation('mw LNs_cohort'), pymaid.get_skids_by_annotation('mw LNs_noncohort')))

KC = pymaid.get_skids_by_annotation('mw KC')
MBON = pymaid.get_skids_by_annotation('mw MBON')
MBIN = pymaid.get_skids_by_annotation('mw MBIN')
MB = KC + MBON + MBIN

RGN = pymaid.get_skids_by_annotation('mw RGN')
dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')
dVNC = pymaid.get_skids_by_annotation('mw dVNC')

columns = []
columns_expanded = []
for i in range(0, len(sens)):
    sens_len = len(sens[i].get_skids())

    order2_LN = len(np.intersect1d(order2[i].get_skids(), LNs))
    order2_LN_o = len(np.intersect1d(order2[i].get_skids(), LNs_o))
    order2_LN_io = len(np.intersect1d(order2[i].get_skids(), LNs_io))
    order2_LN_both = len(np.intersect1d(order2[i].get_skids(), LNs_both))
    order2_MB = len(np.intersect1d(order2[i].get_skids(), MB))
    order2_RGN = len(np.intersect1d(order2[i].get_skids(), RGN))
    order2_dSEZ = len(np.intersect1d(order2[i].get_skids(), dSEZ))
    order2_dVNC = len(np.intersect1d(order2[i].get_skids(), dVNC))
    order2_other = len(np.setdiff1d(order2[i].get_skids(), LNs + MB + RGN + dSEZ + dVNC))
    order2_other_expanded = len(np.setdiff1d(order2[i].get_skids(), LNs + MB + RGN + dSEZ + dVNC))

    order3_LN = len(np.intersect1d(order3[i].get_skids(), LNs))
    order3_LN_o = len(np.intersect1d(order3[i].get_skids(), LNs_o))
    order3_LN_io = len(np.intersect1d(order3[i].get_skids(), LNs_io))
    order3_LN_both = len(np.intersect1d(order3[i].get_skids(), LNs_both))
    order3_MB = len(np.intersect1d(order3[i].get_skids(), MB))
    order3_RGN = len(np.intersect1d(order3[i].get_skids(), RGN))
    order3_dSEZ = len(np.intersect1d(order3[i].get_skids(), dSEZ))
    order3_dVNC = len(np.intersect1d(order3[i].get_skids(), dVNC))
    order3_other = len(np.setdiff1d(order3[i].get_skids(), LNs + MB + RGN + dSEZ + dVNC))
    order3_other_expanded = len(np.setdiff1d(order3[i].get_skids(), LNs + MB + RGN + dSEZ + dVNC))

    order4_LN = len(np.intersect1d(order4[i].get_skids(), LNs))
    order4_LN_o = len(np.intersect1d(order4[i].get_skids(), LNs_o))
    order4_LN_io = len(np.intersect1d(order4[i].get_skids(), LNs_io))
    order4_LN_both = len(np.intersect1d(order4[i].get_skids(), LNs_both))
    order4_MB = len(np.intersect1d(order4[i].get_skids(), MB))
    order4_RGN = len(np.intersect1d(order4[i].get_skids(), RGN))
    order4_dSEZ = len(np.intersect1d(order4[i].get_skids(), dSEZ))
    order4_dVNC = len(np.intersect1d(order4[i].get_skids(), dVNC))
    order4_other = len(np.setdiff1d(order4[i].get_skids(), LNs + MB + RGN + dSEZ + dVNC))
    order4_other_expanded = len(np.setdiff1d(order4[i].get_skids(), LNs + MB + RGN + dSEZ + dVNC))

    print(f'{order[i]}:')
    print(f'sensory count: {sens_len}')
    print(f'2nd_order count: LNs - {order2_LN}, other - {order2_other}, RGN - {order2_RGN}, dSEZ - {order2_dSEZ}, dVNC - {order2_dVNC}')
    print(f'3rd_order count: LNs - {order3_LN}, other - {order3_other}, RGN - {order3_RGN}, dSEZ - {order3_dSEZ}, dVNC - {order3_dVNC}')
    print(f'4th_order count: LNs - {order4_LN}, other - {order4_other}, RGN - {order4_RGN}, dSEZ - {order4_dSEZ}, dVNC - {order4_dVNC}')
    print('')

    columns.append([sens_len, 0, 0, 0, 0, 0])
    columns.append([0, order2_LN, order2_MB, order2_other, order2_RGN, order2_dSEZ, order2_dVNC])
    columns.append([0, order3_LN, order3_MB, order3_other, order3_RGN, order3_dSEZ, order3_dVNC])
    columns.append([0, order4_LN, order4_MB, order4_other, order4_RGN, order4_dSEZ, order4_dVNC])
    columns.append([0, 0, 0, 0, 0, 0]) # add space between each modality in subsequent plots

    # for more complicated plots with all sub-LN types
    columns_expanded.append([0, order2_LN_o, order2_LN_io, order2_LN_both, order2_MB, order2_other, order2_RGN, order2_dSEZ, order2_dVNC])
    columns_expanded.append([0, order3_LN_o, order3_LN_io, order3_LN_both, order3_MB, order3_other, order3_RGN, order3_dSEZ, order3_dVNC])
    columns_expanded.append([0, order4_LN_o, order4_LN_io, order4_LN_both, order4_MB, order4_other, order4_RGN, order4_dSEZ, order4_dVNC])
    columns_expanded.append([0, 0, 0, 0, 0, 0]) # add space between each modality in subsequent plots

df = pd.DataFrame(columns, columns = ['Sens', 'LN', 'MB', 'Other', 'RGN', 'dSEZ', 'dVNC'])
df_expanded = pd.DataFrame(columns_expanded, columns = ['Sens', 'LN_o', 'LN_io', 'LN_both', 'MB', 'Other', 'RGN', 'dSEZ', 'dVNC'])

spacer = 10
width = 0.4
fig, ax = plt.subplots(1,1,figsize=(15,6))
ax.bar(x = df.index, height = df['Sens'], width = width)
ax.bar(x = df.index, height = df['LN'], width = width)
ax.bar(x = df.index, height = df['Other'], bottom = df['LN']+ spacer, width = width)
ax.bar(x = df.index, height = df['MB'], bottom = df['LN']+ df['Other']+spacer*2, width = width)
ax.bar(x = df.index, height = df['RGN'], bottom = df['LN'] + df['Other'] + df['MB'] + spacer*3, width = width)
ax.bar(x = df.index, height = df['dSEZ'], bottom = df['LN'] + df['Other'] + df['RGN'] + df['MB'] + spacer*4, width = width)
ax.bar(x = df.index, height = df['dVNC'], bottom = df['LN'] + df['Other'] + df['RGN'] + df['dSEZ'] + df['MB'] + spacer*5, width = width)
ax.axis('off')

plt.savefig('identify_neuron_classes/plots/source_for_sankey_plot.pdf', bbox_inches='tight', format = 'pdf')

# plot counts of each type
row_sum = df.sum(axis=1).values

df=df.fillna(0) # fill NaN from divide by 0 with 0

fig, ax = plt.subplots(1,1,figsize=(15,6))
ax.bar(x = df.index, height = df['Sens'], color = '#007640')
ax.bar(x = df.index, height = df['LN'], color = '#4F7577')
ax.bar(x = df.index, height = df['Other'], bottom = df['LN'], color = '#00ADEE')
ax.bar(x = df.index, height = df['MB'], bottom = df['LN']+ df['Other'], color = '#C1C1C1')
ax.bar(x = df.index, height = df['RGN'], bottom = df['LN'] + df['Other'] + df['MB'], color = '#9167AB')
ax.bar(x = df.index, height = df['dSEZ'], bottom = df['LN'] + df['Other'] + df['RGN'] + df['MB'], color = '#D77F51')
ax.bar(x = df.index, height = df['dVNC'], bottom = df['LN'] + df['Other'] + df['RGN'] + df['dSEZ'] + df['MB'], color = '#A5292A')

plt.savefig('identify_neuron_classes/plots/counts_LNs-and-outputs_per_neuropil.pdf', bbox_inches='tight', format = 'pdf')

# plot fraction of each type
row_sum = df_expanded.sum(axis=1).values

for i in range(0, len(df_expanded)):
    df_expanded.iloc[i, :] = df_expanded.iloc[i, :]/row_sum[i]

df_expanded=df_expanded.fillna(0) # fill NaN from divide by 0 with 0
df = df_expanded

fig, ax = plt.subplots(1,1,figsize=(15,6))
ax.bar(x = df.index, height = df['LN_o'], color = '#4F7577')
ax.bar(x = df.index, height = df['LN_io'], bottom = df['LN_o'], color = '#2E5151')
ax.bar(x = df.index, height = df['LN_both'], bottom = df['LN_o'] + df['LN_io'], color = '#7FBFBF')
ax.bar(x = df.index, height = df['Other'], bottom = df['LN_o'] + df['LN_io'] + df['LN_both'], color = '#00ADEE')
ax.bar(x = df.index, height = df['MB'], bottom = df['LN_o'] + df['LN_io'] + df['LN_both'] + df['Other'], color = '#C1C1C1')
ax.bar(x = df.index, height = df['RGN'], bottom = df['LN_o'] + df['LN_io'] + df['LN_both'] + df['MB'] + df['Other'], color = '#9167AB')
ax.bar(x = df.index, height = df['dSEZ'], bottom = df['LN_o'] + df['LN_io'] + df['LN_both'] + df['MB'] + df['Other'] + df['RGN'], color='#D77F51')
ax.bar(x = df.index, height = df['dVNC'], bottom = df['LN_o'] + df['LN_io'] + df['LN_both'] + df['MB'] + df['Other'] + df['RGN'] + df['dSEZ'], color='#A5292A')
ax.axis('off')

plt.savefig('identify_neuron_classes/plots/fraction-LNs-and-outputs_per_neuropil.pdf', bbox_inches='tight', format = 'pdf')

# %% 
# modality layers LH vs MB

order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-warm', 'thermo-cold', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']
inputs = [Celltype(name, Celltype_Analyzer.get_skids_from_meta_annotation(f'mw {name}')) for name in order]
order2 = [Celltype(f'{name} 2nd_order', pymaid.get_skids_by_annotation(f'mw {name} 2nd_order')) for name in order]
order3 = [Celltype(f'{name} 3rd_order', pymaid.get_skids_by_annotation(f'mw {name} 3rd_order')) for name in order]
order4 = [Celltype(f'{name} 4th_order', pymaid.get_skids_by_annotation(f'mw {name} 4th_order')) for name in order]

sens = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain sensories')
asc = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain ascendings')
LN = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain LNs')
PN = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain PNs')
PNsomato = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain PNs-somato')
MBIN = pymaid.get_skids_by_annotation('mw MBIN')
KC = pymaid.get_skids_by_annotation('mw KC')
MBON = pymaid.get_skids_by_annotation('mw MBON')
LHN = pymaid.get_skids_by_annotation('mw LHN')
RGN = pymaid.get_skids_by_annotation('mw RGN')
dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')
dVNC = pymaid.get_skids_by_annotation('mw dVNC')

selected_celltypes = [sens, asc, LN, PN, PNsomato, LHN, MBIN, KC, MBON, RGN, dSEZ, dVNC]

# make mutually exclusive
selected_celltypes_exclus = []
for i in range(len(selected_celltypes)):
    if(i==0):
        selected_celltypes_exclus.append(selected_celltypes[i])
    else:
        all_previous = selected_celltypes[:i]
        all_previous = [x for sublist in all_previous for x in sublist]
        exclusive = list(np.setdiff1d(selected_celltypes[i], all_previous))
        selected_celltypes_exclus.append(exclusive)


selected_names = ['sensories', 'ascendings', 'LNs','PNs', 'PNs-somato', 'LHNs', 'MBINs', 'KCs', 'MBONs', 'RGNs', 'dSEZs', 'dVNCs']
selected_colors = ['#00753f','#a0ddf2','#4f7577','#1d79b7','#21cef7','#d4e29e', '#ff8734','#e55560','#f9eb4d','#9467bd','#d88052','#a52a2a']

selected_celltypes_ct = [Celltype(name, selected_celltypes_exclus[i], color=selected_colors[i]) for i, name in enumerate(selected_names)]

order_cts = [[inputs[i], order2[i], order3[i], order4[i], Celltype(f'spacer{i}', [])] for i in range(len(order))]
order_cts = [x for sublist in order_cts for x in sublist]
order_cts = Celltype_Analyzer(order_cts)
order_cts.set_known_types(selected_celltypes_ct)
memberships = order_cts.memberships(raw_num=True).loc[['sensories', 'ascendings', 'LNs','PNs', 'PNs-somato', 'LHNs', 'MBINs', 'KCs', 'MBONs', 'unknown', 'RGNs', 'dSEZs', 'dVNCs'], :]
colors = selected_colors + ['#a6a8ab']
colors = [colors[i] for i in [0,1,2,3,4,5,6,7,8,12,9,10,11]]
order_cts.plot_memberships('identify_neuron_classes/plots/fraction-selected-celltypes_sensory-orders.pdf', (15,6), memberships = memberships, raw_num=True, celltype_colors=colors)

# %%
# known cell types per 2nd/3rd order
order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-warm', 'thermo-cold', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']
sens = [Celltype(name, pymaid.get_skids_by_annotation(f'mw {name}')) for name in order]
order2 = [Celltype(f'{name} 2nd_order', pymaid.get_skids_by_annotation(f'mw {name} 2nd_order')) for name in order]
order3 = [Celltype(f'{name} 3rd_order', pymaid.get_skids_by_annotation(f'mw {name} 3rd_order')) for name in order]
order4 = [Celltype(f'{name} 4th_order', pymaid.get_skids_by_annotation(f'mw {name} 4th_order')) for name in order]

sens_cta = Celltype_Analyzer(sens)
order2_cta = Celltype_Analyzer(order2)
order3_cta = Celltype_Analyzer(order3)
order4_cta = Celltype_Analyzer(order4)

celltypes_data, celltypes = Celltype_Analyzer.default_celltypes() # will have to add a way of removing particular groups from this list in the future

# intercalated 2nd/3rd order identities
intercalated = [x for sublist in list(zip(order2, order3, order4)) for x in sublist]
intercalated_cta = Celltype_Analyzer(intercalated)
intercalated_cta.set_known_types(celltypes)
memberships = intercalated_cta.memberships(raw_num=True).drop('sensories').T

memberships['Total'] = memberships.sum(axis=1)
column_order = [ 'Total', 'LNs', 'PNs', 'LHNs', 'FFNs', 'MBINs', 'KCs', 'MBONs', 'MB-FBNs', 'CNs', 'ascendings', 'pre-dSEZs', 'pre-dVNCs', 'RGNs', 'dSEZs', 'dVNCs', 'unknown']
memberships = memberships.loc[:, column_order]
memberships = memberships.astype(int)

sns.heatmap(memberships, annot=True, fmt='d', cmap='Greens', cbar=False)

# plot 2nd order identities
order2_cta.set_known_types(celltypes)
memberships = order2_cta.memberships(raw_num=True).drop(['sensories', 'ascendings']).T

memberships['Total'] = memberships.sum(axis=1)
column_order = [ 'Total', 'LNs', 'PNs', 'LHNs', 'FFNs', 'MBINs', 'KCs', 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'RGNs', 'dSEZs', 'dVNCs', 'unknown']
memberships = memberships.loc[:, column_order]
memberships = memberships.astype(int)

annotations = memberships.astype(str)
annotations[annotations=='0']=''

fig, ax = plt.subplots(1,1, figsize=(2.35,1.25))
sns.heatmap(memberships.iloc[:, 1:], annot=annotations.iloc[:, 1:], fmt='s', cmap='Greens', cbar=False, ax=ax)
plt.savefig('identify_neuron_classes/plots/cell-identities_2nd_order.pdf', bbox_inches='tight', format = 'pdf')

# plot 3rd order identities
order3_cta.set_known_types(celltypes)
memberships = order3_cta.memberships(raw_num=True).drop(['sensories', 'ascendings']).T

memberships['Total'] = memberships.sum(axis=1)
column_order = [ 'Total', 'LNs', 'PNs', 'LHNs', 'FFNs', 'MBINs', 'KCs', 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'RGNs', 'dSEZs', 'dVNCs', 'unknown']
memberships = memberships.loc[:, column_order]
memberships = memberships.astype(int)

annotations = memberships.astype(str)
annotations[annotations=='0']=''

fig, ax = plt.subplots(1,1, figsize=(2.35,1.25))
sns.heatmap(memberships.iloc[:, 1:], annot=annotations.iloc[:, 1:], fmt='s', cmap='Greens', cbar=False, ax=ax)
plt.savefig('identify_neuron_classes/plots/cell-identites_3rd-order.pdf', bbox_inches='tight', format = 'pdf')

# plot 4th order identities
order4_cta.set_known_types(celltypes)
memberships = order4_cta.memberships(raw_num=True).drop(['sensories', 'ascendings']).T

memberships['Total'] = memberships.sum(axis=1)
column_order = [ 'Total', 'LNs', 'PNs', 'LHNs', 'FFNs', 'MBINs', 'KCs', 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'RGNs', 'dSEZs', 'dVNCs', 'unknown']
memberships = memberships.loc[:, column_order]
memberships = memberships.astype(int)

annotations = memberships.astype(str)
annotations[annotations=='0']=''

fig, ax = plt.subplots(1,1, figsize=(2.35,1.25))
sns.heatmap(memberships.iloc[:, 1:], annot=annotations.iloc[:, 1:], fmt='s', cmap='Greens', cbar=False, ax=ax)
plt.savefig('identify_neuron_classes/plots/cell-identites_4th-order.pdf', bbox_inches='tight', format = 'pdf')

# celltype identities not split by modality (4th- and 5th-order)
order4_all = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain inputs 4th_order')
order5_all = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain inputs 5th_order')

order45_cta = Celltype_Analyzer([Celltype('4th_order', order4_all), Celltype('5th_order', order5_all)])
order45_cta.set_known_types(celltypes)
memberships = order45_cta.memberships(raw_num=True).drop(['sensories', 'ascendings']).T

memberships['Total'] = memberships.sum(axis=1)
column_order = [ 'Total', 'LNs', 'PNs', 'LHNs', 'FFNs', 'MBINs', 'KCs', 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'RGNs', 'dSEZs', 'dVNCs', 'unknown']
memberships = memberships.loc[:, column_order]
memberships = memberships.astype(int)

annotations = memberships.astype(str)
annotations[annotations=='0']=''

fig, ax = plt.subplots(1,1, figsize=(2.35,1.25/6))
sns.heatmap(memberships.iloc[:, 1:], annot=annotations.iloc[:, 1:], fmt='s', cmap='Greens', cbar=False, ax=ax)
plt.savefig('identify_neuron_classes/plots/cell-identites_4th-5th-all-order.pdf', bbox_inches='tight', format = 'pdf')

# %%
# adjacency matrix of all types

brain_inputs = Celltype_Analyzer.get_skids_from_meta_meta_annotation('mw brain sensory modalities')
brain = pymaid.get_skids_by_annotation('mw brain neurons') + list(np.unique([skid for sublist in [x.get_skids() for x in (order2 + order3 + order4)] for skid in sublist]))

adj_names = ['ad', 'aa', 'dd', 'da']
adj_ad, adj_aa, adj_dd, adj_da = [pd.read_csv(f'data/adj/all-neurons_{name}.csv', index_col = 0).rename(columns=int) for name in adj_names]
adj_ad = adj_ad.loc[np.intersect1d(adj_ad.index, brain), np.intersect1d(adj_ad.index, brain)]
adj_aa = adj_aa.loc[np.intersect1d(adj_aa.index, brain), np.intersect1d(adj_aa.index, brain)]
adj_dd = adj_dd.loc[np.intersect1d(adj_dd.index, brain), np.intersect1d(adj_dd.index, brain)]
adj_da = adj_da.loc[np.intersect1d(adj_da.index, brain), np.intersect1d(adj_da.index, brain)]
adjs = [adj_ad, adj_aa, adj_dd, adj_da]

vmaxs = [75, 30, 30, 10]
for i, adj in enumerate(adjs):
    neuron_types_cta = Celltype_Analyzer(order2 + order3 + [Celltype('4th_order', order4_all)] + [Celltype('5th_order', order5_all)])
    summed_mat = neuron_types_cta.connectivity(adj=adj, normalize_post_num=True)
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    sns.heatmap(summed_mat, square=True, vmax=vmaxs[i])
    plt.savefig(f'identify_neuron_classes/plots/connectivity-between-neuropils_{adj_names[i]}_vmax-{vmaxs[i]}.pdf', bbox_inches='tight', format = 'pdf')

# %%
# cascades through sensory neuropils

# load cascades generating in /cascades/multisensory_integration_cascades.py
import pickle

order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-warm', 'thermo-cold', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']
sens = [Celltype(name, pymaid.get_skids_by_annotation(f'mw {name}')) for name in order]
order2 = [Celltype(f'{name} 2nd_order', pymaid.get_skids_by_annotation(f'mw {name} 2nd_order')) for name in order]
order3 = [Celltype(f'{name} 3rd_order', pymaid.get_skids_by_annotation(f'mw {name} 3rd_order')) for name in order]
order4 = [Celltype(f'{name} 4th_order', pymaid.get_skids_by_annotation(f'mw {name} 4th_order')) for name in order]

n_init=1000
hops=8
input_hit_hist_list = pickle.load(open('data/cascades/sensory-modality-cascades_1000-n_init.p', 'rb'))

all_cta = Celltype_Analyzer(order2 + order3 + order4)

columns = []
for hit_hist in input_hit_hist_list:
    column = hit_hist.cascades_in_celltypes(cta=all_cta, hops=hops, n_init=n_init).visits_norm
    column.name = hit_hist.get_name()
    column.index = [x.get_name() for x in all_cta.Celltypes]
    columns.append(column)

cascades_neuropil = pd.concat(columns, axis=1)

fig, ax = plt.subplots(1,1, figsize=(4,1.5))
sns.heatmap(cascades_neuropil.T, ax=ax, cmap='Reds', vmax=1, square=True)
plt.savefig(f'identify_neuron_classes/plots/cascades-to-neuropils.pdf', bbox_inches='tight', format = 'pdf')

fig, ax = plt.subplots(1,1, figsize=(2,1))
sns.heatmap(cascades_neuropil.T.iloc[:, 0:len(order)*2], ax=ax, cmap='Reds', vmax=1, square=True)
plt.savefig(f'identify_neuron_classes/plots/cascades-to-neuropils_2nd-3rd-order.pdf', bbox_inches='tight', format = 'pdf')


# combine all 4th and 5th together
order4_all = Celltype('4th_order', Celltype_Analyzer.get_skids_from_meta_annotation('mw brain inputs 4th_order'))
order5_all = Celltype('5th_order', Celltype_Analyzer.get_skids_from_meta_annotation('mw brain inputs 5th_order'))

all_cta = Celltype_Analyzer(order2 + order3 + [order4_all] + [order5_all])

columns = []
for hit_hist in input_hit_hist_list:
    column = hit_hist.cascades_in_celltypes(cta=all_cta, hops=hops, n_init=n_init).visits_norm
    column.name = hit_hist.get_name()
    column.index = [x.get_name() for x in all_cta.Celltypes]
    columns.append(column)

cascades_neuropil = pd.concat(columns, axis=1)

fig, ax = plt.subplots(1,1, figsize=(4,1.5))
sns.heatmap(cascades_neuropil.T, ax=ax, cmap='Reds', vmax=1, square=True)
plt.savefig(f'identify_neuron_classes/plots/cascades-to-neuropils_combined4th-5th.pdf', bbox_inches='tight', format = 'pdf')

# %%
