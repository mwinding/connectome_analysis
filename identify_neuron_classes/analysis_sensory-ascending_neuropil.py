#%%
import os
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import connectome_tools.cluster_analysis as clust
import connectome_tools.celltype as ct
import connectome_tools.process_graph as pg
import connectome_tools.process_matrix as pm
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
order = ['ORN', 'AN sensories', 'MN sensories', 'photoreceptors', 'thermosensories', 'v\'td', 'A1 ascending noci', 'A1 ascending mechano', 'A1 ascending proprio', 'A1 ascending class II_III']
order2_names = [f'mw brain 2nd_order {celltype}' for celltype in order]
order3_names = [f'mw brain 3rd_order {celltype}' for celltype in order]

order2 = [pymaid.get_skids_by_annotation(annot) for annot in order2_names]
order3 = [pymaid.get_skids_by_annotation(annot) for annot in order3_names]

order2_ct = ct.Celltype_Analyzer([ct.Celltype(order2_names[i].replace('mw brain ', ''), skids) for i, skids in enumerate(order2)])
order3_ct = ct.Celltype_Analyzer([ct.Celltype(order3_names[i].replace('mw brain ', ''), skids) for i, skids in enumerate(order3)])

# upset plots of 2nd/3rd order centers
#   returned values are Celltypes of upset plot partitions 
order2_cats = order2_ct.upset_members(path='identify_neuron_classes/plots/2nd-order_upset-plot', plot_upset=True)
order3_cats = order3_ct.upset_members(path='identify_neuron_classes/plots/3rd-order_upset-plot', plot_upset=True)

# %%
# manually generate Sankey-like plots
# use numbers extracted here for sizes of each bar
order = ['ORN', 'AN sensories', 'MN sensories', 'photoreceptors', 'thermosensories', 'v\'td', 'A1 ascending noci', 'A1 ascending mechano', 'A1 ascending proprio', 'A1 ascending class II_III']
sens = [ct.Celltype(name, pymaid.get_skids_by_annotation(f'mw {name}')) for name in order]
order2 = [ct.Celltype(f'2nd_order {name}', pymaid.get_skids_by_annotation(f'mw brain 2nd_order {name}')) for name in order]
order3 = [ct.Celltype(f'3rd_order {name}', pymaid.get_skids_by_annotation(f'mw brain 3rd_order {name}')) for name in order]

LNs = pymaid.get_skids_by_annotation('mw brain LNs')
RGN = pymaid.get_skids_by_annotation('mw RGN')
dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')
dVNC = pymaid.get_skids_by_annotation('mw dVNC')

columns = []
for i in range(0, len(sens)):
    sens_len = len(sens[i].get_skids())

    order2_LN = len(np.intersect1d(order2[i].get_skids(), LNs))
    order2_RGN = len(np.intersect1d(order2[i].get_skids(), RGN))
    order2_dSEZ = len(np.intersect1d(order2[i].get_skids(), dSEZ))
    order2_dVNC = len(np.intersect1d(order2[i].get_skids(), dVNC))
    order2_other = len(np.setdiff1d(order2[i].get_skids(), LNs + RGN + dSEZ + dVNC))

    order3_LN = len(np.intersect1d(order3[i].get_skids(), LNs))
    order3_RGN = len(np.intersect1d(order3[i].get_skids(), RGN))
    order3_dSEZ = len(np.intersect1d(order3[i].get_skids(), dSEZ))
    order3_dVNC = len(np.intersect1d(order3[i].get_skids(), dVNC))
    order3_other = len(np.setdiff1d(order3[i].get_skids(), LNs + RGN + dSEZ + dVNC))

    print(f'{order[i]}:')
    print(f'sensory count: {sens_len}')
    print(f'2nd_order count: LNs - {order2_LN}, other - {order2_other}, RGN - {order2_RGN}, dSEZ - {order2_dSEZ}, dVNC - {order2_dVNC}')
    print(f'3rd_order count: LNs - {order3_LN}, other - {order3_other}, RGN - {order3_RGN}, dSEZ - {order3_dSEZ}, dVNC - {order3_dVNC}')
    print('')

    columns.append([sens_len, 0, 0, 0, 0, 0])
    columns.append([0, order2_LN, order2_other, order2_RGN, order2_dSEZ, order2_dVNC])
    columns.append([0, order3_LN, order3_other, order3_RGN, order3_dSEZ, order3_dVNC])

df = pd.DataFrame(columns, columns = ['Sens', 'LN', 'Other', 'RGN', 'dSEZ', 'dVNC'])

spacer = 10
width = 0.4
fig, ax = plt.subplots(1,1,figsize=(15,6))
ax.bar(x = df.index, height = df['Sens'], width = width)
ax.bar(x = df.index, height = df['LN'], width = width)
ax.bar(x = df.index, height = df['Other'], bottom = df['LN']+ spacer, width = width)
ax.bar(x = df.index, height = df['RGN'], bottom = df['LN'] + df['Other']+ spacer*2, width = width)
ax.bar(x = df.index, height = df['dSEZ'], bottom = df['LN'] + df['Other'] + df['RGN']+ spacer*3, width = width)
ax.bar(x = df.index, height = df['dVNC'], bottom = df['LN'] + df['Other'] + df['RGN'] + df['dSEZ']+ spacer*4, width = width)
ax.axis('off')

plt.savefig('identify_neuron_classes/plots/source_for_sankey_plot.pdf', bbox_inches='tight', format = 'pdf')

# plot fraction of each type
row_sum = df.sum(axis=1).values

for i in range(0, len(df)):
    df.iloc[i, :] = df.iloc[i, :]/row_sum[i]

fig, ax = plt.subplots(1,1,figsize=(15,6))
ax.bar(x = df.index, height = df['Sens'])
ax.bar(x = df.index, height = df['LN'])
ax.bar(x = df.index, height = df['Other'], bottom = df['LN'])
ax.bar(x = df.index, height = df['RGN'], bottom = df['LN'] + df['Other'])
ax.bar(x = df.index, height = df['dSEZ'], bottom = df['LN'] + df['Other'] + df['RGN'])
ax.bar(x = df.index, height = df['dVNC'], bottom = df['LN'] + df['Other'] + df['RGN'] + df['dSEZ'])
ax.axis('off')

plt.savefig('identify_neuron_classes/plots/fraction-LNs-and-outputs_per_neuropil.pdf', bbox_inches='tight', format = 'pdf')

# %%
# known cell types per 2nd/3rd order
order = ['ORN', 'AN sensories', 'MN sensories', 'photoreceptors', 'thermosensories', 'v\'td', 'A1 ascending noci', 'A1 ascending mechano', 'A1 ascending proprio', 'A1 ascending class II_III']
sens = [ct.Celltype(name, pymaid.get_skids_by_annotation(f'mw {name}')) for name in order]
order2 = [ct.Celltype(f'2nd_order {name}', pymaid.get_skids_by_annotation(f'mw brain 2nd_order {name}')) for name in order]
order3 = [ct.Celltype(f'3rd_order {name}', pymaid.get_skids_by_annotation(f'mw brain 3rd_order {name}')) for name in order]

sens_cta = ct.Celltype_Analyzer(sens)
order2_cta = ct.Celltype_Analyzer(order2)
order3_cta = ct.Celltype_Analyzer(order3)

celltypes_data, celltypes = ct.Celltype_Analyzer.default_celltypes() # will have to add a way of removing particular groups from this list in the future

# intercalated 2nd/3rd order identities
intercalated_order2_3 = [x for sublist in list(zip(order2, order3)) for x in sublist]
intercalated_cta = ct.Celltype_Analyzer(intercalated_order2_3)
intercalated_cta.set_known_types(celltypes)
memberships = intercalated_cta.memberships(raw_num=True).drop('sensories').T

memberships['Total'] = memberships.sum(axis=1)
column_order = [ 'Total', 'LNs', 'PNs', 'LHNs', 'FFNs', 'MBINs', 'KCs', 'MBONs', 'MB-FBNs', 'CNs', 'ascendings', 'pre-dSEZs', 'pre-dVNCs', 'RGNs', 'dSEZs', 'dVNCs', 'unknown']
memberships = memberships.loc[:, column_order]
memberships = memberships.astype(int)

sns.heatmap(memberships, annot=True, fmt='d', cmap='Greens', cbar=False)

# plot 2nd order identities
order2_cta.set_known_types(celltypes)
memberships = order2_cta.memberships(raw_num=True).drop('sensories').T

memberships['Total'] = memberships.sum(axis=1)
column_order = [ 'Total', 'LNs', 'PNs', 'LHNs', 'FFNs', 'MBINs', 'KCs', 'MBONs', 'MB-FBNs', 'CNs', 'ascendings', 'pre-dSEZs', 'pre-dVNCs', 'RGNs', 'dSEZs', 'dVNCs', 'unknown']
memberships = memberships.loc[:, column_order]
memberships = memberships.astype(int)

annotations = memberships.astype(str)
annotations[annotations=='0']=''

fig, ax = plt.subplots(1,1, figsize=(3,2))
sns.heatmap(memberships.iloc[:, 1:], annot=annotations.iloc[:, 1:], fmt='s', cmap='Greens', cbar=False, ax=ax)
plt.savefig('identify_neuron_classes/plots/cell-identities_2nd_order.pdf', bbox_inches='tight', format = 'pdf')

# plot 3rd order identities
order3_cta.set_known_types(celltypes)
memberships = order3_cta.memberships(raw_num=True).drop('sensories').T

memberships['Total'] = memberships.sum(axis=1)
column_order = [ 'Total', 'LNs', 'PNs', 'LHNs', 'FFNs', 'MBINs', 'KCs', 'MBONs', 'MB-FBNs', 'CNs', 'ascendings', 'pre-dSEZs', 'pre-dVNCs', 'RGNs', 'dSEZs', 'dVNCs', 'unknown']
memberships = memberships.loc[:, column_order]
memberships = memberships.astype(int)

annotations = memberships.astype(str)
annotations[annotations=='0']=''

fig, ax = plt.subplots(1,1, figsize=(3,2))
sns.heatmap(memberships.iloc[:, 1:], annot=annotations.iloc[:, 1:], fmt='s', cmap='Greens', cbar=False, ax=ax)
plt.savefig('identify_neuron_classes/plots/cell-identites_3rd-order.pdf', bbox_inches='tight', format = 'pdf')

# %%
# adjacency matrix of all types

brain_inputs = [x for sublist in [pymaid.get_skids_by_annotation(annot) for annot in pymaid.get_annotated('mw brain inputs and ascending').name] for x in sublist]
brain = pymaid.get_skids_by_annotation('mw brain neurons') + brain_inputs

adj_names = ['ad', 'aa', 'dd', 'da']
adj_ad, adj_aa, adj_dd, adj_da = [pd.read_csv(f'data/adj/all-neurons_{name}.csv', index_col = 0).rename(columns=int) for name in adj_names]
adj_ad = adj_ad.loc[np.intersect1d(adj_ad.index, brain), np.intersect1d(adj_ad.index, brain)]
adj_aa = adj_aa.loc[np.intersect1d(adj_aa.index, brain), np.intersect1d(adj_aa.index, brain)]
adj_dd = adj_dd.loc[np.intersect1d(adj_dd.index, brain), np.intersect1d(adj_dd.index, brain)]
adj_da = adj_da.loc[np.intersect1d(adj_da.index, brain), np.intersect1d(adj_da.index, brain)]

neuron_types = sens + order2 + order3
neuron_types_skids = [x.get_skids() for x in neuron_types]

mat = adj_ad
summed_mat = pd.DataFrame(np.zeros(shape=(len(neuron_types_skids),len(neuron_types_skids))),
                                    index = [x.get_name() for x in neuron_types],
                                    columns = [x.get_name() for x in neuron_types])
for i, skids_i in enumerate(neuron_types_skids):
    for j, skids_j in enumerate(neuron_types_skids):
        skids_i = np.intersect1d(skids_i, mat.index)
        skids_j = np.intersect1d(skids_j, mat.index)
        sum_value = mat.loc[skids_i, skids_j].sum(axis=1).sum()
        sum_value = sum_value/len(skids_j)
        summed_mat.iloc[i, j] = sum_value

fig, ax = plt.subplots(1,1,figsize=(5,5))
sns.heatmap(summed_mat.iloc[10:, 10:], square=True, vmax=75)
plt.savefig('identify_neuron_classes/plots/connectivity-between-neuropils.pdf', bbox_inches='tight', format = 'pdf')

# %%
# Sankey plot characterizing number of LNs, outputs, etc. per 2nd/3rd order neurons
'''
import plotly.graph_objects as go # Import the graphical object
from plotly.offline import plot

node_label = ['sens0', 'sens1', 'sens0order2', 'sens01order2', 'sens1order2-1', 'sens1order2-2']
source_node = [0, 0, 1, 1, 1]
target_node = [2, 3, 3, 4, 5]
values = [10, 2, 3, 5, 6]

fig = go.Figure( 
    data=[go.Sankey( # The plot we are interest
        # This part is for the node information
        node = dict( 
            label = node_label
        ),
        # This part is for the link information
        link = dict(
            source = source_node,
            target = target_node,
            value = values
        ))])

# And shows the plot
fig.show()
plt.savefig('identify_neuron_classes/plots/sens-order_types.pdf', bbox_inches='tight')

import plotly.graph_objects as go

# %%
# custom bar plot for cell types

fig = go.Figure()
'''
fig.add_trace(go.Scatter(
    x=[1.5, 4.5],
    y=[0.75, 0.75],
    text=["Unfilled Rectangle", "Filled Rectangle"],
    mode="text",
))
'''
# Set axes properties
fig.update_xaxes(range=[0, 55], showgrid=False)
fig.update_yaxes(range=[0, 8], showgrid=False)

# Add shapes
fig.add_shape(type="rect",
    x0=1, y0=1, x1=22, y1=2,
    line=dict(color="RoyalBlue"), 
    fillcolor="RoyalBlue"
)

fig.add_shape(type="rect",
    x0=1, y0=4, x1=10, y1=5,
    line=dict(
        color="RoyalBlue",
        width=2,
    ),
    fillcolor="RoyalBlue",
)

fig.add_shape(type="rect",
    x0=11, y0=4, x1=50, y1=5,
    line=dict(
        color="RoyalBlue",
        width=2,
    ),
    fillcolor="RoyalBlue",
)

fig.update_shapes(dict(xref='x', yref='y'))
fig.show()

# %%
# testing plot

fig, ax = plt.subplots(1,1,figsize=(4,4))
ax.set(xlim=(0,50), ylim=(0,5))
rect = plt.Rectangle((1,1), 21, 1, fc='gray')
rect2 = plt.Rectangle((1,3), 14, 1, fc='gray')
rect3 = plt.Rectangle((16,3), 20, 1, fc='gray')

ax.add_patch(rect)
ax.add_patch(rect2)
ax.add_patch(rect3)

plt.show()
'''
# %%
# adjaceny matrix of 2nd/3rd order sensory centers (wiring diagram?)

# %%
# cascades through sensory centers