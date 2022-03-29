#%%
from pymaid_creds import url, name, password, token
from data_settings import pairs_path, data_date_A1_brain
import pymaid
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date

from contools import Promat, Celltype, Celltype_Analyzer

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

rm = pymaid.CatmaidInstance(url, token, name, password)
select_neurons = pymaid.get_skids_by_annotation(['mw A1 neurons paired', 'mw dVNC'])
select_neurons = select_neurons + Celltype_Analyzer.get_skids_from_meta_annotation('mw A1 sensories')
ad_edges_A1 = Promat.pull_edges(type_edges='ad', threshold=0.01, data_date=data_date_A1_brain, pairs_combined=False, select_neurons=select_neurons)
pairs = Promat.get_pairs(pairs_path=pairs_path)

A1_neurons = pymaid.get_skids_by_annotation('mw A1 neurons paired') + Celltype_Analyzer.get_skids_from_meta_annotation('mw A1 sensories')
# %%
# load skids

dVNC = pymaid.get_skids_by_annotation('mw dVNC')
A1 = pymaid.get_skids_by_annotation('mw A1 neurons paired')
#A1_acess = pymaid.get_skids_by_annotation('mw A1 accessory neurons') # adds some non-A1 neurons of interest

A1_MN = pymaid.get_skids_by_annotation('mw A1 MN')

A1_proprio = pymaid.get_skids_by_annotation('mw A1 proprio')
A1_chordotonal = pymaid.get_skids_by_annotation('mw A1 chordotonals')
A1_noci = pymaid.get_skids_by_annotation('mw A1 noci')
A1_classII_III = pymaid.get_skids_by_annotation('mw A1 classII_III')
A1_external = pymaid.get_skids_by_annotation('mw A1 external sensories')
A1_unk = pymaid.get_skids_by_annotation('mw A1 unknown sensories')

A1_interneurons = list(np.setdiff1d(pymaid.get_skids_by_annotation('mw A1 neurons paired'), Celltype_Analyzer.get_skids_from_meta_annotation('mw A1 sensories')))
A1_interneurons = list(np.setdiff1d(A1_interneurons, A1_MN))
# %%

# VNC layering with respect to sensories or motorneurons
threshold = 0.01
hops = 2

# manual add neuron groups of interest here and names
names = ['us-MN', 'ds-Proprio', 'ds-Noci', 'ds-Chord', 'ds-ClassII_III', 'ds-ES', 'ds-unk']
general_names = ['pre-MN', 'Proprio', 'Noci', 'Chord', 'ClassII_III', 'ES', 'unk']
exclude = A1_MN + A1_proprio + A1_chordotonal + A1_noci + A1_classII_III + A1_external + A1_unk + dVNC

us_MN = Promat.upstream_multihop(edges=ad_edges_A1, sources=A1_MN, hops=hops, exclude=exclude, pairs=pairs)
ds_proprio = Promat.downstream_multihop(edges=ad_edges_A1, sources=A1_proprio, hops=hops, exclude=exclude, pairs=pairs)
ds_noci = Promat.downstream_multihop(edges=ad_edges_A1, sources=A1_noci, hops=hops, exclude=exclude, pairs=pairs)
ds_chord = Promat.downstream_multihop(edges=ad_edges_A1, sources=A1_chordotonal, hops=hops, exclude=exclude, pairs=pairs)
ds_classII_III = Promat.downstream_multihop(edges=ad_edges_A1, sources=A1_classII_III, hops=hops, exclude=exclude, pairs=pairs)
ds_external = Promat.downstream_multihop(edges=ad_edges_A1, sources=A1_external, hops=hops, exclude=exclude, pairs=pairs)
ds_unk = Promat.downstream_multihop(edges=ad_edges_A1, sources=A1_unk, hops=hops, exclude=exclude, pairs=pairs)
ds_all_sens = Promat.downstream_multihop(edges=ad_edges_A1, sources=A1_proprio + A1_chordotonal + A1_noci + A1_classII_III + A1_external + A1_unk, hops=hops, exclude=exclude, pairs=pairs)

VNC_layers = [us_MN, ds_proprio, ds_noci, ds_chord, ds_classII_III, ds_external, ds_unk]
cat_order = ['pre-MN', 'Proprio', 'Noci', 'Chord', 'ClassII_III', 'ES', 'Unknown']

# %%
# identify neurons downstream of each dVNC-to-A1

exclude = A1_proprio + A1_chordotonal + A1_noci + A1_classII_III + A1_external + A1_unk + dVNC

hops = 1
dVNC_to_A1_pairs = Promat.load_pairs_from_annotation('mw dVNC to A1', pairs)
ds_dVNCs = [Promat.downstream_multihop(edges=ad_edges_A1, sources=list(dVNC_to_A1_pairs.loc[i, :].values), hops=hops, exclude=exclude, pairs=pairs)[0] for i in dVNC_to_A1_pairs.index]

# %%
# make a hop matrix plot for ds partners of dVNCs

from contools import Adjacency_matrix
VNC_layers = [[A1_MN]+us_MN, [A1_proprio]+ds_proprio, [A1_noci]+ds_noci, [A1_chordotonal]+ds_chord, [A1_classII_III]+ds_classII_III, [A1_external]+ds_external, [A1_unk]+ds_unk]
cat_order = ['pre-MN', 'Proprio', 'Noci', 'Chord', 'ClassII_III', 'ES', 'Unknown']

ds_dVNC_A1 = pymaid.get_skids_by_annotation('mw A1 ds_dVNC')
ds_dVNC_layers,ds_dVNC_A1_skids = Celltype_Analyzer.layer_id(VNC_layers, cat_order, ds_dVNC_A1)
ds_dVNC_A1_pairs = Promat.extract_pairs_from_list(ds_dVNC_A1, pairs)[0]
ds_dVNC_mat, ds_dVNC_mat_plotting = Promat.hop_matrix(ds_dVNC_A1_skids.T, cat_order, ds_dVNC_A1_pairs.leftid, include_start=True)

# identify ds_dVNCs based on identified types to sort hop_matrix
exclude = A1_proprio + A1_chordotonal + A1_noci + A1_classII_III + A1_external + A1_unk + dVNC

dVNC_motor = Promat.load_pairs_from_annotation('mw dVNC to A1 motor', pairs)
dVNC_sens = Promat.load_pairs_from_annotation('mw dVNC to A1 sens', pairs)
dVNC_mixed = Promat.load_pairs_from_annotation('mw dVNC to A1 mixed', pairs)

hops = 1
ds_dVNCs_motor = [Promat.downstream_multihop(edges=ad_edges_A1, sources=list(dVNC_motor.loc[i, :].values), hops=hops, exclude=exclude, pairs=pairs)[0] for i in dVNC_motor.index]
ds_dVNC_sens = [Promat.downstream_multihop(edges=ad_edges_A1, sources=list(dVNC_sens.loc[i, :].values), hops=hops, exclude=exclude, pairs=pairs)[0] for i in dVNC_sens.index]
ds_dVNC_mixed = [Promat.downstream_multihop(edges=ad_edges_A1, sources=list(dVNC_mixed.loc[i, :].values), hops=hops, exclude=exclude, pairs=pairs)[0] for i in dVNC_mixed.index]

ds_dVNC_order = ds_dVNCs_motor + ds_dVNC_sens + ds_dVNC_mixed

ds_dVNC_order_pairid = [list(Promat.extract_pairs_from_list(x, pairs)[0].leftid) for x in ds_dVNC_order]
ds_dVNC_order_pairid_order = [x for sublist in ds_dVNC_order_pairid for x in sublist]

dVNC_ordered = list(dVNC_motor.leftid) + list(dVNC_sens.leftid) + list(dVNC_mixed.leftid)
dVNC_types = ['motor']*len(dVNC_motor) + ['sens']*len(dVNC_sens) + ['mixed']*len(dVNC_mixed)

ds_dVNC_order_cols = []
for i, skids in enumerate(ds_dVNC_order_pairid):
    dVNC_i = dVNC_ordered[i]
    col = list(zip([dVNC_types[i]]*len(skids), [dVNC_i]*len(skids), skids))
    ds_dVNC_order_cols.append(col)

ds_dVNC_order_cols = [x for sublist in ds_dVNC_order_cols for x in sublist]
ds_dVNC_order_cols = pd.MultiIndex.from_tuples(ds_dVNC_order_cols, names=['dVNC-type','dVNC', 'ds-dVNC'])

ds_dVNC_mat_plot_expanded = ds_dVNC_mat_plotting.loc[:, ds_dVNC_order_pairid_order]
ds_dVNC_mat_plot_expanded.columns = ds_dVNC_order_cols

fig, ax = plt.subplots(1,1,figsize=(10,10))
sns.heatmap(ds_dVNC_mat_plot_expanded, square=True, ax=ax)
plt.savefig('plots/dVNC-A1_hop-matrix.pdf', format='pdf', bbox_inches='tight')

# %%
# determine motor/sens/mixed based on dVNC-ds_dVNC hop_matrix

# set up hop matrix with multiindex columns
ds_dVNC_order_cols = []
for i, skids in enumerate(ds_dVNC_order_pairid):
    dVNC_i = dVNC_ordered[i]
    col = list(zip([dVNC_i]*len(skids), skids))
    ds_dVNC_order_cols.append(col)

ds_dVNC_order_cols = [x for sublist in ds_dVNC_order_cols for x in sublist]
ds_dVNC_order_cols = pd.MultiIndex.from_tuples(ds_dVNC_order_cols, names=['dVNC', 'ds-dVNC'])

# set up multiindex columns on both dfs
ds_dVNC_hop_matrix = ds_dVNC_mat.loc[:, ds_dVNC_order_pairid_order]
ds_dVNC_hop_matrix.columns = ds_dVNC_order_cols
ds_dVNC_mat_plot_expanded = ds_dVNC_mat_plotting.loc[:, ds_dVNC_order_pairid_order]
ds_dVNC_mat_plot_expanded.columns = ds_dVNC_order_cols

# determine motor/sens character of downstream partners

sens_names = ['Proprio', 'Noci', 'Chord', 'ClassII_III', 'ES', 'Unknown']
dVNC_cols = pd.unique([x[0] for x in ds_dVNC_order_cols]) # pd.unique() preserves order

# generate multiindex columns with motor/sens/mixed info for ds_dVNC at bottom and dVNC at top
multiindex_col = []
for dVNC in dVNC_cols:

    multiindex = []

    ds_dVNCs = ds_dVNC_hop_matrix.loc[:, dVNC].columns
    for ds_dVNC in ds_dVNCs:
        label = 'unknown'
        motor = ds_dVNC_hop_matrix.loc['pre-MN', (dVNC, ds_dVNC)]==1
        sens = (ds_dVNC_hop_matrix.loc[sens_names, (dVNC, ds_dVNC)]==1).sum(axis=0)>0
        
        if(motor): label = 'motor'
        if(sens): label = 'sens'
        if(motor & sens): label = 'mixed'

        if((motor==False) & (sens==False)):
            motor = ds_dVNC_hop_matrix.loc['pre-MN', (dVNC, ds_dVNC)]==2
            sens = (ds_dVNC_hop_matrix.loc[sens_names, (dVNC, ds_dVNC)]==2).sum(axis=0)>0   

            if(motor): label = 'motor'
            if(sens): label = 'sens'
            if(motor & sens): label = 'mixed'

        multiindex.append([dVNC, ds_dVNC, label])

    motor_labels = sum([True if x[2]=='motor' else False for x in multiindex])/len(multiindex)
    sens_labels = sum([True if x[2]=='sens' else False for x in multiindex])/len(multiindex)
    mixed_labels = sum([True if x[2]=='mixed' else False for x in multiindex])/len(multiindex)
    unknown_labels = sum([True if x[2]=='unknown' else False for x in multiindex])/len(multiindex)

    if((motor_labels==1.0) | (motor_labels+unknown_labels==1.0)): dVNC_label = 'motor'
    if((sens_labels==1.0) | ((sens_labels+unknown_labels)==1.0)): dVNC_label = 'sens'
    if((mixed_labels>0.0) | ((sens_labels+unknown_labels<1.0) & (motor_labels+unknown_labels<1.0))): dVNC_label = 'mixed'
    if(unknown_labels==1.0): dVNC_label = 'unknown'

    if(dVNC == 15639294): # exception for dVNC that talks directly to MN
        dVNC_label = 'motor'

    col = [[dVNC_label] + x for x in multiindex]
    multiindex_col.append(col)

multiindex_col = [x for sublist in multiindex_col for x in sublist]
multiindex_col = pd.MultiIndex.from_tuples(multiindex_col, names=['dVNC-type', 'dVNC', 'ds-dVNC', 'ds-dVNC-type'])

# plot 
ds_dVNC_hop_matrix.columns = multiindex_col
ds_dVNC_mat_plot_expanded.columns = multiindex_col

# add in connection to motorneuron
ds_dVNC_mat_plot_expanded.loc['pre-MN', (slice(None), 15639294)]=3.0

# dVNC sort
dVNC_sort = [15639294, 17359501, 10553248, 10728328, 8723983,
                11013583, 19361427, 17053270, 17777031, 7227010, 16851496,
                ] + [x[1] for x in  multiindex_col if x[0]=='mixed']

fig, ax = plt.subplots(1,1,figsize=(5,5))
annots = ds_dVNC_hop_matrix.loc[:, (slice(None), dVNC_sort)].astype(int).astype(str)
annots[annots=='0']=''
sns.heatmap(ds_dVNC_mat_plot_expanded.loc[:, (slice(None), dVNC_sort)], annot=annots, fmt='s', square=True, ax=ax, cmap='Blues')
plt.savefig('plots/dVNC-A1_hop-matrix.pdf', format='pdf', bbox_inches='tight')

# %%
# export dVNC types

# identify and annotate dVNC types
dVNC_motor = [x[1] for x in multiindex_col if x[0]=='motor']
dVNC_sens = [x[1] for x in multiindex_col if x[0]=='sens']
dVNC_mixed = [x[1] for x in multiindex_col if x[0]=='mixed']

dVNC_motor = Promat.get_paired_skids(dVNC_motor, pairs, unlist=True)
dVNC_sens = Promat.get_paired_skids(dVNC_sens, pairs, unlist=True)
dVNC_mixed = Promat.get_paired_skids(dVNC_mixed, pairs, unlist=True)

pymaid.add_annotations(dVNC_motor, 'mw dVNC to A1 motor')
pymaid.add_annotations(dVNC_sens, 'mw dVNC to A1 sens')
pymaid.add_annotations(dVNC_mixed, 'mw dVNC to A1 mixed')

# identify and annotate ds_dVNCs types
ds_dVNC_motor = [x[2] for x in multiindex_col if x[0]=='motor']
ds_dVNC_sens = [x[2] for x in multiindex_col if x[0]=='sens']
ds_dVNC_mixed = [x[2] for x in multiindex_col if x[0]=='mixed']

ds_dVNC_motor = Promat.get_paired_skids(ds_dVNC_motor, pairs, unlist=True)
ds_dVNC_sens = Promat.get_paired_skids(ds_dVNC_sens, pairs, unlist=True)
ds_dVNC_mixed = Promat.get_paired_skids(ds_dVNC_mixed, pairs, unlist=True)

pymaid.add_annotations(ds_dVNC_motor, 'mw A1 ds_dVNC motor')
pymaid.add_annotations(ds_dVNC_sens, 'mw A1 ds_dVNC sens')
pymaid.add_annotations(ds_dVNC_mixed, 'mw A1 ds_dVNC mixed')
# %%
