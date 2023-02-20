# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
from data_settings import data_date, pairs_path
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

from contools import Celltype, Celltype_Analyzer, Promat, Analyze_Cluster

# %%
# 

_, celltypes = Celltype_Analyzer.default_celltypes()
pairs = Promat.get_pairs(pairs_path=pairs_path)

all_neurons = pymaid.get_skids_by_annotation(['mw brain and inputs', 'mw brain accessory neurons'])
#remove_neurons = pymaid.get_skids_by_annotation(['mw brain very incomplete', 'mw partially differentiated', 'mw motor'])
remove_neurons = pymaid.get_skids_by_annotation(['mw motor'])
all_neurons = list(np.setdiff1d(all_neurons, remove_neurons)) # remove neurons that are incomplete or partially differentiated (as well as SEZ motor neurons)

pairs_df = Promat.load_pairs_from_annotation(annot=[], pairList=pairs, return_type='all_pair_ids_bothsides', skids=all_neurons, use_skids=True)

left = pymaid.get_skids_by_annotation('mw left')
right = pymaid.get_skids_by_annotation('mw right')

# %%
# pairs in clusters

lvl=7
clusters = Analyze_Cluster(cluster_lvl=lvl, meta_data_path='data/graphs/meta_data.csv', skids=all_neurons, sort='signal_flow')
mb_nomen = pymaid.get_annotated('MB nomenclature')
mb_nomen = [x for x in list(mb_nomen.name) if ('FB-' not in x)&('LH-' not in x)&('MB2IN-' not in x)&('MB2ON-' not in x)]
mb_nomen_cts = [Celltype(annot.replace('FB2IN-', 'FB2N-'), pymaid.get_skids_by_annotation(annot)) for annot in mb_nomen]

sens_meta_annotations = pymaid.get_annotated('mw brain sensory modalities').name
sens_meta_cts = [Celltype(annot.replace('mw ', ''), Celltype_Analyzer.get_skids_from_meta_annotation(annot)) for annot in sens_meta_annotations]

pair_loops_meta = pymaid.get_annotated('mw pair loops').name
pair_loops_cts = [Celltype(annot, pymaid.get_skids_by_annotation(annot)) for annot in pair_loops_meta]

PN_meta = pymaid.get_annotated('mw brain PNs').name
PN_cts = [Celltype(annot.replace('mw ', ''), pymaid.get_skids_by_annotation(annot)) for annot in PN_meta]

PNsomato_meta = pymaid.get_annotated('mw brain PNs-somato').name
PNsomato_cts = [Celltype(annot.replace('mw ', ''), pymaid.get_skids_by_annotation(annot)) for annot in PNsomato_meta]

unknown_ascending_ct = [Celltype('unknown modality', pymaid.get_skids_by_annotation('mw A1 ascending unknown'))]
pdiff_ct = [Celltype('partially differentiated', pymaid.get_skids_by_annotation('mw partially differentiated'))]

annotated_cts = mb_nomen_cts + sens_meta_cts + pair_loops_cts + PN_cts + PNsomato_cts + unknown_ascending_ct + pdiff_ct

# update names
updated_names = ['sensory',
                'PN',
                'ascending',
                'PN-somato',
                'LN',
                'LHN',
                'MB-FFN',
                'MBIN',
                'KC',
                'MBON',
                'MB-FBN',
                'CN',
                'pre-DN-SEZ',
                'pre-DN-VNC',
                'RGN',
                'DN-SEZ',
                'DN-VNC']

for i in range(len(celltypes)):
    celltypes[i].name = updated_names[i]

skids_celltypes = [skid for sublist in [x.skids for x in celltypes] for skid in sublist]
skids_other = np.setdiff1d(all_neurons, skids_celltypes)
celltypes = celltypes + [Celltype('other', skids_other)]

data = []
for i in pairs_df.index:
    for celltype in celltypes:
        if(pairs_df.loc[i].leftid in celltype.skids):
            if(pairs_df.loc[i].leftid!=pairs_df.loc[i].rightid): # for paired neurons
                data.append([pairs_df.loc[i].leftid, pairs_df.loc[i].rightid, celltype.name])
            if(pairs_df.loc[i].leftid==pairs_df.loc[i].rightid): # for nonpaired
                if(pairs_df.loc[i].leftid in left):
                    data.append([pairs_df.loc[i].leftid, 'no pair', celltype.name])
                if(pairs_df.loc[i].leftid in right):
                    data.append(['no pair', pairs_df.loc[i].leftid, celltype.name])

df = clusters.cluster_df
for i in range(len(data)):
    for j in df.index:
        if((data[i][0] in df.loc[j].skids)|(data[i][1] in df.loc[j].skids)):
            #data[i] = data[i] + [f'{df.index[j]}_level-7_clusterID-{int(df.cluster[j])}']
            data[i] = data[i] + [f'{df.index[j]}']

    if(len(data[i])==3):
        data[i] = data[i] + ['no cluster']    

for i in range(len(data)):
    skid_left = data[i][0]
    skid_right = data[i][1]

    annotated_name = 'no official annotation'
    skid_left_name = 'no pair'
    skid_right_name = 'no pair'

    if(skid_left!='no pair'): 
        skid_left_name = pymaid.get_names(skid_left)[str(skid_left)]

    if(skid_right!='no pair'): 
        skid_right_name = pymaid.get_names(skid_right)[str(skid_right)]

    annotated_list=[]
    for annotated_celltype in annotated_cts:
        if((skid_left in annotated_celltype.skids)|(skid_right in annotated_celltype.skids)):
            annotated_list.append(annotated_celltype.name)
    
    if(len(annotated_list)>0):
        annotated_name = '; '.join(annotated_list)

    data[i] = data[i] + [annotated_name, skid_left_name, skid_right_name]

neurons_meta_df = pd.DataFrame(data, columns=['leftid', 'rightid', 'celltype', 'cluster', 'annotated_name', 'left_name', 'right_name'])

from natsort import natsort_keygen

neurons_meta_df = neurons_meta_df.loc[:, ['leftid', 'rightid', 'celltype', 'annotated_name', 'left_name', 'right_name', 'cluster']]
neurons_meta_df = neurons_meta_df.sort_values(by='cluster', key=natsort_keygen())
neurons_meta_df = neurons_meta_df.reset_index(drop=True)

# add entry for [3813487, 17068730]
# these are center neurons (neither left nor right) and aren't properly handled by the script
added = pd.DataFrame([[3813487, 3813487, 'MBIN', 'OAN-a2', 'Ladder (olfactory) post.', 'Ladder (olfactory) post.', 'no cluster'], 
            [17068730, 17068730, 'MBIN', 'OAN-a1', 'Ladder (olfactory) ant.', 'Ladder (olfactory) ant.', 'no cluster']],
            columns = neurons_meta_df.columns)

neurons_meta_df = pd.concat([neurons_meta_df, added])
neurons_meta_df = neurons_meta_df.reset_index(drop=True)
neurons_meta_df.to_csv('plots/brain-neurons_meta-data.csv', index=False)

# %%
# check

neurons_meta_df.leftid[neurons_meta_df.leftid!='no pair']
neurons_meta_df.rightid[neurons_meta_df.rightid!='no pair']

np.setdiff1d(np.unique(list(neurons_meta_df.leftid[neurons_meta_df.leftid!='no pair']) + list(neurons_meta_df.rightid[neurons_meta_df.rightid!='no pair'])), all_neurons)

# %%
