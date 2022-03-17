#%%

from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import connectome_tools.process_matrix as pm
import connectome_tools.celltype as ct

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import numpy.random as random

import cmasher as cmr

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'


# load axodendritic adjacency matrix
adj = pm.Promat.pull_adj(type_adj = 'ad', subgraph = 'brain')

# load inputs and pair data
inputs = pd.read_csv('data/graphs/inputs.csv', index_col=0)
pairs = pm.Promat.get_pairs()

# load projectome 
projectome = pd.read_csv('data/projectome/projectome_mw brain paper all neurons_split.csv', index_col = 0, header = 0)

bilateral = pymaid.get_skids_by_annotation('mw bilateral axon')
# %%
# prep bilateral data; splitting output functions

left = pymaid.get_skids_by_annotation('mw left')
right = pymaid.get_skids_by_annotation('mw right')

# prep projectome for use comparing left vs right hemisphere axon outputs
is_left = []
for skid in projectome.skeleton:
    if(skid in left):
        is_left.append([skid, 1])
    if(skid in right):
        is_left.append([skid, 0])
    if((skid not in right) & (skid not in left)):
        is_left.append([skid, -1])
is_left = pd.DataFrame(is_left, columns = ['skid', 'is_left'])

projectome['is_left']=is_left.is_left.values
projectome['treenode']=projectome.index

connectors = projectome.set_index('connector')
skeletons = projectome.set_index(['skeleton', 'is_left', 'is_axon', 'is_input', 'Brain Hemisphere left', 'Brain Hemisphere right'])

# load paired bilateral neurons
bilateral = pymaid.get_skids_by_annotation('mw bilateral axon')
br = pymaid.get_skids_by_annotation('mw brain neurons')
outputs = ct.Celltype_Analyzer.get_skids_from_meta_annotation('mw brain outputs')

bilateral = list(np.intersect1d(bilateral, br))

#bilateral_no_brain_input = projectome.groupby('skeleton')['Brain Hemisphere left', 'Brain Hemisphere right'].sum()
#exclude = list(bilateral_no_brain_input[(bilateral_no_brain_input.loc[:, 'Brain Hemisphere left']==0) & (bilateral_no_brain_input.loc[:, 'Brain Hemisphere right']==0)].index)
bilateral = list(np.setdiff1d(bilateral, outputs))

bilateral_pairs = pm.Promat.extract_pairs_from_list(bilateral, pairs)[0]

def ad_edges(connector, projectome):
    connector_details = projectome[projectome.loc[:, 'connector']==connector]
    match = (connector_details.loc[:, ['is_input', 'is_axon']]==[1,0]).sum(axis=1)==2
    skids_target = list(connector_details[match].skeleton)
    return(skids_target)

def edge_counts(skeleton_connector_table, source_skid, connection_type):
    all_skeletons = []
    for connector in list(skeleton_connector_table):
        all_skeletons.append(ad_edges(connector, projectome))

    all_skeletons = [x for sublist in all_skeletons for x in sublist]
    skids = list(np.unique(all_skeletons))

    skid_counts = []
    for target_skid in skids:
        skid_counts.append([source_skid, target_skid, connection_type, sum([x==target_skid for x in all_skeletons])])

    skid_counts = pd.DataFrame(skid_counts, columns=['source', 'target', 'connection_type', 'synapses'])
    return(skid_counts)

def bilateral_ad_connections(leftid, rightid, skeletons, inputs, normalize=True):
    
    Nleft_to_Hright = skeletons.loc[(leftid, slice(None), 1, 0, 0, 1)].connector
    Nright_to_Hleft = skeletons.loc[(rightid, slice(None), 1, 0, 1, 0)].connector

    contra_l = edge_counts(Nleft_to_Hright, leftid, 'contralateral')
    contra_r = edge_counts(Nright_to_Hleft, rightid, 'contralateral')

    Nleft_to_Hleft = skeletons.loc[(leftid, slice(None), 1, 0, 1, 0)].connector
    Nright_to_Hright = skeletons.loc[(rightid, slice(None), 1, 0, 0, 1)].connector

    ipsi_l = edge_counts(Nleft_to_Hleft, leftid, 'ipsilateral')
    ipsi_r = edge_counts(Nright_to_Hright, rightid, 'ipsilateral')

    data = pd.concat([contra_l, contra_r, ipsi_l, ipsi_r], axis=0)
    data.reset_index(inplace=True, drop=True)

    if(normalize==True):
        norm_synapses=[]
        for i, skid in enumerate(data.target):
            norm_synapse = data.synapses[i]/inputs.loc[skid].dendrite_input
            norm_synapses.append(norm_synapse)

        data['synapses'] = norm_synapses 

    return(data)

# set threshold to -1 to not use a threshold; like this by default
def ipsi_contra_ds_partners(bilateral_leftid, bilateral_rightid, skeletons, inputs, threshold=-1, normalize=True):
    # look for ipsi and contra downstream partners over threshold
    bi_ad_connect = bilateral_ad_connections(bilateral_leftid, bilateral_rightid, skeletons, inputs, normalize=normalize)
    bi_ad_connect_index1 = bi_ad_connect.set_index(['connection_type', 'source'])
    if(len(bi_ad_connect_index1)==0):
        return([])

    # contralateral partners
    contra_sources = list(np.unique(bi_ad_connect_index1.loc[('contralateral')].index))
    if(bilateral_leftid not in contra_sources): print(f'skid {bilateral_leftid} has no a-d contralateral partners!')
    if(bilateral_rightid not in contra_sources): print(f'skid {bilateral_rightid} has no a-d contralateral partners!')
    contra_left_partners = bi_ad_connect_index1.loc[('contralateral')].target.loc[bilateral_leftid]
    contra_right_partners = bi_ad_connect_index1.loc[('contralateral')].target.loc[bilateral_rightid]

    if(type(contra_left_partners)==int): # hacky fix for issue where sometimes these variables are int and sometimes they are pd.Series
        contra_left_partners = [contra_left_partners]
    if(type(contra_left_partners)==pd.Series):
        contra_left_partners = list(contra_left_partners)

    if(type(contra_right_partners)==int):
        contra_right_partners = [contra_right_partners]
    if(type(contra_right_partners)==pd.Series):
        contra_right_partners = list(contra_right_partners)

    contra_pair_partners = pm.Promat.extract_pairs_from_list(np.unique(contra_left_partners + contra_right_partners), pairs)[0] # added np.unique to deal with issues with the few bilateral dendrites (MBONs)

    # identify and average contralateral inputs
    bi_ad_connect_index2 = bi_ad_connect.set_index(['connection_type', 'source', 'target'])
    contra_input = []
    for i in range(len(contra_pair_partners)):
        left_target = bi_ad_connect_index2.loc[('contralateral', slice(None), contra_pair_partners.leftid[i])].synapses.values[0]
        right_target = bi_ad_connect_index2.loc[('contralateral', slice(None), contra_pair_partners.rightid[i])].synapses.values[0]
        contra_input.append([bilateral_leftid, 'contralateral', contra_pair_partners.leftid[i], contra_pair_partners.rightid[i], 
                                left_target, right_target, (left_target+right_target)/2])

    # identify over-threshold contra-partners
    contra_input = pd.DataFrame(contra_input, columns = ['source_pairid', 'connection_type','leftid', 'rightid', 'left_input', 'right_input', 'average_input'])
    
    if(threshold!=-1):
        contra_input = contra_input[contra_input.average_input>threshold]


    # ipsilateral partners
    ipsi_sources = list(np.unique(bi_ad_connect_index1.loc[('ipsilateral')].index))
    if(bilateral_leftid not in ipsi_sources): print(f'skid {bilateral_leftid} has no a-d ipsilateral partners!')
    if(bilateral_rightid not in ipsi_sources): print(f'skid {bilateral_rightid} has no a-d ipsilateral partners!')
    ipsi_left_partners = bi_ad_connect_index1.loc[('ipsilateral')].target.loc[bilateral_leftid]
    ipsi_right_partners = bi_ad_connect_index1.loc[('ipsilateral')].target.loc[bilateral_rightid]

    if(type(ipsi_left_partners)==int): # hacky fix for issue where sometimes these variables are int and sometimes they are pd.Series
        ipsi_left_partners = [ipsi_left_partners]
    if(type(ipsi_left_partners)==pd.Series):
        ipsi_left_partners = list(ipsi_left_partners)

    if(type(ipsi_right_partners)==int):
        ipsi_right_partners = [ipsi_right_partners]
    if(type(ipsi_right_partners)==pd.Series):
        ipsi_right_partners = list(ipsi_right_partners)

    ipsi_pair_partners = pm.Promat.extract_pairs_from_list(np.unique(ipsi_left_partners + ipsi_right_partners), pairs)[0] # added np.unique to deal with issues with the few bilateral dendrites (MBONs)

    # identify and average ipsilateral inputs
    bi_ad_connect_index2 = bi_ad_connect.set_index(['connection_type', 'source', 'target'])
    ipsi_input = []
    for i in range(len(ipsi_pair_partners)):
        left_target = bi_ad_connect_index2.loc[('ipsilateral', slice(None), ipsi_pair_partners.leftid[i])].synapses.values[0]
        right_target = bi_ad_connect_index2.loc[('ipsilateral', slice(None), ipsi_pair_partners.rightid[i])].synapses.values[0]
        ipsi_input.append([bilateral_leftid, 'ipsilateral', ipsi_pair_partners.leftid[i], ipsi_pair_partners.rightid[i], 
                            left_target, right_target, (left_target+right_target)/2])

    # identify over-threshold ipsi-partners
    ipsi_input = pd.DataFrame(ipsi_input, columns = ['source_pairid', 'connection_type', 'leftid', 'rightid', 'left_input', 'right_input', 'average_input'])
    ipsi_input = ipsi_input[ipsi_input.average_input>threshold]

    data = pd.concat([ipsi_input, contra_input], axis=0)
    data.reset_index(inplace=True, drop=True)
    return(data)

# %%
# find partners

from tqdm import tqdm

issues_list = []
data_list = []
for i in tqdm(range(49, 50)):
    #try:
    data = ipsi_contra_ds_partners(bilateral_pairs.leftid[i], bilateral_pairs.rightid[i], skeletons, inputs, normalize=True)
    data_list.append(data)
    #except:
    #    print(f'problem with index {i}, skids {bilateral_pairs.leftid[i]}, {bilateral_pairs.rightid[i]}')
    #    issues_list.append(i)

filtered_data = [x for x in data_list if type(x)==pd.DataFrame]
data = pd.concat((x for x in filtered_data), axis=0)
data.reset_index(inplace=True, drop=True)
pair_ids = list(np.unique(data.source_pairid))

data = data.set_index(['source_pairid', 'connection_type'])

merged_list = []
for i in range(len(pair_ids)):
    if(('contralateral' in data.loc[pair_ids[i]].index) & ('ipsilateral' in data.loc[pair_ids[i]].index)): # super hacky, need to check the data
        ipsi_data = pd.DataFrame([data.loc[(pair_ids[i], 'ipsilateral')].average_input.values], index = [pair_ids[i]], columns=list(data.loc[(pair_ids[i], 'ipsilateral')].leftid.values))
        contra_data = pd.DataFrame([data.loc[(pair_ids[i], 'contralateral')].average_input.values], index = [f'{pair_ids[i]}-contra'], columns=list(data.loc[(pair_ids[i], 'contralateral')].leftid.values))

        merged_data = pd.concat([ipsi_data, contra_data], axis=0)
        merged_data.fillna(0, inplace=True)
        merged_list.append(merged_data)
# %%
# calculate cosine similarity between ipsi and contra connections in bilateral pairs

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)

    return(cos)

cos_list = []
for pair in merged_list:
    cos = cosine_similarity(list(pair.iloc[0,:]), list(pair.iloc[1,:]))
    cos_list.append([pair.index[0], cos])

cos_list = pd.DataFrame(cos_list, columns = ['pairid', 'similarity_ipsi_contra'])
# %%
# combine with "ipsi-contra" measure
# below is copy-paste from ipsi_contra_analysis.py

left = pymaid.get_skids_by_annotation('mw left')
right = pymaid.get_skids_by_annotation('mw right')

projectome = pd.read_csv('interhemisphere/data/projectome_mw_brain_matrix_A1_split.csv', index_col = 0, header = 0)

is_left = []
for skid in projectome.skeleton:
    if(skid in left):
        is_left.append([skid, 1])
    if(skid in right):
        is_left.append([skid, 0])
    if((skid not in right) & (skid not in left)):
        is_left.append([skid, -1])
is_left = pd.DataFrame(is_left, columns = ['skid', 'is_left'])

projectome['is_left']=is_left.is_left.values

import connectome_tools.process_matrix as pm

meshes_left = ['Brain Hemisphere left', 'SEZ_left', 'T1_left', 'T2_left', 'T3_left', 'A1_left', 'A2_left', 'A3_left', 'A4_left', 'A5_left', 'A6_left', 'A7_left', 'A8_left']
meshes_right = ['Brain Hemisphere right', 'SEZ_right', 'T1_right', 'T2_right', 'T3_right', 'A1_right', 'A2_right', 'A3_right', 'A4_right', 'A5_right', 'A6_right', 'A7_right', 'A8_right']
meshes = meshes_left + meshes_right

projectome['summed_left'] = projectome[meshes_left].sum(axis=1)
projectome['summed_right'] = projectome[meshes_right].sum(axis=1)
proj_group = projectome.groupby(['skeleton', 'is_left','is_axon', 'is_input'])['summed_left', 'summed_right'].sum()
#proj_group = projectome.groupby(['skeleton', 'is_left','is_axon', 'is_input'])['Brain Hemisphere left', 'Brain Hemisphere right'].sum()
br_pairs = pm.Promat.extract_pairs_from_list(br, pairs)[0]

right_contra_axon_outputs = proj_group.loc[(br, 0, 1, 0), :] # right side, axon outputs
left_contra_axon_outputs = proj_group.loc[(br, 1, 1, 0), :] # left side, axon outputs

right_contra_axon_outputs['ratio'] = right_contra_axon_outputs['summed_left']/(right_contra_axon_outputs['summed_left'] + right_contra_axon_outputs['summed_right'])
left_contra_axon_outputs['ratio'] = left_contra_axon_outputs['summed_right']/(left_contra_axon_outputs['summed_left'] + left_contra_axon_outputs['summed_right'])

br_pairs = pm.Promat.extract_pairs_from_list(br, pairs)[0]

contra_pairs = []
for i in range(len(br_pairs)):
    if((br_pairs.leftid[i] in [x[0] for x in left_contra_axon_outputs.index]) & (br_pairs.rightid[i] in [x[0] for x in right_contra_axon_outputs.index])):
        ratio_l = left_contra_axon_outputs.loc[br_pairs.leftid[i], 'ratio'].values[0]
        ratio_r = right_contra_axon_outputs.loc[br_pairs.rightid[i], 'ratio'].values[0]
        contra_pairs.append([br_pairs.leftid[i], br_pairs.rightid[i], ratio_l, ratio_r, (ratio_l+ratio_r)/2])

contra_pairs = pd.DataFrame(contra_pairs, columns = ['leftid', 'rightid', 'ratio_left', 'ratio_right', 'ratio_aver'])
contra_pairs.set_index('leftid', inplace=True)
# add ratio data to cosine similarity
cos_list['ratio'] = abs(contra_pairs.loc[cos_list.pairid, 'ratio_aver'].values-0.5)
# %%
# plot cosine similarity

fig, ax = plt.subplots(1,1, figsize=(2,2))
sns.distplot(cos_list.similarity_ipsi_contra, kde=False, bins=40, rug=True, rug_kws={"linewidth": 0.25, "alpha": 0.75, "height": 0.04}, ax=ax, color=sns.color_palette()[1])
fig.savefig('interhemisphere/plots/cosine-similarity_ipsi-vs-contra-partners.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1, figsize=(.75,1.75))
sns.swarmplot(cos_list.similarity_ipsi_contra, orient='v', size=2, ax=ax, color=sns.color_palette()[1])
fig.savefig('interhemisphere/plots/cosine-similarity_ipsi-vs-contra-partners_swarm.pdf', format='pdf', bbox_inches='tight')

sns.jointplot(x=cos_list.similarity_ipsi_contra, y=cos_list.ratio, kind='reg', scatter_kws={'s': 1}, size=5, ylim=(0.5,0), xlim=(0,1), color=sns.color_palette()[1])
plt.savefig('interhemisphere/plots/cosine-similarity-ipsi-contra-partners_vs_bilateral-character.pdf', format='pdf', bbox_inches='tight')

# %%
# identify ds neurons from each bilateral neuron: ipsi and contra separately

bilateral_partners = []
for pair in merged_list:
    ipsi_partners = list(pair.loc[:, pair.iloc[0, :]>=0.01].columns)
    contra_partners = list(pair.loc[:, pair.iloc[1, :]>=0.01].columns)

    bilateral_partners.append([pair.index[0], ipsi_partners, contra_partners])

bilateral_partners = pd.DataFrame(bilateral_partners, columns = ['pairid', 'ipsi_ds_partners', 'contra_ds_partners'])

iou_list=[]
for i in range(len(bilateral_partners)):
    if((len(bilateral_partners.loc[i].ipsi_ds_partners)>0) & (len(bilateral_partners.loc[i].contra_ds_partners)>0)):
        iou = len(np.intersect1d(bilateral_partners.loc[i].ipsi_ds_partners, bilateral_partners.loc[i].contra_ds_partners))/((len(bilateral_partners.loc[i].contra_ds_partners)+len(bilateral_partners.loc[i].ipsi_ds_partners))/2)
        
    if((len(bilateral_partners.loc[i].ipsi_ds_partners)==0) & (len(bilateral_partners.loc[i].contra_ds_partners)==0)):
        iou = -1
    iou_list.append(iou)

sns.distplot(iou_list, bins=40, kde=False)

cos_list['iou'] = iou_list
cos_list['ipsi_ds_partners'] = bilateral_partners.ipsi_ds_partners
cos_list['contra_ds_partners'] = bilateral_partners.contra_ds_partners
cos_list.set_index(cos_list.pairid, inplace=True, drop=True)

asymmetric_ipsi_contra = []
for i in cos_list.index:

    if((len(cos_list.loc[i].ipsi_ds_partners)==0) | (len(cos_list.loc[i].contra_ds_partners)==0)):
        continue

    intersect = np.intersect1d(cos_list.loc[i].ipsi_ds_partners, cos_list.loc[i].contra_ds_partners)
    if((len(intersect)/len(cos_list.loc[i].ipsi_ds_partners)==1) | (len(intersect)/len(cos_list.loc[i].contra_ds_partners)==1)):
        continue

    bool_list = cos_list[(cos_list.iou<=0.25)] # (cos_list.similarity_ipsi_contra<=0.5) & (cos_list.ratio<=0.25)
    if(i in bool_list.index):
        asymmetric_ipsi_contra.append(bool_list.loc[i])

asymmetric_ipsi_contra = pd.DataFrame(asymmetric_ipsi_contra)

# %%
# plot ipsi_contra neuron networks
from connectome_tools.cascade_analysis import Celltype, Celltype_Analyzer

celltypes, celltype_names = pm.Promat.celltypes()
all_celltypes = list(map(lambda pair: Celltype(pair[0], pair[1]), zip(celltype_names, celltypes)))
ct_analyzer = Celltype_Analyzer(all_celltypes)

# general bilaterals
ipsi_partners_all = [x for sublist in list(cos_list.ipsi_ds_partners) for x in sublist]
contra_partners_all = [x for sublist in list(cos_list.contra_ds_partners) for x in sublist]
bilateral_analyzer = Celltype_Analyzer([Celltype('ipsi_partners', ipsi_partners_all), Celltype('contra_partners', contra_partners_all)])
bilateral_analyzer.set_known_types(all_celltypes)
bilateral_memberships = bilateral_analyzer.memberships()

# asymmetric bilaterals
ipsi_partners = [x for sublist in list(asymmetric_ipsi_contra.ipsi_ds_partners) for x in sublist]
contra_partners = [x for sublist in list(asymmetric_ipsi_contra.contra_ds_partners) for x in sublist]
asym_bilateral_analyzer = Celltype_Analyzer([Celltype('ipsi_partners', ipsi_partners), Celltype('contra_partners', contra_partners)])
asym_bilateral_analyzer.set_known_types(all_celltypes)
asym_bilateral_memberships = asym_bilateral_analyzer.memberships()

# individual asymmetric bilateral pathways
asym_bilateral_paths = []
for pairid in asymmetric_ipsi_contra.index:
    ipsi_partners = [x for x in list(asymmetric_ipsi_contra.loc[pairid].ipsi_ds_partners)]
    contra_partners = [x for x in list(asymmetric_ipsi_contra.loc[pairid].contra_ds_partners)]
    ipsi_partners = Celltype(f'{pairid}_ipsi_partners', ipsi_partners)
    contra_partners = Celltype(f'{pairid}_contra_partners', contra_partners)
    asym_bilateral_paths.append(ipsi_partners)
    asym_bilateral_paths.append(contra_partners)

paths_asym_bilat_analyzer = Celltype_Analyzer(asym_bilateral_paths)
paths_asym_bilat_analyzer.set_known_types(all_celltypes)
path_memberships = paths_asym_bilat_analyzer.memberships()
path_memberships = path_memberships[path_memberships.sum(axis=1)!=0]

fraction_types_names = ['LHN', 'MB-FBN', 'CN', 'pre-dSEZ', 'pre-dVNC', 'RGN', 'dSEZ', 'dVNC', 'unknown']
colors = ['#D4E29E', '#C144BC', '#8C7700', '#77CDFC','#E0B1AD', 'tab:purple','#D88052', '#A52A2A', 'tab:grey']
ind = [x for x in range(0, len(path_memberships.columns))]

fig, ax = plt.subplots(1,1, figsize=(3,2))

# summary plot of 1st order upstream of dVNCs
plts = []
plt1 = plt.bar(ind, path_memberships.iloc[0,:], color = colors[0])
bottom = path_memberships.iloc[0,:]
plts.append(plt1)

for i in range(1, len(path_memberships.index)):
    plt1 = plt.bar(ind, path_memberships.iloc[i,:], bottom = bottom, color = colors[i])
    bottom = bottom + path_memberships.iloc[i,:]
    plts.append(plt1)

plt.savefig('interhemisphere/plots/individual_asym_bilateral_partners.pdf', format='pdf', bbox_inches='tight')
#plt.legend(plts, fraction_types_names)

# %%
# plot overview network all bilateral neurons
# divided into 4 categories

bilateral_cat = cos_list[cos_list.iou!=-1]
len(bilateral_cat[bilateral_cat.iou<=0.1])
len(bilateral_cat[(bilateral_cat.iou>0.1) & (bilateral_cat.iou<=0.25)])
len(bilateral_cat[(bilateral_cat.iou>0.25) & (bilateral_cat.iou<=0.5)])
len(bilateral_cat[(bilateral_cat.iou>0.5) & (bilateral_cat.iou<=0.75)])
len(bilateral_cat[(bilateral_cat.iou>0.75) & (bilateral_cat.iou<=1)])

cat1 = bilateral_cat[bilateral_cat.iou<=0.1]
cat2 = bilateral_cat[(bilateral_cat.iou>0.1) & (bilateral_cat.iou<=0.25)]
cat3 = bilateral_cat[(bilateral_cat.iou>0.25) & (bilateral_cat.iou<=0.5)]
cat4 = bilateral_cat[(bilateral_cat.iou>0.5) & (bilateral_cat.iou<=0.75)]
cat5 = bilateral_cat[(bilateral_cat.iou>0.75) & (bilateral_cat.iou<=1)]

cats = [cat1, cat2, cat3, cat4, cat5]

all_cats = []
for i in range(len(cats)):
    all_cats.append(Celltype(f'cat{i}_ipsi_partners', list(np.unique([x for sublist in cats[i].ipsi_ds_partners for x in sublist]))))
    all_cats.append(Celltype(f'cat{i}_contra_partners', list(np.unique([x for sublist in cats[i].contra_ds_partners for x in sublist]))))

all_cats_analyzer = Celltype_Analyzer(all_cats)
all_cats_analyzer.set_known_types(all_celltypes)
bilateral_cats_memberships = all_cats_analyzer.memberships()
bilateral_cats_memberships = bilateral_cats_memberships[bilateral_cats_memberships.sum(axis=1)!=0]

fraction_types_names = ['PN', 'LHN', 'MBIN', 'MBON','MB-FBN', 'CN', 'pre-dSEZ', 'pre-dVNC', 'RGN', 'dSEZ', 'dVNC', 'unknown']
colors = ['#1D79B7', '#D4E29E', '#FF8734', '#F9EB4D', '#C144BC', '#8C7700', '#77CDFC','#E0B1AD', 'tab:purple','#D88052', '#A52A2A', 'tab:grey']
ind = [x for x in range(0, len(bilateral_cats_memberships.columns))]

fig, ax = plt.subplots(1,1, figsize=(3,2))

# summary plot of 1st order upstream of dVNCs
plts = []
plt1 = plt.bar(ind, bilateral_cats_memberships.iloc[0,:], color = colors[0])
bottom = bilateral_cats_memberships.iloc[0,:]
plts.append(plt1)

for i in range(1, len(bilateral_cats_memberships.index)):
    plt1 = plt.bar(ind, bilateral_cats_memberships.iloc[i,:], bottom = bottom, color = colors[i])
    bottom = bottom + bilateral_cats_memberships.iloc[i,:]
    plts.append(plt1)

#plt.legend(plts, fraction_types_names)
plt.savefig('interhemisphere/plots/summary_bilateral_partners_5cats.pdf', format='pdf', bbox_inches='tight')


cat1 = bilateral_cat[bilateral_cat.iou<=0.1]
cat2 = bilateral_cat[(bilateral_cat.iou>0.1) & (bilateral_cat.iou<=0.25)]
cat3 = bilateral_cat[(bilateral_cat.iou>0.25) & (bilateral_cat.iou<=1)]

cats = [cat1, cat2, cat3]

all_cats = []
for i in range(len(cats)):
    all_cats.append(Celltype(f'cat{i}_ipsi_partners', list(np.unique([x for sublist in cats[i].ipsi_ds_partners for x in sublist]))))
    all_cats.append(Celltype(f'cat{i}_contra_partners', list(np.unique([x for sublist in cats[i].contra_ds_partners for x in sublist]))))

all_cats_analyzer = Celltype_Analyzer(all_cats)
all_cats_analyzer.set_known_types(all_celltypes)
bilateral_cats_memberships = all_cats_analyzer.memberships()
bilateral_cats_memberships = bilateral_cats_memberships[bilateral_cats_memberships.sum(axis=1)!=0]

fraction_types_names = ['PN', 'LHN', 'MBIN', 'MBON','MB-FBN', 'CN', 'pre-dSEZ', 'pre-dVNC', 'RGN', 'dSEZ', 'dVNC', 'unknown']
colors = ['#1D79B7', '#D4E29E', '#FF9852', '#F9EB4D', '#C144BC', '#8C7700', '#77CDFC','#E0B1AD', 'tab:purple','#C47451', '#A52A2A', 'tab:grey']
ind = [x for x in range(0, len(bilateral_cats_memberships.columns))]

fig, ax = plt.subplots(1,1, figsize=(1.5,2))

# summary plot of 1st order upstream of dVNCs
plts = []
plt1 = plt.bar(ind, bilateral_cats_memberships.iloc[0,:], color = colors[0])
bottom = bilateral_cats_memberships.iloc[0,:]
plts.append(plt1)

for i in range(1, len(bilateral_cats_memberships.index)):
    plt1 = plt.bar(ind, bilateral_cats_memberships.iloc[i,:], bottom = bottom, color = colors[i])
    bottom = bottom + bilateral_cats_memberships.iloc[i,:]
    plts.append(plt1)

plt.savefig('interhemisphere/plots/summary_bilateral_partners_3cats.pdf', format='pdf', bbox_inches='tight')

# %%
# plot simple 1-hop downstream of ipsi and contralateral partners
# not very interesting

ipsi = pymaid.get_skids_by_annotation('mw ipsilateral axon')
bilateral = pymaid.get_skids_by_annotation('mw bilateral axon')
contra = pymaid.get_skids_by_annotation('mw contralateral axon')

ipsi_pairs = pm.Promat.extract_pairs_from_list(ipsi, pairs)[0]
bi_pairs = pm.Promat.extract_pairs_from_list(bilateral, pairs)[0]
contra_pairs = pm.Promat.extract_pairs_from_list(contra, pairs)[0]

adj_mat = pm.Adjacency_matrix(adj.values, adj.index, pairs, inputs, 'axo-dendritic')
hops=1
threshold = 0.01

# upstream ipsi, bi, contra

us_ipsi_list = []
for i in tqdm(range(len(ipsi_pairs))):
    us = adj_mat.upstream_multihop(list(ipsi_pairs.loc[i]), threshold, min_members = 0, hops=hops, allow_source_us=True)
    us_ipsi_list.append(us)

us_bi_list = []
for i in tqdm(range(len(bi_pairs))):
    us = adj_mat.upstream_multihop(list(bi_pairs.loc[i]), threshold, min_members = 0, hops=hops, allow_source_us=True)
    us_bi_list.append(us)

us_contra_list = []
for i in tqdm(range(len(contra_pairs))):
    us = adj_mat.upstream_multihop(list(contra_pairs.loc[i]), threshold, min_members = 0, hops=hops, allow_source_us=True)
    us_contra_list.append(us)

us_ipsi = list(np.unique([x for sublist in us_ipsi_list for x in sublist[0]]))
us_bi = list(np.unique([x for sublist in us_bi_list for x in sublist[0]]))
us_contra = list(np.unique([x for sublist in us_contra_list for x in sublist[0]]))

us_ipsi_ipsi = len(np.intersect1d(us_ipsi, ipsi))/len(us_ipsi)
us_ipsi_bilateral = len(np.intersect1d(us_ipsi, bilateral))/len(us_ipsi)
us_ipsi_contra = len(np.intersect1d(us_ipsi, contra))/len(us_ipsi)

us_bi_ipsi = len(np.intersect1d(us_bi, ipsi))/len(us_bi)
us_bi_bilateral = len(np.intersect1d(us_bi, bilateral))/len(us_bi)
us_bi_contra = len(np.intersect1d(us_bi, contra))/len(us_bi)

us_contra_ipsi = len(np.intersect1d(us_contra, ipsi))/len(us_contra)
us_contra_bilateral = len(np.intersect1d(us_contra, bilateral))/len(us_contra)
us_contra_contra = len(np.intersect1d(us_contra, contra))/len(us_contra)

# bilateral downstream
iPartners = list(np.unique(ipsi_partners_all))
iPartners_ipsi = len(np.intersect1d(iPartners, ipsi))/len(iPartners)
iPartners_bilateral = len(np.intersect1d(iPartners, bilateral))/len(iPartners)
iPartners_contra = len(np.intersect1d(iPartners, contra))/len(iPartners)

cPartners = list(np.unique(contra_partners_all))
cPartners_ipsi = len(np.intersect1d(cPartners, ipsi))/len(iPartners)
cPartners_bilateral = len(np.intersect1d(cPartners, bilateral))/len(iPartners)
cPartners_contra = len(np.intersect1d(cPartners, contra))/len(iPartners)

# downstream ipsi, contra
ds_ipsi_list = []
for i in tqdm(range(len(ipsi_pairs))):
    ds_ipsi = adj_mat.downstream_multihop(source = list(ipsi_pairs.loc[i]), threshold = threshold, min_members = 0, hops=hops, allow_source_ds=True)
    ds_ipsi_list.append(ds_ipsi)
    
ds_contra_list = []
for i in tqdm(range(len(contra_pairs))):
    ds_contra = adj_mat.downstream_multihop(list(contra_pairs.loc[i]), threshold, min_members = 0, hops=hops, allow_source_ds=True)
    ds_contra_list.append(ds_contra)

ds_ipsi = list(np.unique([x for sublist in ds_ipsi_list for x in sublist[0]]))
ds_contra = list(np.unique([x for sublist in ds_contra_list for x in sublist[0]]))

ds_ipsi_ipsi = len(np.intersect1d(ds_ipsi, ipsi))/len(ds_ipsi)
ds_ipsi_bilateral = len(np.intersect1d(ds_ipsi, bilateral))/len(ds_ipsi)
ds_ipsi_contra = len(np.intersect1d(ds_ipsi, contra))/len(ds_ipsi)

ds_contra_ipsi = len(np.intersect1d(ds_contra, ipsi))/len(ds_contra)
ds_contra_bilateral = len(np.intersect1d(ds_contra, bilateral))/len(ds_contra)
ds_contra_contra = len(np.intersect1d(ds_contra, contra))/len(ds_contra)

# barplot downstream
ind = np.arange(0, 4)
us_ipsi = np.array([ds_ipsi_ipsi, iPartners_ipsi, cPartners_ipsi, ds_contra_ipsi])
us_bi = np.array([ds_ipsi_bilateral, iPartners_bilateral, cPartners_bilateral, ds_contra_bilateral])
us_contra = np.array([ds_ipsi_contra, iPartners_contra, cPartners_contra, ds_ipsi_contra])

fig, ax = plt.subplots(1,1, figsize=(2,2))
plt.bar(x=ind, height=us_ipsi)
plt.bar(x=ind, height=us_bi, bottom=us_ipsi)
plt.bar(x=ind, height=us_contra, bottom=us_ipsi+us_bi)

# %%
# plot simple 1-hop upstream/downstream of ipsi, bi, contra with known cell types

# %%
# types of self-loops

threshold = 0.01

CNs = pymaid.get_skids_by_annotation('mw CN')
CN_pairs = pm.Promat.extract_pairs_from_list(CNs, pairs)[0]
pairs = CN_pairs.iloc[0:5, :]

all_edges_list=[]
for pair_id in pairs.leftid:
    _, ds, ds_edges = adj_mat.downstream(pair_id, threshold)
    ds_edges, CN33_ds_partners = adj_mat.edge_threshold(ds_edges, threshold, 'downstream')
    initial_overthres_ds_edges = ds_edges[ds_edges.overthres==True]
    initial_overthres_ds_edges.reset_index(inplace=True)
    ds_partners = initial_overthres_ds_edges.downstream_pair_id

    all_edges = []
    for i, partner in enumerate(ds_partners):
        _, ds, edges = adj_mat.downstream(partner, threshold)
        ds_edges, ds_partners = adj_mat.edge_threshold(edges, threshold, 'downstream')
        overthres_ds_edges = ds_edges[ds_edges.overthres==True]
        overthres_ds_edges.reset_index(inplace=True)

        for j in range(len(overthres_ds_edges)):
            path = pd.concat([initial_overthres_ds_edges.iloc[i, :], overthres_ds_edges.iloc[j, :]], axis=1).T
            path.reset_index(inplace=True)
            path = path.drop(labels=['index', 'level_0', 'overthres'], axis=1)
            path['path'] = f'path-{pair_id}_{i}_{j}'
            all_edges.append(path)

    all_edges = pd.concat(all_edges, axis=0)
    all_edges.set_index(['path'], inplace=True)
    all_edges_list.append(all_edges)



