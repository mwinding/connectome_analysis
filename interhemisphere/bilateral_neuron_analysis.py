#%%
import os
import sys
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/maggot_models/')
    sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
except:
    pass


from pymaid_creds import url, name, password, token
import pymaid

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

rm = pymaid.CatmaidInstance(url, name, password, token)

#adj = mg.adj  # adjacency matrix from the "mg" object
adj = pd.read_csv('VNC_interaction/data/brA1_axon-dendrite.csv', header = 0, index_col = 0)
adj.columns = adj.columns.astype(int) #convert column names to int for easier indexing

# remove A1 except for ascendings
A1_ascending = pymaid.get_skids_by_annotation('mw A1 neurons paired ascending')
A1 = pymaid.get_skids_by_annotation('mw A1 neurons paired')
A1_local = list(np.setdiff1d(A1, A1_ascending)) # all A1 without A1_ascending
pruned_index = list(np.setdiff1d(adj.index, A1_local)) 
adj = adj.loc[pruned_index, pruned_index] # remove all local A1 skids from adjacency matrix

# load inputs and pair data
inputs = pd.read_csv('VNC_interaction/data/brA1_input_counts.csv', index_col = 0)
inputs = pd.DataFrame(inputs.values, index = inputs.index, columns = ['axon_input', 'dendrite_input'])
pairs = pd.read_csv('VNC_interaction/data/pairs-2020-10-26.csv', header = 0) # import pairs

# load projectome 
projectome = pd.read_csv('interhemisphere/data/projectome_mw_brain_matrix_A1_split.csv', index_col = 0, header = 0)

bilateral = pymaid.get_skids_by_annotation('mw bilateral axon')
# %%
# 
from connectome_tools.process_matrix import Promat

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
dVNC = pymaid.get_skids_by_annotation('mw dVNC')
dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')

bilateral = list(np.intersect1d(bilateral, br))

bilateral_no_brain_input = projectome.groupby('skeleton')['Brain Hemisphere left', 'Brain Hemisphere right'].sum()
exclude = list(bilateral_no_brain_input[(bilateral_no_brain_input.loc[:, 'Brain Hemisphere left']==0) & (bilateral_no_brain_input.loc[:, 'Brain Hemisphere right']==0)].index)
bilateral = list(np.setdiff1d(bilateral, exclude + dVNC + dSEZ))

bilateral_pairs = Promat.extract_pairs_from_list(bilateral, pairs)[0]

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
def ipsi_contra_ds_partners(bilateral_leftid, bilateral_rightid, skeletons, inputs, threshold=-1):
    # look for ipsi and contra downstream partners over threshold
    bi_ad_connect = bilateral_ad_connections(bilateral_leftid, bilateral_rightid, skeletons, inputs)
    bi_ad_connect_index1 = bi_ad_connect.set_index(['connection_type', 'source'])
    if(len(bi_ad_connect_index1)==0):
        return([])

    # contralateral partners
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

    contra_pair_partners = Promat.extract_pairs_from_list(contra_left_partners + contra_right_partners, pairs)[0]

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

    ipsi_pair_partners = Promat.extract_pairs_from_list(ipsi_left_partners + ipsi_right_partners, pairs)[0]

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
#

from tqdm import tqdm

issues_list = []
data_list = []
for i in tqdm(range(1, len(bilateral_pairs))):
    try:
        data = ipsi_contra_ds_partners(bilateral_pairs.leftid[i], bilateral_pairs.rightid[i], skeletons, inputs)
        data_list.append(data)
    except:
        print(f'problem with {i}')
        issues_list.append(i)

filtered_data = [x for x in data_list if type(x)==pd.DataFrame]
data = pd.concat((x for x in filtered_data), axis=0)
data.reset_index(inplace=True, drop=True)
pair_ids = list(np.unique(data.source_pairid))

data = data.set_index(['source_pairid', 'connection_type'])

merged_list = []
for i in range(len(pair_ids)):
    if(('contralateral' in data.loc[pair_ids[i]].index) & ('ipsilateral' in data.loc[pair_ids[i]].index)): # super hacky, need to check the data
        ipsi_data = pd.DataFrame([data.loc[(pair_ids[i], 'ipsilateral')].average_input.values], index = [f'{pair_ids[i]}-ipsi'], columns=list(data.loc[(pair_ids[i], 'ipsilateral')].leftid.values))
        contra_data = pd.DataFrame([data.loc[(pair_ids[i], 'contralateral')].average_input.values], index = [f'{pair_ids[i]}-contra'], columns=list(data.loc[(pair_ids[i], 'contralateral')].leftid.values))

        merged_data = pd.concat([ipsi_data, contra_data], axis=0)
        merged_data.fillna(0, inplace=True)
        merged_list.append(merged_data)
# %%
