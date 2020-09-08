#%%
import os
import sys
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:
    pass

# %%
import sys
sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/maggot_models/')
sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
from pymaid_creds import url, name, password, token
import pymaid

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from graspy.plot import gridplot, heatmap
from graspy.utils import binarize, pass_to_ranks
from src.data import load_metagraph
from src.visualization import CLASS_COLOR_DICT, adjplot

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
#plt.rcParams.update({'font.size': 6})

rm = pymaid.CatmaidInstance(url, name, password, token)

mg = load_metagraph("Gad", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg.calculate_degrees(inplace=True)

adj = mg.adj  # adjacency matrix from the "mg" object

def skid_to_index(skid, mg):
    index_match = np.where(mg.meta.index == skid)[0]
    if(len(index_match)==1):
        return(index_match[0])
    if(len(index_match)!=1):
        print('Not one match for skid %i!' %skid)


# %%
# which cell types are in a set of skids?

annot_list_types = ['sensory', 'PN', 'LHN', 'MBIN', 'KC', 'MBON', 'FBN', 'CN', 'dVNC', 'dSEZ', 'RGN']
annot_list = [list(pymaid.get_annotated('mw brain inputs').name), 
            list(pymaid.get_annotated('mw brain inputs 2nd_order PN').name),
            ['mw LHN'], ['mw MBIN'], ['mw KC'], ['mw MBON'],
            ['mw FBN', 'mw FB2N', 'mw FAN'],
            ['mw CN']
            ]

inputs_skids = pymaid.get_annotated('mw brain inputs', include_sub_annotations = True)
inputs_skids = inputs_skids[inputs_skids.type == 'neuron'].skeleton_ids
inputs_skids = [val for sublist in list(inputs_skids) for val in sublist]

PN_skids = pymaid.get_annotated('mw brain inputs 2nd_order PN', include_sub_annotations = True)
PN_skids = PN_skids[PN_skids.type == 'neuron'].skeleton_ids
PN_skids = [val for sublist in list(PN_skids) for val in sublist]

LHN_skids = pymaid.get_skids_by_annotation('mw LHN')
MBIN_skids = pymaid.get_skids_by_annotation('mw MBIN')
KC_skids = pymaid.get_skids_by_annotation('mw KC')
MBON_skids = pymaid.get_skids_by_annotation('mw MBON')
FBN_skids = pymaid.get_skids_by_annotation(['mw FBN', 'mw FB2N', 'mw FAN'], allow_partial=True)
CN_skids = pymaid.get_skids_by_annotation('mw CN')
dVNC_skids = pymaid.get_skids_by_annotation('mw dVNC')
dSEZ_skids = pymaid.get_skids_by_annotation('mw dSEZ')
RGN_skids = pymaid.get_skids_by_annotation('mw RG')

skid_list = [inputs_skids, PN_skids, LHN_skids, MBIN_skids, KC_skids, MBON_skids, 
            FBN_skids, CN_skids, dVNC_skids, dSEZ_skids, RGN_skids]

def member_types(data, skid_list, celltype_names, col_name):

    fraction_type = []
    for skids in skid_list:
        fraction = len(np.intersect1d(data, skids))/len(data)
        fraction_type.append(fraction)

    fraction_type = pd.DataFrame(fraction_type, index = celltype_names, columns = [col_name])
    return(fraction_type)

pre_dVNC_type = member_types(pymaid.get_skids_by_annotation('mw pre-dVNC'), skid_list, annot_list_types, 'pre-dVNC')
pre_dSEZ_type = member_types(pymaid.get_skids_by_annotation('mw pre-dSEZ'), skid_list, annot_list_types, 'pre-dSEZ')
pre_RGN_type = member_types(pymaid.get_skids_by_annotation('mw pre-RG'), skid_list, annot_list_types, 'pre-RGN')
dVNC_type = member_types(pymaid.get_skids_by_annotation('mw dVNC'), skid_list, annot_list_types, 'dVNC')
dSEZ_type = member_types(pymaid.get_skids_by_annotation('mw dSEZ'), skid_list, annot_list_types, 'dSEZ')
RGN_type = member_types(pymaid.get_skids_by_annotation('mw RG'), skid_list, annot_list_types, 'RGN')
dVNC2_type = member_types(pymaid.get_skids_by_annotation('mw dVNC 2nd_order'), skid_list, annot_list_types, 'dVNC2')
dSEZ2_type = member_types(pymaid.get_skids_by_annotation('mw dSEZ 2nd_order'), skid_list, annot_list_types, 'dSEZ2')

dVNC_FB_type = member_types(pymaid.get_skids_by_annotation('mw dVNC feedback 3hop 7-Sept 2020'), skid_list, annot_list_types, 'c:dVNC')
dSEZ_FB_type = member_types(pymaid.get_skids_by_annotation('mw dSEZ feedback 3hop 7-Sept 2020'), skid_list, annot_list_types, 'c:dSEZ')
predVNC_FB_type = member_types(pymaid.get_skids_by_annotation('mw pre-dVNC feedback 3hop 7-Sept 2020'), skid_list, annot_list_types, 'c:pre-dVNC')
predSEZ_FB_type = member_types(pymaid.get_skids_by_annotation('mw pre-dSEZ feedback 3hop 7-Sept 2020'), skid_list, annot_list_types, 'c:pre-dSEZ')
preRGN_FB_type = member_types(pymaid.get_skids_by_annotation('mw pre-RG feedback 3hop 7-Sept 2020'), skid_list, annot_list_types, 'c:pre-RGN')

dVNC_types = pd.concat([pre_dVNC_type, dVNC_type, dVNC2_type, dVNC_FB_type, predVNC_FB_type], axis = 1)
dSEZ_types = pd.concat([pre_dSEZ_type, dSEZ_type, dSEZ2_type, dSEZ_FB_type, predSEZ_FB_type], axis = 1)
RG_types = pd.concat([pre_RGN_type, RGN_type, preRGN_FB_type], axis = 1)

import cmasher as cmr

width = 1.25
height = 1.25
vmax = 0.3
cmap = cmr.lavender
cbar = False
fontsize = 5

fig, axs = plt.subplots(
    1, 1, figsize = (width, height)
)
sns.heatmap(dVNC_types.drop(index = ['sensory', 'KC']), annot=True, fmt=".0%", linewidth = 0.25, vmax = vmax, ax = axs, cmap = cmap, cbar = cbar, annot_kws={"size": fontsize})
plt.savefig('cascades/feedback_through_brain/plots/celltypes_dVNC_pathway.pdf', bbox_inches='tight')

fig, axs = plt.subplots(
    1, 1, figsize = (width, height)
)
sns.heatmap(dSEZ_types.drop(index = ['sensory', 'KC']), annot=True, fmt=".0%", linewidth = 0.25, vmax = vmax, cmap = cmap, cbar = cbar, annot_kws={"size": fontsize})
plt.savefig('cascades/feedback_through_brain/plots/celltypes_dSEZ_pathway.pdf', bbox_inches='tight')

fig, axs = plt.subplots(
    1, 1, figsize = (width*3/5, height)
)
sns.heatmap(RG_types.drop(index = ['sensory', 'KC']), annot=True, fmt=".0%", linewidth = 0.25, vmax = vmax, cmap = cmap, cbar = cbar, annot_kws={"size": fontsize})
plt.savefig('cascades/feedback_through_brain/plots/celltypes_RG_pathway.pdf', bbox_inches='tight')

'''
ind = range(len(dVNC_types.columns))
colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'blue', 'brown', 'orange', 'purple', 'black', 'white', 'magenta']
plt.bar(ind, [1,1,1],color = 'gray')
for i in range(len(dVNC_types.index)):
    plt.bar(ind, dVNC_types.iloc[i, :], bottom = dVNC_types.iloc[0:i, :].sum(axis = 0),color = colors[i])
plt.savefig('cascades/feedback_through_brain/plots/celltypes_dVNC_pathway.pdf', bbox_inches='tight')
        
ind = range(len(dSEZ_types.columns))
colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'blue', 'brown', 'orange', 'purple']
plt.bar(ind, [1,1,1],color = 'gray')
for i in range(len(dSEZ_types.index)):
    plt.bar(ind, dSEZ_types.iloc[i, :], bottom = dSEZ_types.iloc[0:i, :].sum(axis = 0),color = colors[i])
plt.savefig('cascades/feedback_through_brain/plots/celltypes_dSEZ_pathway.pdf', bbox_inches='tight')

ind = range(len(RG_types.columns))
colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'blue', 'brown', 'orange', 'purple']
plt.bar(ind, [1,1,1],color = 'gray')
for i in range(len(RG_types.index)):
    plt.bar(ind, RG_types.iloc[i, :], bottom = RG_types.iloc[0:i, :].sum(axis = 0),color = colors[i])
plt.savefig('cascades/feedback_through_brain/plots/celltypes_RG_pathway.pdf', bbox_inches='tight')
'''

# %%
