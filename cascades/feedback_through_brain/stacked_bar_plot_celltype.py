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
plt.rcParams.update({'font.size': 6})

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

annot_list_types = ['sensory', 'PN', 'LHN', 'MBIN', 'KC', 'MBON', 'FBN', 'CN', 'SEZ-motor', 'dVNC', 'dSEZ', 'RGN', 'others']

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
FBN_skids = pymaid.get_skids_by_annotation(['mw FBN', 'mw FB2N', 'mw FAN'])
CN_skids = pymaid.get_skids_by_annotation('mw CN')
motor_skids = pymaid.get_skids_by_annotation('mw motor')
dVNC_skids = pymaid.get_skids_by_annotation('mw dVNC')
dSEZ_skids = pymaid.get_skids_by_annotation('mw dSEZ')
RGN_skids = pymaid.get_skids_by_annotation('mw RG')

skid_list = [inputs_skids, PN_skids, LHN_skids, MBIN_skids, KC_skids, MBON_skids, 
            FBN_skids, CN_skids, motor_skids, dVNC_skids, dSEZ_skids, RGN_skids]

others = np.setdiff1d(mg.meta.index, np.unique([val for sublist in skid_list for val in sublist])) # list of all unannotated neurons
skid_list.append(others)

def member_types(data, skid_list, celltype_names, col_name):

    fraction_type = []
    for skids in skid_list:
        fraction = len(np.intersect1d(data, skids))/len(data)
        fraction_type.append(fraction)

    fraction_type = pd.DataFrame(fraction_type, index = celltype_names, columns = ['%s (%i)' %(col_name, len(data))])
    return(fraction_type)

pre_dVNC_type = member_types(pymaid.get_skids_by_annotation('mw pre-dVNC'), skid_list, annot_list_types, 'pre-dVNC')
pre_dSEZ_type = member_types(pymaid.get_skids_by_annotation('mw pre-dSEZ'), skid_list, annot_list_types, 'pre-dSEZ')
pre_RGN_type = member_types(pymaid.get_skids_by_annotation('mw pre-RG'), skid_list, annot_list_types, 'pre-RGN')
dVNC_type = member_types(pymaid.get_skids_by_annotation('mw dVNC'), skid_list, annot_list_types, 'dVNC')
dSEZ_type = member_types(pymaid.get_skids_by_annotation('mw dSEZ'), skid_list, annot_list_types, 'dSEZ')
RGN_type = member_types(pymaid.get_skids_by_annotation('mw RG'), skid_list, annot_list_types, 'RGN')
dVNC2_type = member_types(pymaid.get_skids_by_annotation('mw dVNC 2nd_order'), skid_list, annot_list_types, 'dVNC2')
dSEZ2_type = member_types(pymaid.get_skids_by_annotation('mw dSEZ 2nd_order'), skid_list, annot_list_types, 'dSEZ2')
ds_pre_dVNC_type = member_types(pymaid.get_skids_by_annotation('mw ds-pre-dVNC'), skid_list, annot_list_types, 'ds pre-dVNC')
ds_pre_dSEZ_type = member_types(pymaid.get_skids_by_annotation('mw ds-pre-dSEZ'), skid_list, annot_list_types, 'ds pre-dSEZ')
ds_pre_RGN_type = member_types(pymaid.get_skids_by_annotation('mw ds-pre-RGN'), skid_list, annot_list_types, 'ds pre-RGN')

dVNC_FB_type = member_types(pymaid.get_skids_by_annotation('mw dVNC feedback 3hop 16-Sept 2020'), skid_list, annot_list_types, 'c:dVNC')
dSEZ_FB_type = member_types(pymaid.get_skids_by_annotation('mw dSEZ feedback 3hop 16-Sept 2020'), skid_list, annot_list_types, 'c:dSEZ')
predVNC_FB_type = member_types(pymaid.get_skids_by_annotation('mw pre-dVNC feedback 3hop 16-Sept 2020'), skid_list, annot_list_types, 'c:pre-dVNC')
predSEZ_FB_type = member_types(pymaid.get_skids_by_annotation('mw pre-dSEZ feedback 3hop 16-Sept 2020'), skid_list, annot_list_types, 'c:pre-dSEZ')
preRGN_FB_type = member_types(pymaid.get_skids_by_annotation('mw pre-RG feedback 3hop 16-Sept 2020'), skid_list, annot_list_types, 'c:pre-RGN')

dVNC_types = pd.concat([pre_dVNC_type, ds_pre_dVNC_type, predVNC_FB_type, dVNC_type, dVNC2_type, dVNC_FB_type], axis = 1)
dSEZ_types = pd.concat([pre_dSEZ_type, ds_pre_dSEZ_type, predSEZ_FB_type, dSEZ_type, dSEZ2_type, dSEZ_FB_type], axis = 1)
RG_types = pd.concat([pre_RGN_type, ds_pre_RGN_type, preRGN_FB_type, RGN_type], axis = 1)

# remove row categories that are not useful

#remove = ['sensory', 'KC', 'SEZ-motor']
remove = ['sensory']
dVNC_types = dVNC_types.drop(index = remove)
dSEZ_types = dSEZ_types.drop(index = remove)
RG_types = RG_types.drop(index = remove)

import cmasher as cmr

width = 1.25
height = 1.5
vmax = 0.3
cmap = cmr.lavender
cbar = False
fontsize = 5

fig, axs = plt.subplots(
    1, 1, figsize = (width, height)
)
sns.heatmap(dVNC_types, annot=True, fmt=".0%", linewidth = 0.25, vmax = vmax, ax = axs, cmap = cmap, cbar = cbar, annot_kws={"size": fontsize})
plt.savefig('cascades/feedback_through_brain/plots/celltypes_dVNC_pathway.pdf', bbox_inches='tight')

fig, axs = plt.subplots(
    1, 1, figsize = (width, height)
)
sns.heatmap(dSEZ_types, annot=True, fmt=".0%", linewidth = 0.25, vmax = vmax, cmap = cmap, cbar = cbar, annot_kws={"size": fontsize})
plt.savefig('cascades/feedback_through_brain/plots/celltypes_dSEZ_pathway.pdf', bbox_inches='tight')

fig, axs = plt.subplots(
    1, 1, figsize = (width*3/5, height)
)
sns.heatmap(RG_types, annot=True, fmt=".0%", linewidth = 0.25, vmax = vmax, cmap = cmap, cbar = cbar, annot_kws={"size": fontsize})
plt.savefig('cascades/feedback_through_brain/plots/celltypes_RG_pathway.pdf', bbox_inches='tight')

# %%
# intersection between types

dVNC_FB_type = member_types(pymaid.get_skids_by_annotation('mw dVNC feedback 3hop 16-Sept 2020'), skid_list, annot_list_types, 'c:dVNC')
dSEZ_FB_type = member_types(pymaid.get_skids_by_annotation('mw dSEZ feedback 3hop 16-Sept 2020'), skid_list, annot_list_types, 'c:dSEZ')
predVNC_FB_type = member_types(pymaid.get_skids_by_annotation('mw pre-dVNC feedback 3hop 16-Sept 2020'), skid_list, annot_list_types, 'c:pre-dVNC')
predSEZ_FB_type = member_types(pymaid.get_skids_by_annotation('mw pre-dSEZ feedback 3hop 16-Sept 2020'), skid_list, annot_list_types, 'c:pre-dSEZ')
preRGN_FB_type = member_types(pymaid.get_skids_by_annotation('mw pre-RG feedback 3hop 16-Sept 2020'), skid_list, annot_list_types, 'c:pre-RGN')

pre_dVNC = pymaid.get_skids_by_annotation('mw pre-dVNC')
pre_dSEZ = pymaid.get_skids_by_annotation('mw pre-dSEZ')
pre_RGN = pymaid.get_skids_by_annotation('mw pre-RG')

ds_pre_dVNC = pymaid.get_skids_by_annotation('mw ds-pre-dVNC')
ds_pre_dSEZ = pymaid.get_skids_by_annotation('mw ds-pre-dSEZ')
ds_pre_RGN = pymaid.get_skids_by_annotation('mw ds-pre-RGN')

dVNC = pymaid.get_skids_by_annotation('mw dVNC')
dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')
RGN = pymaid.get_skids_by_annotation('mw RG')

ds_dVNC = pymaid.get_skids_by_annotation('mw dVNC 2nd_order')
ds_dSEZ = pymaid.get_skids_by_annotation('mw dSEZ 2nd_order')

dVNC_FB = pymaid.get_skids_by_annotation('mw dVNC feedback 3hop 16-Sept 2020')
dSEZ_FB = pymaid.get_skids_by_annotation('mw dSEZ feedback 3hop 16-Sept 2020')

pre_dVNC_FB = pymaid.get_skids_by_annotation('mw pre-dVNC feedback 3hop 16-Sept 2020')
pre_dSEZ_FB = pymaid.get_skids_by_annotation('mw pre-dSEZ feedback 3hop 16-Sept 2020')
pre_RGN_FB = pymaid.get_skids_by_annotation('mw pre-RG feedback 3hop 16-Sept 2020')

iou_data = [pre_dVNC, pre_dSEZ, pre_RGN, dVNC, dSEZ, RGN, ds_dVNC, 
            ds_dSEZ, ds_pre_dVNC, ds_pre_dSEZ, ds_pre_RGN, dVNC_FB, 
            pre_dVNC_FB, dSEZ_FB, pre_dSEZ_FB, pre_RGN_FB]

names = ['pre-dVNC', 'dVNC', 'ds-dVNC', 'dVNC-c1', 'dVNC-c2',
        'pre-dSEZ', 'dSEZ', 'ds-dSEZ', 'ds-pre-dVNC', 'ds-pre-dSEZ', 'ds-pre-RGN', 'dSEZ-c1', 'dSEZ-c2',
        'pre-RGN', 'RGN', 'RGN-c2']

def iou_matrix(data, names):
    iou_matrix = np.zeros((len(data), len(data)))

    for i in range(len(data)):
        for j in range(len(data)):
            if(len(np.union1d(data[i], data[j])) > 0):
                iou = len(np.intersect1d(data[i], data[j]))/len(np.union1d(data[i], data[j]))
                iou_matrix[i, j] = iou

    iou_matrix = pd.DataFrame(iou_matrix, index = names, columns = names)
    return(iou_matrix)

iou = iou_matrix(iou_data, names)

fig, axs = plt.subplots(
    1, 1, figsize=(2.5, 2.5)
)

ax = axs
fig.tight_layout(pad=2.0)
sns.heatmap(iou, ax = ax, square = True)
#sns.clustermap(iou)
ax.set_title('Membership Similarity')
plt.savefig('cascades/feedback_through_brain/plots/output_FBN_centers_primary.pdf', format='pdf', bbox_inches='tight')

# %%
