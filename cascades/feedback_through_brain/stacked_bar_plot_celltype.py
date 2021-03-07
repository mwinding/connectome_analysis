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

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 6})

rm = pymaid.CatmaidInstance(url, token, name, password)
mg = load_metagraph("Gad", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')

# %%
# which cell types are in a set of skids?
from connectome_tools.process_matrix import Promat
from connectome_tools.cascade_analysis import Celltype, Celltype_Analyzer

# set up list of known cell types
sensory = Celltype('sensory', pymaid.get_skids_by_annotation(pymaid.get_annotated('mw brain inputs').name))
PN = Celltype('PN', pymaid.get_skids_by_annotation(pymaid.get_annotated('mw brain inputs 2nd_order PN').name))
LHN = Celltype('LHN', pymaid.get_skids_by_annotation('mw LHN'))
MBIN = Celltype('MBIN', pymaid.get_skids_by_annotation('mw MBIN'))
KC = Celltype('KC', pymaid.get_skids_by_annotation('mw KC'))
MBON = Celltype('MBON', pymaid.get_skids_by_annotation('mw MBON'))
FBN = Celltype('FBN', pymaid.get_skids_by_annotation(['mw FBN', 'mw FB2N', 'mw FAN']))
CN = Celltype('CN', pymaid.get_skids_by_annotation('mw CN'))
motor = Celltype('SEZ-motor', pymaid.get_skids_by_annotation('mw motor'))
dVNC = Celltype('dVNC', pymaid.get_skids_by_annotation('mw dVNC'))
dSEZ = Celltype('dSEZ', pymaid.get_skids_by_annotation('mw dSEZ'))
RGN = Celltype('RGN', pymaid.get_skids_by_annotation('mw RGN'))

known_types = [sensory, PN, LHN, MBIN, KC, MBON, FBN, CN, RGN, dSEZ, motor, dVNC]

# set up list of dVNC cell types and determine membership in known cell types
pre_dVNC = Celltype('pre-dVNC', pymaid.get_skids_by_annotation('mw pre-dVNC'))
ds_pre_dVNC = Celltype('ds-pre-dVNC', pymaid.get_skids_by_annotation('mw ds-pre-dVNC'))
c_pre_dVNC = Celltype('c:pre-dVNC', pymaid.get_skids_by_annotation('mw pre-dVNC feedback 3hop 16-Sept 2020'))
dVNC = Celltype('dVNC', pymaid.get_skids_by_annotation('mw dVNC'))
ds_dVNC = Celltype('ds-dVNC', pymaid.get_skids_by_annotation('mw dVNC 2nd_order'))
c_dVNC = Celltype('c:dVNC', pymaid.get_skids_by_annotation('mw dVNC feedback 3hop 16-Sept 2020'))

dVNC_types = Celltype_Analyzer([pre_dVNC, ds_pre_dVNC, c_pre_dVNC, dVNC, ds_dVNC, c_dVNC], mg)
dVNC_types.set_known_types(known_types)
dVNC_memberships = dVNC_types.memberships()

# set up list of dSEZ cell types and determine membership in known cell types
pre_dSEZ = Celltype('pre-dSEZ', pymaid.get_skids_by_annotation('mw pre-dSEZ'))
ds_pre_dSEZ = Celltype('ds-pre-dSEZ', pymaid.get_skids_by_annotation('mw ds-pre-dSEZ'))
dSEZ = Celltype('dSEZ', pymaid.get_skids_by_annotation('mw dSEZ'))
ds_dSEZ = Celltype('ds-dSEZ', pymaid.get_skids_by_annotation('mw dSEZ 2nd_order'))
c_dSEZ = Celltype('c:dSEZ', pymaid.get_skids_by_annotation('mw dSEZ feedback 3hop 16-Sept 2020'))
c_pre_dSEZ = Celltype('c:pre-dSEZ', pymaid.get_skids_by_annotation('mw pre-dSEZ feedback 3hop 16-Sept 2020'))

dSEZ_types = Celltype_Analyzer([pre_dSEZ, ds_pre_dSEZ, c_pre_dSEZ, dSEZ, ds_dSEZ, c_dSEZ], mg)
dSEZ_types.set_known_types(known_types)
dSEZ_memberships = dSEZ_types.memberships()

# set up list of RGN cell types and determine membership in known cell types
pre_RGN = Celltype('pre-RGN', pymaid.get_skids_by_annotation('mw pre-RGN'))
ds_pre_RGN = Celltype('ds-pre-RGN', pymaid.get_skids_by_annotation('mw ds-pre-RGN'))
c_pre_RGN = Celltype('c:pre-RGN', pymaid.get_skids_by_annotation('mw pre-RGN feedback 3hop 16-Sept 2020'))
dRGN = Celltype('RGN', pymaid.get_skids_by_annotation('mw RGN'))

RGN_types = Celltype_Analyzer([pre_RGN, ds_pre_RGN, c_pre_RGN, RGN], mg)
RGN_types.set_known_types(known_types)
RGN_memberships = RGN_types.memberships()

#remove = ['sensory', 'KC', 'SEZ-motor']
remove = ['sensory']
dVNC_memberships = dVNC_memberships.drop(index = remove)
dSEZ_memberships = dSEZ_memberships.drop(index = remove)
RGN_memberships = RGN_memberships.drop(index = remove)

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
sns.heatmap(dVNC_memberships, annot=True, fmt=".0%", linewidth = 0.25, vmax = vmax, ax = axs, cmap = cmap, cbar = cbar, annot_kws={"size": fontsize})
plt.savefig('cascades/feedback_through_brain/plots/celltypes_dVNC_pathway.pdf', bbox_inches='tight')

fig, axs = plt.subplots(
    1, 1, figsize = (width, height)
)
sns.heatmap(dSEZ_memberships, annot=True, fmt=".0%", linewidth = 0.25, vmax = vmax, cmap = cmap, cbar = cbar, annot_kws={"size": fontsize})
plt.savefig('cascades/feedback_through_brain/plots/celltypes_dSEZ_pathway.pdf', bbox_inches='tight')

fig, axs = plt.subplots(
    1, 1, figsize = (width*3/5, height)
)
sns.heatmap(RGN_memberships, annot=True, fmt=".0%", linewidth = 0.25, vmax = vmax, cmap = cmap, cbar = cbar, annot_kws={"size": fontsize})
plt.savefig('cascades/feedback_through_brain/plots/celltypes_RG_pathway.pdf', bbox_inches='tight')

# %%
# intersection between types

all_types = Celltype_Analyzer([pre_dVNC, ds_pre_dVNC, c_pre_dVNC, dVNC, ds_dVNC, c_dVNC,
                                pre_dSEZ, ds_pre_dSEZ, c_pre_dSEZ, dSEZ, ds_dSEZ, c_dSEZ,
                                pre_RGN, ds_pre_RGN, c_pre_RGN, RGN], mg)
all_types.set_known_types(known_types)
iou = all_types.compare_membership()

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
# direct connectivity between cell types

cmap = 'Blues'
width = 2.5
height = 2.5

'''
fig, axs = plt.subplots(
    1, 1, figsize=(2.5, 2.5)
)
ax = axs
mat = all_types.connectivtiy(['pre-dVNC', 'ds-pre-dVNC', 'dVNC', 'ds-dVNC', 'pre-dSEZ', 'ds-pre-dSEZ', 'dSEZ', 'ds-dSEZ', 'pre-RGN', 'ds-pre-RGN', 'RGN'])
sns.heatmap(mat, square = True, ax = ax, cmap = cmap)

fig, axs = plt.subplots(
    1, 1, figsize=(2.5, 2.5)
)
ax = axs
mat = all_types.connectivtiy(['pre-dVNC', 'ds-pre-dVNC', 'dVNC', 'ds-dVNC', 'pre-dSEZ', 'ds-pre-dSEZ', 'dSEZ', 'ds-dSEZ', 'pre-RGN', 'ds-pre-RGN', 'RGN'], normalize_pre_num=True)
sns.heatmap(mat, square = True, ax = ax, cmap = cmap)
'''

fig, axs = plt.subplots(
    1, 1, figsize=(2.5, 2.5)
)
ax = axs
mat = all_types.connectivtiy(['pre-dVNC', 'ds-pre-dVNC', 'dVNC', 'ds-dVNC', 'pre-dSEZ', 'ds-pre-dSEZ', 'dSEZ', 'ds-dSEZ', 'pre-RGN', 'ds-pre-RGN', 'RGN'], normalize_post_num=True)
sns.heatmap(mat, square = True, ax = ax, cmap = cmap)
# %%
