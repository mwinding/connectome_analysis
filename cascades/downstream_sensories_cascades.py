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

rm = pymaid.CatmaidInstance(url, name, password, token)

mg = load_metagraph("Gad", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
#mg = load_metagraph("G", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg.calculate_degrees(inplace=True)

adj = mg.adj  # adjacency matrix from the "mg" object

#%%
# pull sensory annotations and then pull associated skids
ORN_skids = pymaid.get_skids_by_annotation('mw ORN')
AN_skids = pymaid.get_skids_by_annotation('mw AN sensories')
MN_skids = pymaid.get_skids_by_annotation('mw MN sensories')
A00c_skids = pymaid.get_skids_by_annotation('mw A00c')
vtd_skids = pymaid.get_skids_by_annotation('mw v\'td')
thermo_skids = pymaid.get_skids_by_annotation('mw thermosensories')
photo_skids = pymaid.get_skids_by_annotation('mw photoreceptors')

output_skids = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain outputs').name))
output_skids = [val for sublist in output_skids for val in sublist]

RG_skids = pymaid.get_skids_by_annotation('mw RG')
dVNC_skids = pymaid.get_skids_by_annotation('mw dVNC')
dSEZ_skids = pymaid.get_skids_by_annotation('mw dSEZ')


#%%
# better understanding how stop nodes work in the context of complex cascades
from src.traverse import Cascade, to_transmission_matrix
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot

# convert skids to indices
ORN_indices = np.where([x in ORN_skids for x in mg.meta.index])[0]
AN_indices = np.where([x in AN_skids for x in mg.meta.index])[0]
MN_indices = np.where([x in MN_skids for x in mg.meta.index])[0]
A00c_indices = np.where([x in A00c_skids for x in mg.meta.index])[0]
vtd_indices = np.where([x in vtd_skids for x in mg.meta.index])[0]
thermo_indices = np.where([x in thermo_skids for x in mg.meta.index])[0]
photo_indices = np.where([x in photo_skids for x in mg.meta.index])[0]

RG_indices = np.where([x in RG_skids for x in mg.meta.index])[0]
dVNC_indices = np.where([x in dVNC_skids for x in mg.meta.index])[0]
dSEZ_indices = np.where([x in dSEZ_skids for x in mg.meta.index])[0]
output_indices = np.where([x in output_skids for x in mg.meta.index])[0]

p = 0.05
max_hops = 16
n_init = 100
simultaneous = True
transition_probs = to_transmission_matrix(adj, p)

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    stop_nodes = output_indices,
    max_hops=max_hops,
    allow_loops = False,
    n_init=n_init,
    simultaneous=simultaneous,
)
ORN_hit_hist = cdispatch.multistart(start_nodes = ORN_indices)
AN_hit_hist = cdispatch.multistart(start_nodes = AN_indices)
MN_hit_hist = cdispatch.multistart(start_nodes = MN_indices)
A00c_hit_hist = cdispatch.multistart(start_nodes = A00c_indices)
vtd_hit_hist = cdispatch.multistart(start_nodes = vtd_indices)
thermo_hit_hist = cdispatch.multistart(start_nodes = thermo_indices)
photo_hit_hist = cdispatch.multistart(start_nodes = photo_indices)

# %%
# plot
fig, axs = plt.subplots(
    7, 1, figsize=(10, 20)
)

fig.tight_layout(pad=2.0)

ax = axs[0]
ax.set_xlabel('ORN signal')    
sns.heatmap(ORN_hit_hist, ax = ax)

ax = axs[1]
ax.set_xlabel('photo signal')    
sns.heatmap(photo_hit_hist, ax = ax)

ax = axs[2]
ax.set_xlabel('thermo signal')    
sns.heatmap(thermo_hit_hist, ax = ax)

ax = axs[3]
ax.set_xlabel('AN signal')    
sns.heatmap(AN_hit_hist, ax = ax)

ax = axs[4]
ax.set_xlabel('MN signal')    
sns.heatmap(MN_hit_hist, ax = ax)

ax = axs[5]
ax.set_xlabel('A00c signal')    
sns.heatmap(A00c_hit_hist, ax = ax)

ax = axs[6]
ax.set_xlabel('vtd signal')    
sns.heatmap(vtd_hit_hist, ax = ax)

plt.savefig('cascades/plots/sensory_modality_signals.pdf', format='pdf', bbox_inches='tight')

import os
os.system('say "code executed"')

# %%
# how close are descending neurons to sensory?
dVNC_ORN_hit = ORN_hit_hist[dVNC_indices, :]
dVNC_AN_hit = AN_hit_hist[dVNC_indices, :]
dVNC_MN_hit = MN_hit_hist[dVNC_indices, :]
dVNC_A00c_hit = A00c_hit_hist[dVNC_indices, :]
dVNC_vtd_hit = vtd_hit_hist[dVNC_indices, :]
dVNC_thermo_hit = thermo_hit_hist[dVNC_indices, :]
dVNC_photo_hit = photo_hit_hist[dVNC_indices, :]

dSEZ_ORN_hit = ORN_hit_hist[dSEZ_indices, :]
dSEZ_AN_hit = AN_hit_hist[dSEZ_indices, :]
dSEZ_MN_hit = MN_hit_hist[dSEZ_indices, :]
dSEZ_A00c_hit = A00c_hit_hist[dSEZ_indices, :]
dSEZ_vtd_hit = vtd_hit_hist[dSEZ_indices, :]
dSEZ_thermo_hit = thermo_hit_hist[dSEZ_indices, :]
dSEZ_photo_hit = photo_hit_hist[dSEZ_indices, :]

RG_ORN_hit = ORN_hit_hist[RG_indices, :]
RG_AN_hit = AN_hit_hist[RG_indices, :]
RG_MN_hit = MN_hit_hist[RG_indices, :]
RG_A00c_hit = A00c_hit_hist[RG_indices, :]
RG_vtd_hit = vtd_hit_hist[RG_indices, :]
RG_thermo_hit = thermo_hit_hist[RG_indices, :]
RG_photo_hit = photo_hit_hist[RG_indices, :]

max_dVNC_hits = len(dVNC_indices)*n_init
max_dSEZ_hits = len(dSEZ_indices)*n_init
max_RG_hits = len(RG_indices)*n_init

dVNC_matrix = pd.DataFrame(([dVNC_ORN_hit.sum(axis = 0), 
                            dVNC_AN_hit.sum(axis = 0), 
                            dVNC_MN_hit.sum(axis = 0), 
                            dVNC_A00c_hit.sum(axis = 0), 
                            dVNC_vtd_hit.sum(axis = 0), 
                            dVNC_thermo_hit.sum(axis = 0), 
                            dVNC_photo_hit.sum(axis = 0)]),
                            index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

dSEZ_matrix = pd.DataFrame(([dSEZ_ORN_hit.sum(axis = 0), 
                            dSEZ_AN_hit.sum(axis = 0), 
                            dSEZ_MN_hit.sum(axis = 0), 
                            dSEZ_A00c_hit.sum(axis = 0), 
                            dSEZ_vtd_hit.sum(axis = 0), 
                            dSEZ_thermo_hit.sum(axis = 0), 
                            dSEZ_photo_hit.sum(axis = 0)]),
                            index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

RG_matrix = pd.DataFrame(([RG_ORN_hit.sum(axis = 0), 
                            RG_AN_hit.sum(axis = 0), 
                            RG_MN_hit.sum(axis = 0), 
                            RG_A00c_hit.sum(axis = 0), 
                            RG_vtd_hit.sum(axis = 0), 
                            RG_thermo_hit.sum(axis = 0), 
                            RG_photo_hit.sum(axis = 0)]),
                            index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

fig, axs = plt.subplots(
    3, 1, figsize=(8, 8)
)

fig.tight_layout(pad=3.0)

vmax = 6000

ax = axs[0]
ax.set_title('Signal to VNC Descending Neurons')
sns.heatmap(dVNC_matrix, ax = ax, vmax = vmax, rasterized=True)
ax.set(xlim = (0, 13))

ax = axs[1]
ax.set_title('Signal to SEZ Descending Neurons')
sns.heatmap(dSEZ_matrix, ax = ax, vmax = vmax, rasterized=True)
ax.set(xlim = (0, 13))

ax = axs[2]
ax.set_title('Signal to Ring Gland Neurons')
sns.heatmap(RG_matrix, ax = ax, vmax = vmax, rasterized=True)
ax.set_xlabel('Hops from sensory')
ax.set(xlim = (0, 13))

plt.savefig('cascades/plots/sensory_modality_signals_to_output.pdf', format='pdf', bbox_inches='tight')

'''
x = range(max_hops)
ax = axs
sns.lineplot(x = x, y = dVNC_ORN_hit.sum(axis = 0), ax = ax, label = 'ORN')
sns.lineplot(x = x, y = dVNC_AN_hit.sum(axis = 0), ax = ax, label = 'AN')
sns.lineplot(x = x, y = dVNC_MN_hit.sum(axis = 0), ax = ax, label = 'MN')
sns.lineplot(x = x, y = dVNC_A00c_hit.sum(axis = 0), ax = ax, label = 'A00c')
sns.lineplot(x = x, y = dVNC_vtd_hit.sum(axis = 0), ax = ax, label = 'vtd')
sns.lineplot(x = x, y = dVNC_thermo_hit.sum(axis = 0), ax = ax, label = 'thermo')
sns.lineplot(x = x, y = dVNC_photo_hit.sum(axis = 0), ax = ax, label = 'photo')
'''


# %%
# num of descendings at each level
count50_matrix = pd.DataFrame(([np.array(dVNC_ORN_hit>50).sum(axis = 0),
                                np.array(dVNC_AN_hit>50).sum(axis = 0),
                                np.array(dVNC_MN_hit>50).sum(axis = 0),
                                np.array(dVNC_A00c_hit>50).sum(axis = 0),
                                np.array(dVNC_vtd_hit>50).sum(axis = 0),
                                np.array(dVNC_thermo_hit>50).sum(axis = 0),
                                np.array(dVNC_photo_hit>50).sum(axis = 0)]),
                                index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

# %%
# sensory characterization of brain


# %%
import plotly.express as px
from pandas.plotting import parallel_coordinates

# sensory characterization of clusters
sensory_profile = pd.DataFrame(([ORN_hit_hist.sum(axis = 1), 
                    AN_hit_hist.sum(axis = 1),
                    MN_hit_hist.sum(axis = 1), 
                    A00c_hit_hist.sum(axis = 1),
                    vtd_hit_hist.sum(axis = 1), 
                    thermo_hit_hist.sum(axis = 1),
                    photo_hit_hist.sum(axis = 1)]), 
                    index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

sensory_profile0 = pd.DataFrame(([ORN_hit_hist[:, 0], 
                    AN_hit_hist[:, 0],
                    MN_hit_hist[:, 0], 
                    A00c_hit_hist[:, 0],
                    vtd_hit_hist[:, 0], 
                    thermo_hit_hist[:, 0],
                    photo_hit_hist[:, 0]]), 
                    index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

sensory_profile1 = pd.DataFrame(([ORN_hit_hist[:, 1], 
                    AN_hit_hist[:, 1],
                    MN_hit_hist[:, 1], 
                    A00c_hit_hist[:, 1],
                    vtd_hit_hist[:, 1], 
                    thermo_hit_hist[:, 1],
                    photo_hit_hist[:, 1]]), 
                    index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

sensory_profile2 = pd.DataFrame(([ORN_hit_hist[:, 2], 
                    AN_hit_hist[:, 2],
                    MN_hit_hist[:, 2], 
                    A00c_hit_hist[:, 2],
                    vtd_hit_hist[:, 2], 
                    thermo_hit_hist[:, 2],
                    photo_hit_hist[:, 2]]), 
                    index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

sensory_profile3 = pd.DataFrame(([ORN_hit_hist[:, 3], 
                    AN_hit_hist[:, 3],
                    MN_hit_hist[:, 3], 
                    A00c_hit_hist[:, 3],
                    vtd_hit_hist[:, 3], 
                    thermo_hit_hist[:, 3],
                    photo_hit_hist[:, 3]]), 
                    index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

sensory_profile4 = pd.DataFrame(([ORN_hit_hist[:, 4], 
                    AN_hit_hist[:, 4],
                    MN_hit_hist[:, 4], 
                    A00c_hit_hist[:, 4],
                    vtd_hit_hist[:, 4], 
                    thermo_hit_hist[:, 4],
                    photo_hit_hist[:, 4]]), 
                    index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

sensory_profile5 = pd.DataFrame(([ORN_hit_hist[:, 5], 
                    AN_hit_hist[:, 5],
                    MN_hit_hist[:, 5], 
                    A00c_hit_hist[:, 5],
                    vtd_hit_hist[:, 5], 
                    thermo_hit_hist[:, 5],
                    photo_hit_hist[:, 5]]), 
                    index = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

sensory_profile = sensory_profile.T

# %%
# multisensory elements of each layer of each sensory modality
# multisensory elements were summed for all hops, doesn't work that well

threshold = 0

ORN0_indices = np.where(ORN_hit_hist[:, 0]>threshold)[0]
ORN1_indices = np.where(ORN_hit_hist[:, 1]>threshold)[0]
ORN2_indices = np.where(ORN_hit_hist[:, 2]>threshold)[0]
ORN3_indices = np.where(ORN_hit_hist[:, 3]>threshold)[0]
ORN4_indices = np.where(ORN_hit_hist[:, 4]>threshold)[0]
ORN5_indices = np.where(ORN_hit_hist[:, 5]>threshold)[0]

ORN_profile = pd.DataFrame([np.array(sensory_profile.iloc[ORN0_indices, :].sum(axis=0)/len(ORN0_indices)), 
                    np.array(sensory_profile.iloc[ORN1_indices, :].sum(axis=0)/len(ORN1_indices)),
                    np.array(sensory_profile.iloc[ORN2_indices, :].sum(axis=0)/len(ORN2_indices)),
                    np.array(sensory_profile.iloc[ORN3_indices, :].sum(axis=0)/len(ORN3_indices)), 
                    np.array(sensory_profile.iloc[ORN4_indices, :].sum(axis=0)/len(ORN4_indices)), 
                    np.array(sensory_profile.iloc[ORN5_indices, :].sum(axis=0)/len(ORN5_indices))],
                    index = ['ORN0', 'ORN1', 'ORN2', 'ORN3', 'ORN4', 'ORN5'], columns = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

AN0_indices = np.where(AN_hit_hist[:, 0]>threshold)[0]
AN1_indices = np.where(AN_hit_hist[:, 1]>threshold)[0]
AN2_indices = np.where(AN_hit_hist[:, 2]>threshold)[0]
AN3_indices = np.where(AN_hit_hist[:, 3]>threshold)[0]
AN4_indices = np.where(AN_hit_hist[:, 4]>threshold)[0]
AN5_indices = np.where(AN_hit_hist[:, 5]>threshold)[0]

AN_profile = pd.DataFrame([np.array(sensory_profile.iloc[AN0_indices, :].sum(axis=0)/len(AN0_indices)), 
                    np.array(sensory_profile.iloc[AN1_indices, :].sum(axis=0)/len(AN1_indices)),
                    np.array(sensory_profile.iloc[AN2_indices, :].sum(axis=0)/len(AN2_indices)),
                    np.array(sensory_profile.iloc[AN3_indices, :].sum(axis=0)/len(AN3_indices)), 
                    np.array(sensory_profile.iloc[AN4_indices, :].sum(axis=0)/len(AN4_indices)), 
                    np.array(sensory_profile.iloc[AN5_indices, :].sum(axis=0)/len(AN5_indices))],
                    index = ['AN0', 'AN1', 'AN2', 'AN3', 'AN4', 'AN5'], columns = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

MN0_indices = np.where(MN_hit_hist[:, 0]>threshold)[0]
MN1_indices = np.where(MN_hit_hist[:, 1]>threshold)[0]
MN2_indices = np.where(MN_hit_hist[:, 2]>threshold)[0]
MN3_indices = np.where(MN_hit_hist[:, 3]>threshold)[0]
MN4_indices = np.where(MN_hit_hist[:, 4]>threshold)[0]
MN5_indices = np.where(MN_hit_hist[:, 5]>threshold)[0]

MN_profile = pd.DataFrame([np.array(sensory_profile.iloc[MN0_indices, :].sum(axis=0)/len(MN0_indices)), 
                    np.array(sensory_profile.iloc[MN1_indices, :].sum(axis=0)/len(MN1_indices)),
                    np.array(sensory_profile.iloc[MN2_indices, :].sum(axis=0)/len(MN2_indices)),
                    np.array(sensory_profile.iloc[MN3_indices, :].sum(axis=0)/len(MN3_indices)), 
                    np.array(sensory_profile.iloc[MN4_indices, :].sum(axis=0)/len(MN4_indices)), 
                    np.array(sensory_profile.iloc[MN5_indices, :].sum(axis=0)/len(MN5_indices))],
                    index = ['MN0', 'MN1', 'MN2', 'MN3', 'MN4', 'MN5'], columns = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

A00c0_indices = np.where(A00c_hit_hist[:, 0]>threshold)[0]
A00c1_indices = np.where(A00c_hit_hist[:, 1]>threshold)[0]
A00c2_indices = np.where(A00c_hit_hist[:, 2]>threshold)[0]
A00c3_indices = np.where(A00c_hit_hist[:, 3]>threshold)[0]
A00c4_indices = np.where(A00c_hit_hist[:, 4]>threshold)[0]
A00c5_indices = np.where(A00c_hit_hist[:, 5]>threshold)[0]

A00c_profile = pd.DataFrame([np.array(sensory_profile.iloc[A00c0_indices, :].sum(axis=0)/len(A00c0_indices)), 
                    np.array(sensory_profile.iloc[A00c1_indices, :].sum(axis=0)/len(A00c1_indices)),
                    np.array(sensory_profile.iloc[A00c2_indices, :].sum(axis=0)/len(A00c2_indices)),
                    np.array(sensory_profile.iloc[A00c3_indices, :].sum(axis=0)/len(A00c3_indices)), 
                    np.array(sensory_profile.iloc[A00c4_indices, :].sum(axis=0)/len(A00c4_indices)), 
                    np.array(sensory_profile.iloc[A00c5_indices, :].sum(axis=0)/len(A00c5_indices))],
                    index = ['A00c0', 'A00c1', 'A00c2', 'A00c3', 'A00c4', 'A00c5'], columns = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

vtd0_indices = np.where(vtd_hit_hist[:, 0]>threshold)[0]
vtd1_indices = np.where(vtd_hit_hist[:, 1]>threshold)[0]
vtd2_indices = np.where(vtd_hit_hist[:, 2]>threshold)[0]
vtd3_indices = np.where(vtd_hit_hist[:, 3]>threshold)[0]
vtd4_indices = np.where(vtd_hit_hist[:, 4]>threshold)[0]
vtd5_indices = np.where(vtd_hit_hist[:, 5]>threshold)[0]

vtd_profile = pd.DataFrame([np.array(sensory_profile.iloc[vtd0_indices, :].sum(axis=0)/len(vtd0_indices)), 
                    np.array(sensory_profile.iloc[vtd1_indices, :].sum(axis=0)/len(vtd1_indices)),
                    np.array(sensory_profile.iloc[vtd2_indices, :].sum(axis=0)/len(vtd2_indices)),
                    np.array(sensory_profile.iloc[vtd3_indices, :].sum(axis=0)/len(vtd3_indices)), 
                    np.array(sensory_profile.iloc[vtd4_indices, :].sum(axis=0)/len(vtd4_indices)), 
                    np.array(sensory_profile.iloc[vtd5_indices, :].sum(axis=0)/len(vtd5_indices))],
                    index = ['vtd0', 'vtd1', 'vtd2', 'vtd3', 'vtd4', 'vtd5'], columns = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

thermo0_indices = np.where(thermo_hit_hist[:, 0]>threshold)[0]
thermo1_indices = np.where(thermo_hit_hist[:, 1]>threshold)[0]
thermo2_indices = np.where(thermo_hit_hist[:, 2]>threshold)[0]
thermo3_indices = np.where(thermo_hit_hist[:, 3]>threshold)[0]
thermo4_indices = np.where(thermo_hit_hist[:, 4]>threshold)[0]
thermo5_indices = np.where(thermo_hit_hist[:, 5]>threshold)[0]

thermo_profile = pd.DataFrame([np.array(sensory_profile.iloc[thermo0_indices, :].sum(axis=0)/len(thermo0_indices)), 
                    np.array(sensory_profile.iloc[thermo1_indices, :].sum(axis=0)/len(thermo1_indices)),
                    np.array(sensory_profile.iloc[thermo2_indices, :].sum(axis=0)/len(thermo2_indices)),
                    np.array(sensory_profile.iloc[thermo3_indices, :].sum(axis=0)/len(thermo3_indices)), 
                    np.array(sensory_profile.iloc[thermo4_indices, :].sum(axis=0)/len(thermo4_indices)), 
                    np.array(sensory_profile.iloc[thermo5_indices, :].sum(axis=0)/len(thermo5_indices))],
                    index = ['thermo0', 'thermo1', 'thermo2', 'thermo3', 'thermo4', 'thermo5'], columns = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

photo0_indices = np.where(photo_hit_hist[:, 0]>threshold)[0]
photo1_indices = np.where(photo_hit_hist[:, 1]>threshold)[0]
photo2_indices = np.where(photo_hit_hist[:, 2]>threshold)[0]
photo3_indices = np.where(photo_hit_hist[:, 3]>threshold)[0]
photo4_indices = np.where(photo_hit_hist[:, 4]>threshold)[0]
photo5_indices = np.where(photo_hit_hist[:, 5]>threshold)[0]

photo_profile = pd.DataFrame([np.array(sensory_profile.iloc[photo0_indices, :].sum(axis=0)/len(photo0_indices)), 
                    np.array(sensory_profile.iloc[photo1_indices, :].sum(axis=0)/len(photo1_indices)),
                    np.array(sensory_profile.iloc[photo2_indices, :].sum(axis=0)/len(photo2_indices)),
                    np.array(sensory_profile.iloc[photo3_indices, :].sum(axis=0)/len(photo3_indices)), 
                    np.array(sensory_profile.iloc[photo4_indices, :].sum(axis=0)/len(photo4_indices)), 
                    np.array(sensory_profile.iloc[photo5_indices, :].sum(axis=0)/len(photo5_indices))],
                    index = ['photo0', 'photo1', 'photo2', 'photo3', 'photo4', 'photo5'], columns = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

ORN0_indices = np.where(ORN_hit_hist[:, 0]>threshold)[0]
ORN1_indices = np.where(ORN_hit_hist[:, 1]>threshold)[0]
ORN2_indices = np.where(ORN_hit_hist[:, 2]>threshold)[0]
ORN3_indices = np.where(ORN_hit_hist[:, 3]>threshold)[0]
ORN4_indices = np.where(ORN_hit_hist[:, 4]>threshold)[0]
ORN5_indices = np.where(ORN_hit_hist[:, 5]>threshold)[0]

AN0_indices = np.where(AN_hit_hist[:, 0]>threshold)[0]
AN1_indices = np.where(AN_hit_hist[:, 1]>threshold)[0]
AN2_indices = np.where(AN_hit_hist[:, 2]>threshold)[0]
AN3_indices = np.where(AN_hit_hist[:, 3]>threshold)[0]
AN4_indices = np.where(AN_hit_hist[:, 4]>threshold)[0]
AN5_indices = np.where(AN_hit_hist[:, 5]>threshold)[0]

MN0_indices = np.where(MN_hit_hist[:, 0]>threshold)[0]
MN1_indices = np.where(MN_hit_hist[:, 1]>threshold)[0]
MN2_indices = np.where(MN_hit_hist[:, 2]>threshold)[0]
MN3_indices = np.where(MN_hit_hist[:, 3]>threshold)[0]
MN4_indices = np.where(MN_hit_hist[:, 4]>threshold)[0]
MN5_indices = np.where(MN_hit_hist[:, 5]>threshold)[0]

A00c0_indices = np.where(A00c_hit_hist[:, 0]>threshold)[0]
A00c1_indices = np.where(A00c_hit_hist[:, 1]>threshold)[0]
A00c2_indices = np.where(A00c_hit_hist[:, 2]>threshold)[0]
A00c3_indices = np.where(A00c_hit_hist[:, 3]>threshold)[0]
A00c4_indices = np.where(A00c_hit_hist[:, 4]>threshold)[0]
A00c5_indices = np.where(A00c_hit_hist[:, 5]>threshold)[0]

vtd0_indices = np.where(vtd_hit_hist[:, 0]>threshold)[0]
vtd1_indices = np.where(vtd_hit_hist[:, 1]>threshold)[0]
vtd2_indices = np.where(vtd_hit_hist[:, 2]>threshold)[0]
vtd3_indices = np.where(vtd_hit_hist[:, 3]>threshold)[0]
vtd4_indices = np.where(vtd_hit_hist[:, 4]>threshold)[0]
vtd5_indices = np.where(vtd_hit_hist[:, 5]>threshold)[0]

thermo0_indices = np.where(thermo_hit_hist[:, 0]>threshold)[0]
thermo1_indices = np.where(thermo_hit_hist[:, 1]>threshold)[0]
thermo2_indices = np.where(thermo_hit_hist[:, 2]>threshold)[0]
thermo3_indices = np.where(thermo_hit_hist[:, 3]>threshold)[0]
thermo4_indices = np.where(thermo_hit_hist[:, 4]>threshold)[0]
thermo5_indices = np.where(thermo_hit_hist[:, 5]>threshold)[0]

photo0_indices = np.where(photo_hit_hist[:, 0]>threshold)[0]
photo1_indices = np.where(photo_hit_hist[:, 1]>threshold)[0]
photo2_indices = np.where(photo_hit_hist[:, 2]>threshold)[0]
photo3_indices = np.where(photo_hit_hist[:, 3]>threshold)[0]
photo4_indices = np.where(photo_hit_hist[:, 4]>threshold)[0]
photo5_indices = np.where(photo_hit_hist[:, 5]>threshold)[0]

ORN_profile = pd.DataFrame([np.array(sensory_profile.iloc[ORN0_indices, :].sum(axis=0)/len(ORN0_indices)), 
                    np.array(sensory_profile.iloc[ORN1_indices, :].sum(axis=0)/len(ORN1_indices)),
                    np.array(sensory_profile.iloc[ORN2_indices, :].sum(axis=0)/len(ORN2_indices)),
                    np.array(sensory_profile.iloc[ORN3_indices, :].sum(axis=0)/len(ORN3_indices)), 
                    np.array(sensory_profile.iloc[ORN4_indices, :].sum(axis=0)/len(ORN4_indices)), 
                    np.array(sensory_profile.iloc[ORN5_indices, :].sum(axis=0)/len(ORN5_indices))],
                    index = ['ORN0', 'ORN1', 'ORN2', 'ORN3', 'ORN4', 'ORN5'], columns = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

AN_profile = pd.DataFrame([np.array(sensory_profile.iloc[AN0_indices, :].sum(axis=0)/len(AN0_indices)), 
                    np.array(sensory_profile.iloc[AN1_indices, :].sum(axis=0)/len(AN1_indices)),
                    np.array(sensory_profile.iloc[AN2_indices, :].sum(axis=0)/len(AN2_indices)),
                    np.array(sensory_profile.iloc[AN3_indices, :].sum(axis=0)/len(AN3_indices)), 
                    np.array(sensory_profile.iloc[AN4_indices, :].sum(axis=0)/len(AN4_indices)), 
                    np.array(sensory_profile.iloc[AN5_indices, :].sum(axis=0)/len(AN5_indices))],
                    index = ['AN0', 'AN1', 'AN2', 'AN3', 'AN4', 'AN5'], columns = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

MN_profile = pd.DataFrame([np.array(sensory_profile.iloc[MN0_indices, :].sum(axis=0)/len(MN0_indices)), 
                    np.array(sensory_profile.iloc[MN1_indices, :].sum(axis=0)/len(MN1_indices)),
                    np.array(sensory_profile.iloc[MN2_indices, :].sum(axis=0)/len(MN2_indices)),
                    np.array(sensory_profile.iloc[MN3_indices, :].sum(axis=0)/len(MN3_indices)), 
                    np.array(sensory_profile.iloc[MN4_indices, :].sum(axis=0)/len(MN4_indices)), 
                    np.array(sensory_profile.iloc[MN5_indices, :].sum(axis=0)/len(MN5_indices))],
                    index = ['MN0', 'MN1', 'MN2', 'MN3', 'MN4', 'MN5'], columns = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

A00c_profile = pd.DataFrame([np.array(sensory_profile.iloc[A00c0_indices, :].sum(axis=0)/len(A00c0_indices)), 
                    np.array(sensory_profile.iloc[A00c1_indices, :].sum(axis=0)/len(A00c1_indices)),
                    np.array(sensory_profile.iloc[A00c2_indices, :].sum(axis=0)/len(A00c2_indices)),
                    np.array(sensory_profile.iloc[A00c3_indices, :].sum(axis=0)/len(A00c3_indices)), 
                    np.array(sensory_profile.iloc[A00c4_indices, :].sum(axis=0)/len(A00c4_indices)), 
                    np.array(sensory_profile.iloc[A00c5_indices, :].sum(axis=0)/len(A00c5_indices))],
                    index = ['A00c0', 'A00c1', 'A00c2', 'A00c3', 'A00c4', 'A00c5'], columns = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

vtd_profile = pd.DataFrame([np.array(sensory_profile.iloc[vtd0_indices, :].sum(axis=0)/len(vtd0_indices)), 
                    np.array(sensory_profile.iloc[vtd1_indices, :].sum(axis=0)/len(vtd1_indices)),
                    np.array(sensory_profile.iloc[vtd2_indices, :].sum(axis=0)/len(vtd2_indices)),
                    np.array(sensory_profile.iloc[vtd3_indices, :].sum(axis=0)/len(vtd3_indices)), 
                    np.array(sensory_profile.iloc[vtd4_indices, :].sum(axis=0)/len(vtd4_indices)), 
                    np.array(sensory_profile.iloc[vtd5_indices, :].sum(axis=0)/len(vtd5_indices))],
                    index = ['vtd0', 'vtd1', 'vtd2', 'vtd3', 'vtd4', 'vtd5'], columns = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

thermo_profile = pd.DataFrame([np.array(sensory_profile.iloc[thermo0_indices, :].sum(axis=0)/len(thermo0_indices)), 
                    np.array(sensory_profile.iloc[thermo1_indices, :].sum(axis=0)/len(thermo1_indices)),
                    np.array(sensory_profile.iloc[thermo2_indices, :].sum(axis=0)/len(thermo2_indices)),
                    np.array(sensory_profile.iloc[thermo3_indices, :].sum(axis=0)/len(thermo3_indices)), 
                    np.array(sensory_profile.iloc[thermo4_indices, :].sum(axis=0)/len(thermo4_indices)), 
                    np.array(sensory_profile.iloc[thermo5_indices, :].sum(axis=0)/len(thermo5_indices))],
                    index = ['thermo0', 'thermo1', 'thermo2', 'thermo3', 'thermo4', 'thermo5'], columns = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

photo_profile = pd.DataFrame([np.array(sensory_profile.iloc[photo0_indices, :].sum(axis=0)/len(photo0_indices)), 
                    np.array(sensory_profile.iloc[photo1_indices, :].sum(axis=0)/len(photo1_indices)),
                    np.array(sensory_profile.iloc[photo2_indices, :].sum(axis=0)/len(photo2_indices)),
                    np.array(sensory_profile.iloc[photo3_indices, :].sum(axis=0)/len(photo3_indices)), 
                    np.array(sensory_profile.iloc[photo4_indices, :].sum(axis=0)/len(photo4_indices)), 
                    np.array(sensory_profile.iloc[photo5_indices, :].sum(axis=0)/len(photo5_indices))],
                    index = ['photo0', 'photo1', 'photo2', 'photo3', 'photo4', 'photo5'], columns = ['ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'])

fig, axs = plt.subplots(
    4, 2, figsize=(5, 8)
)

ax = axs[0, 0] 
sns.heatmap(ORN_profile.T, ax = ax)
#fig.tight_layout(pad=3.0)

ax = axs[1, 0]
sns.heatmap(AN_profile.T, ax = ax)

ax = axs[2, 0]
sns.heatmap(MN_profile.T, ax = ax)

ax = axs[3, 0]
sns.heatmap(A00c_profile.T, ax = ax)

ax = axs[0, 1]
sns.heatmap(vtd_profile.T, ax = ax)

ax = axs[1, 1]
sns.heatmap(thermo_profile.T, ax = ax)

ax = axs[2, 1]
sns.heatmap(photo_profile.T, ax = ax)

# %%
# multisensory elements per layer (apples to apples)
# %%
# parallel coordinate plot
fig, axs = plt.subplots(
    1, 1, figsize=(8, 8)
)

ax = axs
axs.set(ylim = (0, 100))
#fig.tight_layout(pad=3.0)

sensory_profile['class'] = np.zeros(len(sensory_profile))

parallel_coordinates(sensory_profile.iloc[np.where(ORN_hit_hist[:, 0]>50)[0], :], class_column = 'class', ax = ax, alpha = 0.5, color = 'blue')
parallel_coordinates(sensory_profile.iloc[np.where(ORN_hit_hist[:, 1]>50)[0], :], class_column = 'class', ax = ax, alpha = 0.05, color = 'orange')
parallel_coordinates(sensory_profile.iloc[np.where(ORN_hit_hist[:, 2]>50)[0], :], class_column = 'class', ax = ax, alpha = 0.05, color = 'green')
parallel_coordinates(sensory_profile.iloc[np.where(ORN_hit_hist[:, 3]>50)[0], :], class_column = 'class', ax = ax, alpha = 0.05, color = 'red')
plt.show()
#px.parallel_coordinates(sensory_profile0.T.iloc[np.where(ORN_hit_hist[:, 0]>50)[0], :], labels={'ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'})
#px.parallel_coordinates(sensory_profile1.T.iloc[np.where(ORN_hit_hist[:, 1]>50)[0], :], labels={'ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'})
#px.parallel_coordinates(sensory_profile2.T.iloc[np.where(ORN_hit_hist[:, 2]>50)[0], :], labels={'ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'})
#px.parallel_coordinates(sensory_profile3.T.iloc[np.where(ORN_hit_hist[:, 3]>50)[0], :], labels={'ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'})
#px.parallel_coordinates(sensory_profile4.T.iloc[np.where(ORN_hit_hist[:, 4]>50)[0], :], labels={'ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'})

#px.parallel_coordinates(sensory_profile.T.iloc[np.where(ORN_hit_hist[:, 0]>50)[0], :], labels={'ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'})
#px.parallel_coordinates(sensory_profile.T.iloc[np.where(ORN_hit_hist[:, 1]>50)[0], :], labels={'ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'})
#px.parallel_coordinates(sensory_profile.T.iloc[np.where(ORN_hit_hist[:, 2]>50)[0], :], labels={'ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'})
#px.parallel_coordinates(sensory_profile.T.iloc[np.where(ORN_hit_hist[:, 3]>50)[0], :], labels={'ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'})
#px.parallel_coordinates(sensory_profile.T.iloc[np.where(ORN_hit_hist[:, 4]>50)[0], :], labels={'ORN', 'AN', 'MN', 'A00c', 'vtd', 'thermo', 'photo'})


# %%
