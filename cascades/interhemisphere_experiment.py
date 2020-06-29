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
import numpy.random as random

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
# pull ORN skids and then divide into left and right hemisphere
ORN_skids = pymaid.get_skids_by_annotation('mw ORN')
AN_skids = pymaid.get_skids_by_annotation('mw AN sensories')
MN_skids = pymaid.get_skids_by_annotation('mw MN sensories')
A00c_skids = pymaid.get_skids_by_annotation('mw A00c')
vtd_skids = pymaid.get_skids_by_annotation('mw v\'td')
thermo_skids = pymaid.get_skids_by_annotation('mw thermosensories')
photo_skids = pymaid.get_skids_by_annotation('mw photoreceptors')
left = pymaid.get_skids_by_annotation('mw left')
right = pymaid.get_skids_by_annotation('mw right')

RG_skids = pymaid.get_skids_by_annotation('mw RG')
dVNC_skids = pymaid.get_skids_by_annotation('mw dVNC')
dSEZ_skids = pymaid.get_skids_by_annotation('mw dSEZ')

output_skids = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain outputs').name))
output_skids = [val for sublist in output_skids for val in sublist]

input_skids = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain inputs').name))
input_skids = [val for sublist in input_skids for val in sublist]

# identify left and right side for each skid category
def split_hemilateral_to_indices(skids, left, right, mg):
    intersect_left = np.intersect1d(skids, left)
    indices_left = np.where([x in intersect_left for x in mg.meta.index])[0]
    intersect_right = np.intersect1d(skids, right)
    indices_right = np.where([x in intersect_right for x in mg.meta.index])[0]

    return(indices_left, indices_right, intersect_left, intersect_right)

ORN_indices_left, ORN_indices_right, ORN_left, ORN_right = split_hemilateral_to_indices(ORN_skids, left, right, mg)
AN_indices_left, AN_indices_right, AN_left, AN_right = split_hemilateral_to_indices(AN_skids, left, right, mg)
MN_indices_left, MN_indices_right, MN_left, MN_right = split_hemilateral_to_indices(MN_skids, left, right, mg)
A00c_indices_left, A00c_indices_right, A00c_left, A00c_right = split_hemilateral_to_indices(A00c_skids, left, right, mg)
vtd_indices_left, vtd_indices_right, vtd_left, vtd_right = split_hemilateral_to_indices(vtd_skids, left, right, mg)
thermo_indices_left, thermo_indices_right, thermo_left, thermo_right = split_hemilateral_to_indices(thermo_skids, left, right, mg)
photo_indices_left, photo_indices_right, photo_left, photo_right = split_hemilateral_to_indices(photo_skids, left, right, mg)

RG_indices_left, RG_indices_right, RG_left, RG_right = split_hemilateral_to_indices(RG_skids, left, right, mg)
dVNC_indices_left, dVNC_indices_right, dVNC_left, dVNC_right = split_hemilateral_to_indices(dVNC_skids, left, right, mg)
dSEZ_indices_left, dSEZ_indices_right, dSEZ_left, dSEZ_right = split_hemilateral_to_indices(dSEZ_skids, left, right, mg)

input_indices_left, input_indices_right, input_left, input_right = split_hemilateral_to_indices(input_skids, left, right, mg)
output_indices = np.where([x in output_skids for x in mg.meta.index])[0]

# add A00c left to inputs right and A00c right to inputs left
    # this is because A00cs are ascending contralateral inputs

# identify appropriate indices for A00c neurons
input_indices_left_A00c_index = np.where([x in A00c_indices_left for x in input_indices_left])[0]
input_indices_right_A00c_index = np.where([x in A00c_indices_right for x in input_indices_right])[0]
input_left_A00c_index = np.where([x in A00c_left for x in input_left])[0]
input_right_A00c_index = np.where([x in A00c_right for x in input_right])[0]

# delete A00c skids/indices from each np.array
input_indices_left = np.delete(input_indices_left, input_indices_left_A00c_index)
input_indices_right = np.delete(input_indices_right, input_indices_right_A00c_index)
input_left = np.delete(input_left, input_left_A00c_index)
input_right = np.delete(input_right, input_right_A00c_index)

# add appropriate A00c skids/indices
input_indices_left = np.append(input_indices_left, A00c_indices_right)
input_indices_right = np.append(input_indices_right, A00c_indices_left)
input_left = np.append(input_left, A00c_right)
input_right = np.append(input_right, A00c_left)


#%%
'''
random.seed(0)
ORN_indices_left_half = ORN_indices_left[random.randint(0, len(ORN_indices_left), int(np.round(len(ORN_indices_left)/2)))]
ORN_indices_right_half = ORN_indices_right[random.randint(0, len(ORN_indices_right), int(np.round(len(ORN_indices_right)/2)))]
'''
#%%
# cascades from left and right ORNs

from src.traverse import Cascade, to_transmission_matrix
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot

p = 0.05
max_hops = 10
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

ORN_hit_hist_left = cdispatch.multistart(start_nodes = ORN_indices_left)
ORN_hit_hist_right = cdispatch.multistart(start_nodes = ORN_indices_right)
AN_hit_hist_left = cdispatch.multistart(start_nodes = AN_indices_left)
AN_hit_hist_right = cdispatch.multistart(start_nodes = AN_indices_right)
MN_hit_hist_left = cdispatch.multistart(start_nodes = MN_indices_left)
MN_hit_hist_right = cdispatch.multistart(start_nodes = MN_indices_right)
A00c_hit_hist_left = cdispatch.multistart(start_nodes = A00c_indices_left)
A00c_hit_hist_right = cdispatch.multistart(start_nodes = A00c_indices_right)
vtd_hit_hist_left = cdispatch.multistart(start_nodes = vtd_indices_left)
vtd_hit_hist_right = cdispatch.multistart(start_nodes = vtd_indices_right)
thermo_hit_hist_left = cdispatch.multistart(start_nodes = thermo_indices_left)
thermo_hit_hist_right = cdispatch.multistart(start_nodes = thermo_indices_right)
photo_hit_hist_left = cdispatch.multistart(start_nodes = photo_indices_left)
photo_hit_hist_right = cdispatch.multistart(start_nodes = photo_indices_right)

all_inputs_hit_hist_left = cdispatch.multistart(start_nodes = input_indices_left)
all_inputs_hit_hist_right = cdispatch.multistart(start_nodes = input_indices_right)

#ORN_hit_hist_left_half = cdispatch.multistart(start_nodes = ORN_indices_left_half)
#ORN_hit_hist_right_half = cdispatch.multistart(start_nodes = ORN_indices_right_half)

# %%
# signal through ipsilateral and contralateral structures
ipsi = pymaid.get_skids_by_annotation('mw brain ipsilateral')
contra = pymaid.get_skids_by_annotation('mw brain contralateral')

ipsi_indices_left, ipsi_indices_right, ipsi_left, ipsi_right = split_hemilateral_to_indices(ipsi, left, right, mg)
contra_indices_left, contra_indices_right, contra_left, contra_right = split_hemilateral_to_indices(contra, left, right, mg)

ipsi_indices_left = np.concatenate((ORN_indices_left, AN_indices_left,
                                    MN_indices_left, vtd_indices_left,
                                    thermo_indices_left, photo_indices_left,
                                    ipsi_indices_left), axis = 0)

ipsi_indices_right = np.concatenate((ORN_indices_right, AN_indices_right,
                                    MN_indices_right, vtd_indices_right,
                                    thermo_indices_right, photo_indices_right,
                                    ipsi_indices_right), axis = 0)

contra_indices_left = np.concatenate((A00c_indices_left,
                                    contra_indices_left), axis = 0)

contra_indices_right = np.concatenate((A00c_indices_right,
                                    contra_indices_right), axis = 0)
# %%
# number of ipsi or contra neurons visited per hop
# shows nicely the flow of information through two hemispheres
# folds left and right ipsilateral and left and right contralateral together
fig, axs = plt.subplots(
    6, 1, figsize=(8, 20)
)
threshold = 50
fig.tight_layout(pad=2.5)

ORN_ic = pd.DataFrame([(ORN_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0) + (ORN_hit_hist_right[ipsi_indices_right]>threshold).sum(axis = 0), 
                            (ORN_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0) + (ORN_hit_hist_right[contra_indices_right]>threshold).sum(axis = 0),
                            (ORN_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0) + (ORN_hit_hist_right[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (ORN_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0) + (ORN_hit_hist_right[contra_indices_left]>threshold).sum(axis = 0)], 
                            index = ['Ipsilateral same side', 'Contralateral same side', 
                                    'Ipsilateral opposite side', 'Contralateral opposite side'])

AN_ic = pd.DataFrame([(AN_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0) + (AN_hit_hist_right[ipsi_indices_right]>threshold).sum(axis = 0), 
                            (AN_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0) + (AN_hit_hist_right[contra_indices_right]>threshold).sum(axis = 0),
                            (AN_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0) + (AN_hit_hist_right[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (AN_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0) + (AN_hit_hist_right[contra_indices_left]>threshold).sum(axis = 0)], 
                            index = ['Ipsilateral same side', 'Contralateral same side', 
                                    'Ipsilateral opposite side', 'Contralateral opposite side'])

MN_ic = pd.DataFrame([(MN_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0) + (MN_hit_hist_right[ipsi_indices_right]>threshold).sum(axis = 0), 
                            (MN_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0) + (MN_hit_hist_right[contra_indices_right]>threshold).sum(axis = 0),
                            (MN_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0) + (MN_hit_hist_right[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (MN_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0) + (MN_hit_hist_right[contra_indices_left]>threshold).sum(axis = 0)], 
                            index = ['Ipsilateral same side', 'Contralateral same side', 
                                    'Ipsilateral opposite side', 'Contralateral opposite side'])

A00c_ic = pd.DataFrame([(A00c_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0) + (A00c_hit_hist_right[ipsi_indices_right]>threshold).sum(axis = 0), 
                            (A00c_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0) + (A00c_hit_hist_right[contra_indices_right]>threshold).sum(axis = 0),
                            (A00c_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0) + (A00c_hit_hist_right[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (A00c_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0) + (A00c_hit_hist_right[contra_indices_left]>threshold).sum(axis = 0)], 
                            index = ['Ipsilateral same side', 'Contralateral same side', 
                                    'Ipsilateral opposite side', 'Contralateral opposite side'])
vtd_ic = pd.DataFrame([(vtd_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0) + (vtd_hit_hist_right[ipsi_indices_right]>threshold).sum(axis = 0), 
                            (vtd_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0) + (vtd_hit_hist_right[contra_indices_right]>threshold).sum(axis = 0),
                            (vtd_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0) + (vtd_hit_hist_right[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (vtd_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0) + (vtd_hit_hist_right[contra_indices_left]>threshold).sum(axis = 0)], 
                            index = ['Ipsilateral same side', 'Contralateral same side', 
                                    'Ipsilateral opposite side', 'Contralateral opposite side'])

thermo_ic = pd.DataFrame([(thermo_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0) + (thermo_hit_hist_right[ipsi_indices_right]>threshold).sum(axis = 0), 
                            (thermo_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0) + (thermo_hit_hist_right[contra_indices_right]>threshold).sum(axis = 0),
                            (thermo_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0) + (thermo_hit_hist_right[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (thermo_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0) + (thermo_hit_hist_right[contra_indices_left]>threshold).sum(axis = 0)], 
                            index = ['Ipsilateral same side', 'Contralateral same side', 
                                    'Ipsilateral opposite side', 'Contralateral opposite side'])

photo_ic = pd.DataFrame([(photo_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0) + (photo_hit_hist_right[ipsi_indices_right]>threshold).sum(axis = 0), 
                            (photo_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0) + (photo_hit_hist_right[contra_indices_right]>threshold).sum(axis = 0),
                            (photo_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0) + (photo_hit_hist_right[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (photo_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0) + (photo_hit_hist_right[contra_indices_left]>threshold).sum(axis = 0)], 
                            index = ['Ipsilateral same side', 'Contralateral same side', 
                                    'Ipsilateral opposite side', 'Contralateral opposite side'])

ax = axs[0]
ax.set_title('ORN signal')
sns.heatmap(ORN_ic, ax = ax, rasterized = True, annot=True, fmt="d", cbar = False)

ax = axs[1]
ax.set_title('AN signal')
sns.heatmap(AN_ic, ax = ax, rasterized = True, annot=True, fmt="d", cbar = False)

ax = axs[2]
ax.set_title('MN signal')
sns.heatmap(MN_ic, ax = ax, rasterized = True, annot=True, fmt="d", cbar = False)

ax = axs[3]
ax.set_title('A00c signal')
sns.heatmap(A00c_ic, ax = ax, rasterized = True, annot=True, fmt="d", cbar = False)

#ax = axs[4]
#ax.set_title('vtd signal')
#sns.heatmap(vtd_ic, ax = ax, rasterized = True, annot=True, fmt="d")

ax = axs[4]
ax.set_title('thermo signal')
sns.heatmap(thermo_ic, ax = ax, rasterized = True, annot=True, fmt="d", cbar = False)

ax = axs[5]
ax.set_title('photo signal')
ax.set_xlabel('Hops from sensory signal')
sns.heatmap(photo_ic, ax = ax, rasterized = True, annot=True, fmt="d", cbar = False)

fig.savefig('cascades/interhemisphere_plots/num_ipsicontra_ds_each_sensory.pdf', format='pdf', bbox_inches='tight')


# %%
# which neurons are collecting information from both left/right hemispheres? for each modality? for all modalities?

def intersect_stats(hit_hist1, hit_hist2, threshold, hops):

    intersect_hops = []
    total_hops = []

    for i in np.arange(0, hops):
        intersect = np.logical_and(hit_hist1[:,i]>threshold, hit_hist2[:,i]>threshold)
        total = np.logical_or(hit_hist1[:,i]>threshold, hit_hist2[:,i]>threshold)
        intersect_hops.append(intersect)
        total_hops.append(total)

    percent = []
    for i in np.arange(0, hops):
        percent.append(sum(intersect_hops[i])/sum(total_hops[i]))

    return(np.array(intersect_hops), np.array(total_hops), percent)

threshold = 50
hops = 10

ORN_intersect, ORN_total, ORN_percent = intersect_stats(ORN_hit_hist_left, ORN_hit_hist_right, threshold, hops)
AN_intersect, AN_total, AN_percent = intersect_stats(AN_hit_hist_left, AN_hit_hist_right, threshold, hops)
MN_intersect, MN_total, MN_percent = intersect_stats(MN_hit_hist_left, MN_hit_hist_right, threshold, hops)
A00c_intersect, A00c_total, A00c_percent = intersect_stats(A00c_hit_hist_left, A00c_hit_hist_right, threshold, hops)
vtd_intersect, vtd_total, vtd_percent = intersect_stats(vtd_hit_hist_left, vtd_hit_hist_right, threshold, hops)
thermo_intersect, thermo_total, thermo_percent = intersect_stats(thermo_hit_hist_left, thermo_hit_hist_right, threshold, hops)
photo_intersect, photo_total, photo_percent = intersect_stats(photo_hit_hist_left, photo_hit_hist_right, threshold, hops)

all_inputs_intersect, all_inputs_total, all_inputs_percent = intersect_stats(all_inputs_hit_hist_left, all_inputs_hit_hist_right, threshold, hops)

#%%
# identify sites of convergence between left/right hemisphere
# correlation matrix per hop between sites of convergence between all sensory modalities

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import cmasher as cmr

fig, axs = plt.subplots(
    3, 1, figsize=(6, 10), sharey = True
)
threshold = 50
fig.tight_layout(pad=2.5)

# ORN signal
all_inputs_ic_left = pd.DataFrame([(all_inputs_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (all_inputs_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0),
                            (all_inputs_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0), 
                            (all_inputs_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0)], 
                            index = ['Ipsilateral left', 'Contralateral left', 
                                    'Contralateral right', 'Ipsilateral right'])

all_inputs_ic_right = pd.DataFrame([(all_inputs_hit_hist_right[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (all_inputs_hit_hist_right[contra_indices_left]>threshold).sum(axis = 0),
                            (all_inputs_hit_hist_right[contra_indices_right]>threshold).sum(axis = 0), 
                            (all_inputs_hit_hist_right[ipsi_indices_right]>threshold).sum(axis = 0)], 
                            index = ['Ipsilateral right', 'Contralateral right', 
                                    'Contralateral right', 'Ipsilateral right'])

all_inputs_intersect_matrix = pd.DataFrame([all_inputs_intersect.T[ipsi_indices_left].sum(axis=0)/all_inputs_total.T[ipsi_indices_left].sum(axis=0), 
                                    all_inputs_intersect.T[contra_indices_left].sum(axis=0)/all_inputs_total.T[contra_indices_left].sum(axis=0),
                                    all_inputs_intersect.T[contra_indices_right].sum(axis=0)/all_inputs_total.T[contra_indices_right].sum(axis=0),
                                    all_inputs_intersect.T[ipsi_indices_right].sum(axis=0)/all_inputs_total.T[ipsi_indices_right].sum(axis=0)], 
                            index = ['Ipsilateral left', 'Contralateral left', 
                                    'Contralateral right', 'Ipsilateral right'])

ax = axs[0]
ax.set_title('Number of Neurons downstream of left sensory signals')
sns.heatmap(all_inputs_ic_left.iloc[:, 0:5], ax = ax, rasterized = True, annot=True, fmt="d", cbar = False)

ax = axs[1]
ax.set_title('Number of Neurons downstream of right sensory signals')
sns.heatmap(all_inputs_ic_right.iloc[:, 0:5], ax = ax, rasterized = True, annot=True, fmt="d", cbar = False)

ax = axs[2]
ax.set_title('Fraction of neurons receiving both left and right signals')
sns.heatmap(all_inputs_intersect_matrix.iloc[:, 0:5], ax = ax, rasterized = True, annot=True, fmt="f", cbar = False, cmap = cmr.lavender)
ax.set_xlabel('Hops')

plt.savefig('cascades/interhemisphere_plots/simulated_all_left-right_signals.pdf', format='pdf', bbox_inches='tight')

# %%
# correlation between visit numbers between pairs and across nodes

correlate_nodes = []

for j in range(0, len(all_inputs_hit_hist_left[0, :])):
    for i in range(0, len(all_inputs_hit_hist_left)):
        correlate_nodes.append([mg.meta.index[i], j, 
                                all_inputs_hit_hist_left[i, j], all_inputs_hit_hist_right[i, j],
                                ORN_hit_hist_left[i, j], ORN_hit_hist_right[i, j],
                                AN_hit_hist_left[i, j], AN_hit_hist_right[i, j],
                                MN_hit_hist_left[i, j], MN_hit_hist_right[i, j],
                                A00c_hit_hist_left[i, j], A00c_hit_hist_right[i, j],
                                vtd_hit_hist_left[i, j], vtd_hit_hist_right[i, j],
                                thermo_hit_hist_left[i, j], thermo_hit_hist_right[i, j],
                                photo_hit_hist_left[i, j], photo_hit_hist_right[i, j]])

correlate_nodes = pd.DataFrame(correlate_nodes, columns = ['skid', 'hop', 
                                                            'all_left_visits', 'all_right_visits',
                                                            'ORN_left_visits', 'ORN_right_visits',
                                                            'AN_left_visits', 'AN_right_visits',
                                                            'MN_left_visits', 'MN_right_visits',
                                                            'A00c_left_visits', 'A00c_right_visits',
                                                            'vtd_left_visits', 'vtd_right_visits',
                                                            'thermo_left_visits', 'thermo_right_visits',
                                                            'photo_left_visits', 'photo_right_visits'])

threshold = 0
cor_index = (correlate_nodes['right_visits']>threshold) | (correlate_nodes['left_visits']>threshold)

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

p = sns.JointGrid(
    x = correlate_nodes['right_visits'][cor_index],
    y = correlate_nodes['left_visits'][cor_index]
    )

p = p.plot_joint(
    plt.hexbin, cmap = 'Blues', bins = 'log', gridsize = 40,
    )

p.set_axis_labels('Signal from Right', 'Signal from Left')

p.ax_marg_x.hist(
    correlate_nodes['right_visits'][((correlate_nodes['right_visits']>50) | (correlate_nodes['left_visits']>50)) & (correlate_nodes.hop!=0)],
    bins = 40,
    alpha = 0.5
    )

p.ax_marg_y.hist(
    correlate_nodes['left_visits'][((correlate_nodes['right_visits']>50) | (correlate_nodes['left_visits']>50)) & (correlate_nodes.hop!=0)],
    orientation = 'horizontal',
    bins = 40,
    alpha = 0.5,
    )

p.savefig('cascades/interhemisphere_plots/left_vs_right_visits_allsensory.pdf', format='pdf', bbox_inches='tight')

#sns.jointplot(x = correlate_nodes['right_visits'][cor_index], y = correlate_nodes['left_visits'][cor_index], 
#            kind = 'hex', joint_kws={'gridsize':40, 'bins':'log'})

# %%
# correlation between left and right visits per hop
# nice supplemental figure
hops = 0
p = sns.JointGrid(
    x = correlate_nodes['right_visits'][(cor_index) & (correlate_nodes.hop==hops)],
    y = correlate_nodes['left_visits'][(cor_index) & (correlate_nodes.hop==hops)],
    )

p = p.plot_joint(
    plt.hexbin, cmap = 'Blues', bins = 'log', gridsize = 40,
    )

p.set_axis_labels('Signal from Right', 'Signal from Left')

p.ax_marg_x.hist(
    correlate_nodes['right_visits'][((correlate_nodes['right_visits']>25) | (correlate_nodes['left_visits']>25)) & (correlate_nodes.hop==hops)],
    bins = 40,
    alpha = 0.5
    )

p.ax_marg_y.hist(
    correlate_nodes['left_visits'][((correlate_nodes['right_visits']>25) | (correlate_nodes['left_visits']>25)) & (correlate_nodes.hop==hops)],
    orientation = 'horizontal',
    bins = 40,
    alpha = 0.5,
    )

p.savefig('cascades/interhemisphere_plots/left_vs_right_visits_allsensory_hop0.pdf', format='pdf', bbox_inches='tight')


# hop 1
hops = 1
p = sns.JointGrid(
    x = correlate_nodes['right_visits'][(cor_index) & (correlate_nodes.hop==hops)],
    y = correlate_nodes['left_visits'][(cor_index) & (correlate_nodes.hop==hops)]
    )

p = p.plot_joint(
    plt.hexbin, cmap = 'Blues', bins = 'log', gridsize = 40,
    )

p.set_axis_labels('Signal from Right', 'Signal from Left')

p.ax_marg_x.hist(
    correlate_nodes['right_visits'][((correlate_nodes['right_visits']>25) | (correlate_nodes['left_visits']>25)) & (correlate_nodes.hop==hops)],
    bins = 40,
    alpha = 0.5
    )

p.ax_marg_y.hist(
    correlate_nodes['left_visits'][((correlate_nodes['right_visits']>25) | (correlate_nodes['left_visits']>25)) & (correlate_nodes.hop==hops)],
    orientation = 'horizontal',
    bins = 40,
    alpha = 0.5,
    )

p.savefig('cascades/interhemisphere_plots/left_vs_right_visits_allsensory_hop1.pdf', format='pdf', bbox_inches='tight')

# hop 2
ax = axs[2, 0]
hops = 2
p = sns.JointGrid(
    x = correlate_nodes['right_visits'][(cor_index) & (correlate_nodes.hop==hops)],
    y = correlate_nodes['left_visits'][(cor_index) & (correlate_nodes.hop==hops)]
    )

p = p.plot_joint(
    plt.hexbin, cmap = 'Blues', bins = 'log', gridsize = 40,
    )

p.set_axis_labels('Signal from Right', 'Signal from Left')

p.ax_marg_x.hist(
    correlate_nodes['right_visits'][((correlate_nodes['right_visits']>25) | (correlate_nodes['left_visits']>25)) & (correlate_nodes.hop==hops)],
    bins = 40,
    alpha = 0.5
    )

p.ax_marg_y.hist(
    correlate_nodes['left_visits'][((correlate_nodes['right_visits']>25) | (correlate_nodes['left_visits']>25)) & (correlate_nodes.hop==hops)],
    orientation = 'horizontal',
    bins = 40,
    alpha = 0.5,
    )

p.savefig('cascades/interhemisphere_plots/left_vs_right_visits_allsensory_hop2.pdf', format='pdf', bbox_inches='tight')

# hop 3
hops = 3
p = sns.JointGrid(
    x = correlate_nodes['right_visits'][(cor_index) & (correlate_nodes.hop==hops)],
    y = correlate_nodes['left_visits'][(cor_index) & (correlate_nodes.hop==hops)]
    )

p = p.plot_joint(
    plt.hexbin, cmap = 'Blues', bins = 'log', gridsize = 40,
    )

p.set_axis_labels('Signal from Right', 'Signal from Left')

p.ax_marg_x.hist(
    correlate_nodes['right_visits'][((correlate_nodes['right_visits']>25) | (correlate_nodes['left_visits']>25)) & (correlate_nodes.hop==hops)],
    bins = 40,
    alpha = 0.5
    )

p.ax_marg_y.hist(
    correlate_nodes['left_visits'][((correlate_nodes['right_visits']>25) | (correlate_nodes['left_visits']>25)) & (correlate_nodes.hop==hops)],
    orientation = 'horizontal',
    bins = 40,
    alpha = 0.5,
    )

p.savefig('cascades/interhemisphere_plots/left_vs_right_visits_allsensory_hop3.pdf', format='pdf', bbox_inches='tight')

# hop 4
hops = 4
p = sns.JointGrid(
    x = correlate_nodes['right_visits'][(cor_index) & (correlate_nodes.hop==hops)],
    y = correlate_nodes['left_visits'][(cor_index) & (correlate_nodes.hop==hops)]
    )

p = p.plot_joint(
    plt.hexbin, cmap = 'Blues', bins = 'log', gridsize = 40,
    )

p.set_axis_labels('Signal from Right', 'Signal from Left')

p.ax_marg_x.hist(
    correlate_nodes['right_visits'][((correlate_nodes['right_visits']>25) | (correlate_nodes['left_visits']>25)) & (correlate_nodes.hop==hops)],
    bins = 40,
    alpha = 0.5
    )

p.ax_marg_y.hist(
    correlate_nodes['left_visits'][((correlate_nodes['right_visits']>25) | (correlate_nodes['left_visits']>25)) & (correlate_nodes.hop==hops)],
    orientation = 'horizontal',
    bins = 40,
    alpha = 0.5,
    )

p.savefig('cascades/interhemisphere_plots/left_vs_right_visits_allsensory_hop4.pdf', format='pdf', bbox_inches='tight')

# hop 5
hops = 5
p = sns.JointGrid(
    x = correlate_nodes['right_visits'][(cor_index) & (correlate_nodes.hop==hops)],
    y = correlate_nodes['left_visits'][(cor_index) & (correlate_nodes.hop==hops)]
    )

p = p.plot_joint(
    plt.hexbin, cmap = 'Blues', bins = 'log', gridsize = 40,
    )

p.set_axis_labels('Signal from Right', 'Signal from Left')

p.ax_marg_x.hist(
    correlate_nodes['right_visits'][((correlate_nodes['right_visits']>25) | (correlate_nodes['left_visits']>25)) & (correlate_nodes.hop==hops)],
    bins = 40,
    alpha = 0.5
    )

p.ax_marg_y.hist(
    correlate_nodes['left_visits'][((correlate_nodes['right_visits']>25) | (correlate_nodes['left_visits']>25)) & (correlate_nodes.hop==hops)],
    orientation = 'horizontal',
    bins = 40,
    alpha = 0.5,
    )

p.savefig('cascades/interhemisphere_plots/left_vs_right_visits_allsensory_hop5.pdf', format='pdf', bbox_inches='tight')

# %%
# identify group of integrator neurons that receive from both left and right signals
# compare between all sensory and individual sensory modalities

integrators = correlate_nodes[(correlate_nodes.all_right_visits>50) & (correlate_nodes.all_left_visits>50)]

# how many for each hop?
sum_per_hop = []
for i in range(0, max(integrators.hop)):
    sum_per_hop.append(sum(integrators.hop==i))

fig, axs = plt.subplots(
    1, 1, figsize=(6, 6)
)

ax = axs
ax.set_xlabel = 'Hops from Sensory'
ax.set_ylabel = 'Number of Integrator Neurons'

sns.barplot(x = list(range(0, max(integrators.hop))), y = sum_per_hop, color = 'blue', alpha = 0.5, ax = ax)

fig.savefig('cascades/interhemisphere_plots/num_integrators_per_hop.pdf', format='pdf', bbox_inches='tight')

# identify integrators for each modality
def membership(list1, list2):
    set1 = set(list1)
    return [item in set1 for item in list2]

integrators_ORN = correlate_nodes[(correlate_nodes.ORN_right_visits>50) & (correlate_nodes.ORN_left_visits>50)]
integrators_AN = correlate_nodes[(correlate_nodes.AN_right_visits>50) & (correlate_nodes.AN_left_visits>50)]
integrators_MN = correlate_nodes[(correlate_nodes.MN_right_visits>50) & (correlate_nodes.MN_left_visits>50)]
integrators_A00c = correlate_nodes[(correlate_nodes.A00c_right_visits>50) & (correlate_nodes.A00c_left_visits>50)]
integrators_vtd = correlate_nodes[(correlate_nodes.vtd_right_visits>50) & (correlate_nodes.vtd_left_visits>50)]
integrators_thermo = correlate_nodes[(correlate_nodes.thermo_right_visits>50) & (correlate_nodes.thermo_left_visits>50)]
integrators_photo = correlate_nodes[(correlate_nodes.photo_right_visits>50) & (correlate_nodes.photo_left_visits>50)]

integrator_list = [integrators, integrators_ORN, integrators_AN, integrators_MN, integrators_A00c, integrators_thermo, integrators_photo]

fraction_same = np.zeros([len(integrator_list), len(integrator_list)])
for i in range(0, len(integrator_list)):
    for j in range(0, len(integrator_list)):
        ij_mem = membership(integrator_list[i].skid, integrator_list[j].skid)
        if((len(integrator_list[i].skid) + len(integrator_list[j].skid) - sum(ij_mem))>0):
            ij_mem = sum(ij_mem)/(len(integrator_list[i].skid) + len(integrator_list[j].skid) - sum(ij_mem))
            fraction_same[i, j] = ij_mem

fraction_same = pd.DataFrame(fraction_same, columns = ['all inputs', 'ORN', 'AN', 'MN', 'A00c', 'thermo', 'photo'], 
                                            index = ['all inputs', 'ORN', 'AN', 'MN', 'A00c', 'thermo', 'photo'])

import cmasher as cmr

fig, axs = plt.subplots(
    1, 1, figsize=(4, 4)
)

ax = axs
sns.heatmap(fraction_same.iloc[1:len(fraction_same), 1:len(fraction_same)], ax = ax, cmap = cmr.amber, 
            cbar_kws = dict(use_gridspec=False,location="top", 
                        label = 'Similarity between Integration Centers'),
            square = True)

fig.savefig('cascades/interhemisphere_plots/similarity_between_integration_centers.pdf', format='pdf', bbox_inches='tight', rasterized = True)

# %%
# number of ipsi or contra neurons visited per hop
# shows nicely the flow of information through two hemispheres
# probably full supplemental figure

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import cmasher as cmr

fig, axs = plt.subplots(
    9, 2, figsize=(12, 20), sharey = True
)
threshold = 50
fig.tight_layout(pad=2.5)

# ORN signal

ORN_ic_left = pd.DataFrame([(ORN_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (ORN_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0),
                            (ORN_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0), 
                            (ORN_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0)], 
                            index = ['Ipsilateral left', 'Contralateral left', 
                                    'Contralateral right', 'Ipsilateral right'])

ORN_ic_right = pd.DataFrame([(ORN_hit_hist_right[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (ORN_hit_hist_right[contra_indices_left]>threshold).sum(axis = 0),
                            (ORN_hit_hist_right[contra_indices_right]>threshold).sum(axis = 0), 
                            (ORN_hit_hist_right[ipsi_indices_right]>threshold).sum(axis = 0)], 
                            index = ['Ipsilateral left', 'Contralateral left', 
                                    'Contralateral right', 'Ipsilateral right'])

ORN_intersect_matrix = pd.DataFrame([ORN_intersect.T[ipsi_indices_left].sum(axis=0)/ORN_total.T[ipsi_indices_left].sum(axis=0), 
                                    ORN_intersect.T[contra_indices_left].sum(axis=0)/ORN_total.T[contra_indices_left].sum(axis=0),
                                    ORN_intersect.T[contra_indices_right].sum(axis=0)/ORN_total.T[contra_indices_right].sum(axis=0),
                                    ORN_intersect.T[ipsi_indices_right].sum(axis=0)/ORN_total.T[ipsi_indices_right].sum(axis=0)], 
                            index = ['Ipsilateral left', 'Contralateral left', 
                                    'Contralateral right', 'Ipsilateral right'])

ORN_intersect_matrix = ORN_intersect_matrix.fillna(0)

# AN signal
AN_ic_left = pd.DataFrame([(AN_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (AN_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0),
                            (AN_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0), 
                            (AN_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0)], 
                            index = ['Ipsilateral left', 'Contralateral left', 
                                    'Contralateral right', 'Ipsilateral right'])

AN_ic_right = pd.DataFrame([(AN_hit_hist_right[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (AN_hit_hist_right[contra_indices_left]>threshold).sum(axis = 0),
                            (AN_hit_hist_right[contra_indices_right]>threshold).sum(axis = 0), 
                            (AN_hit_hist_right[ipsi_indices_right]>threshold).sum(axis = 0)], 
                            index = ['Ipsilateral left', 'Contralateral left', 
                                    'Contralateral right', 'Ipsilateral right'])

AN_intersect_matrix = pd.DataFrame([AN_intersect.T[ipsi_indices_left].sum(axis=0)/AN_total.T[ipsi_indices_left].sum(axis=0), 
                                    AN_intersect.T[contra_indices_left].sum(axis=0)/AN_total.T[contra_indices_left].sum(axis=0),
                                    AN_intersect.T[contra_indices_right].sum(axis=0)/AN_total.T[contra_indices_right].sum(axis=0),
                                    AN_intersect.T[ipsi_indices_right].sum(axis=0)/AN_total.T[ipsi_indices_right].sum(axis=0)], 
                            index = ['Ipsilateral left', 'Contralateral left', 
                                    'Contralateral right', 'Ipsilateral right'])

AN_intersect_matrix = AN_intersect_matrix.fillna(0)


# MN signal
MN_ic_left = pd.DataFrame([(MN_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (MN_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0),
                            (MN_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0), 
                            (MN_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0)], 
                            index = ['Ipsilateral left', 'Contralateral left', 
                                    'Contralateral right', 'Ipsilateral right'])

MN_ic_right = pd.DataFrame([(MN_hit_hist_right[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (MN_hit_hist_right[contra_indices_left]>threshold).sum(axis = 0),
                            (MN_hit_hist_right[contra_indices_right]>threshold).sum(axis = 0), 
                            (MN_hit_hist_right[ipsi_indices_right]>threshold).sum(axis = 0)], 
                            index = ['Ipsilateral left', 'Contralateral left', 
                                    'Contralateral right', 'Ipsilateral right'])

MN_intersect_matrix = pd.DataFrame([MN_intersect.T[ipsi_indices_left].sum(axis=0)/MN_total.T[ipsi_indices_left].sum(axis=0), 
                                    MN_intersect.T[contra_indices_left].sum(axis=0)/MN_total.T[contra_indices_left].sum(axis=0),
                                    MN_intersect.T[contra_indices_right].sum(axis=0)/MN_total.T[contra_indices_right].sum(axis=0),
                                    MN_intersect.T[ipsi_indices_right].sum(axis=0)/MN_total.T[ipsi_indices_right].sum(axis=0)], 
                            index = ['Ipsilateral left', 'Contralateral left', 
                                    'Contralateral right', 'Ipsilateral right'])

MN_intersect_matrix = MN_intersect_matrix.fillna(0)

# A00c signal
A00c_ic_left = pd.DataFrame([(A00c_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (A00c_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0),
                            (A00c_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0), 
                            (A00c_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0)], 
                            index = ['Ipsilateral left', 'Contralateral left', 
                                    'Contralateral right', 'Ipsilateral right'])

A00c_ic_right = pd.DataFrame([(A00c_hit_hist_right[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (A00c_hit_hist_right[contra_indices_left]>threshold).sum(axis = 0),
                            (A00c_hit_hist_right[contra_indices_right]>threshold).sum(axis = 0), 
                            (A00c_hit_hist_right[ipsi_indices_right]>threshold).sum(axis = 0)], 
                            index = ['Ipsilateral left', 'Contralateral left', 
                                    'Contralateral right', 'Ipsilateral right'])

A00c_intersect_matrix = pd.DataFrame([A00c_intersect.T[ipsi_indices_left].sum(axis=0)/A00c_total.T[ipsi_indices_left].sum(axis=0), 
                                    A00c_intersect.T[contra_indices_left].sum(axis=0)/A00c_total.T[contra_indices_left].sum(axis=0),
                                    A00c_intersect.T[contra_indices_right].sum(axis=0)/A00c_total.T[contra_indices_right].sum(axis=0),
                                    A00c_intersect.T[ipsi_indices_right].sum(axis=0)/A00c_total.T[ipsi_indices_right].sum(axis=0)], 
                            index = ['Ipsilateral left', 'Contralateral left', 
                                    'Contralateral right', 'Ipsilateral right'])

A00c_intersect_matrix = A00c_intersect_matrix.fillna(0)

# thermo signal
thermo_ic_left = pd.DataFrame([(thermo_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (thermo_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0),
                            (thermo_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0), 
                            (thermo_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0)], 
                            index = ['Ipsilateral left', 'Contralateral left', 
                                    'Contralateral right', 'Ipsilateral right'])

thermo_ic_right = pd.DataFrame([(thermo_hit_hist_right[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (thermo_hit_hist_right[contra_indices_left]>threshold).sum(axis = 0),
                            (thermo_hit_hist_right[contra_indices_right]>threshold).sum(axis = 0), 
                            (thermo_hit_hist_right[ipsi_indices_right]>threshold).sum(axis = 0)], 
                            index = ['Ipsilateral left', 'Contralateral left', 
                                    'Contralateral right', 'Ipsilateral right'])

thermo_intersect_matrix = pd.DataFrame([thermo_intersect.T[ipsi_indices_left].sum(axis=0)/thermo_total.T[ipsi_indices_left].sum(axis=0), 
                                    thermo_intersect.T[contra_indices_left].sum(axis=0)/thermo_total.T[contra_indices_left].sum(axis=0),
                                    thermo_intersect.T[contra_indices_right].sum(axis=0)/thermo_total.T[contra_indices_right].sum(axis=0),
                                    thermo_intersect.T[ipsi_indices_right].sum(axis=0)/thermo_total.T[ipsi_indices_right].sum(axis=0)], 
                            index = ['Ipsilateral left', 'Contralateral left', 
                                    'Contralateral right', 'Ipsilateral right'])

thermo_intersect_matrix = thermo_intersect_matrix.fillna(0)


# photo signal
photo_ic_left = pd.DataFrame([(photo_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (photo_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0),
                            (photo_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0), 
                            (photo_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0)], 
                            index = ['Ipsilateral left', 'Contralateral left', 
                                    'Contralateral right', 'Ipsilateral right'])

photo_ic_right = pd.DataFrame([(photo_hit_hist_right[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (photo_hit_hist_right[contra_indices_left]>threshold).sum(axis = 0),
                            (photo_hit_hist_right[contra_indices_right]>threshold).sum(axis = 0), 
                            (photo_hit_hist_right[ipsi_indices_right]>threshold).sum(axis = 0)], 
                            index = ['Ipsilateral left', 'Contralateral left', 
                                    'Contralateral right', 'Ipsilateral right'])

photo_intersect_matrix = pd.DataFrame([photo_intersect.T[ipsi_indices_left].sum(axis=0)/photo_total.T[ipsi_indices_left].sum(axis=0), 
                                    photo_intersect.T[contra_indices_left].sum(axis=0)/photo_total.T[contra_indices_left].sum(axis=0),
                                    photo_intersect.T[contra_indices_right].sum(axis=0)/photo_total.T[contra_indices_right].sum(axis=0),
                                    photo_intersect.T[ipsi_indices_right].sum(axis=0)/photo_total.T[ipsi_indices_right].sum(axis=0)], 
                            index = ['Ipsilateral left', 'Contralateral left', 
                                    'Contralateral right', 'Ipsilateral right'])

photo_intersect_matrix = photo_intersect_matrix.fillna(0)

# ORN
ax = axs[0, 0]
ax.get_xaxis().set_visible(False)
ax.set_title('Number of Neurons downstream of ORN left signal')
sns.heatmap(ORN_ic_left.iloc[:, 0:6], ax = ax, rasterized = True, annot=True, fmt="d", cbar = False)

ax = axs[1, 0]
ax.get_xaxis().set_visible(False)
ax.set_title('Number of Neurons downstream of ORN right signal')
sns.heatmap(ORN_ic_right.iloc[:, 0:6], ax = ax, rasterized = True, annot=True, fmt="d", cbar = False)

ax = axs[2, 0]
ax.set_title('Fraction of neurons receiving left and right signals')
sns.heatmap(ORN_intersect_matrix.iloc[:, 0:6], ax = ax, rasterized = True, annot=True, fmt="f", cbar = False, cmap = cmr.lavender)
ax.set_xlabel('Hops')

# AN
ax = axs[3, 0]
ax.get_xaxis().set_visible(False)
ax.set_title('Number of Neurons downstream of AN left signal')
sns.heatmap(AN_ic_left.iloc[:, 0:6], ax = ax, rasterized = True, annot=True, fmt="d", cbar = False)

ax = axs[4, 0]
ax.get_xaxis().set_visible(False)
ax.set_title('Number of Neurons downstream of AN right signal')
sns.heatmap(AN_ic_right.iloc[:, 0:6], ax = ax, rasterized = True, annot=True, fmt="d", cbar = False)

ax = axs[5, 0]
ax.set_title('Fraction of neurons receiving left and right signals')
sns.heatmap(AN_intersect_matrix.iloc[:, 0:6], ax = ax, rasterized = True, annot=True, fmt="f", cbar = False, cmap = cmr.lavender)
ax.set_xlabel('Hops')

# MN
ax = axs[6, 0]
ax.get_xaxis().set_visible(False)
ax.set_title('Number of Neurons downstream of MN left signal')
sns.heatmap(MN_ic_left.iloc[:, 0:6], ax = ax, rasterized = True, annot=True, fmt="d", cbar = False)

ax = axs[7, 0]
ax.get_xaxis().set_visible(False)
ax.set_title('Number of Neurons downstream of MN right signal')
sns.heatmap(MN_ic_right.iloc[:, 0:6], ax = ax, rasterized = True, annot=True, fmt="d", cbar = False)

ax = axs[8, 0]
ax.set_title('Fraction of neurons receiving left and right signals')
sns.heatmap(MN_intersect_matrix.iloc[:, 0:6], ax = ax, rasterized = True, annot=True, fmt="f", cbar = False, cmap = cmr.lavender)
ax.set_xlabel('Hops')

# A00c
ax = axs[0, 1]
ax.get_xaxis().set_visible(False)
ax.set_title('Number of Neurons downstream of A00c left signal')
sns.heatmap(A00c_ic_left.iloc[:, 0:6], ax = ax, rasterized = True, annot=True, fmt="d", cbar = False)

ax = axs[1, 1]
ax.get_xaxis().set_visible(False)
ax.set_title('Number of Neurons downstream of A00c right signal')
sns.heatmap(A00c_ic_right.iloc[:, 0:6], ax = ax, rasterized = True, annot=True, fmt="d", cbar = False)

ax = axs[2, 1]
ax.set_title('Fraction of neurons receiving left and right signals')
sns.heatmap(A00c_intersect_matrix.iloc[:, 0:6], ax = ax, rasterized = True, annot=True, fmt="f", cbar = False, cmap = cmr.lavender)
ax.set_xlabel('Hops')

# thermo
ax = axs[3, 1]
ax.get_xaxis().set_visible(False)
ax.set_title('Number of Neurons downstream of thermo left signal')
sns.heatmap(thermo_ic_left.iloc[:, 0:6], ax = ax, rasterized = True, annot=True, fmt="d", cbar = False)

ax = axs[4, 1]
ax.get_xaxis().set_visible(False)
ax.set_title('Number of Neurons downstream of thermo right signal')
sns.heatmap(thermo_ic_right.iloc[:, 0:6], ax = ax, rasterized = True, annot=True, fmt="d", cbar = False)

ax = axs[5, 1]
ax.set_title('Fraction of neurons receiving left and right signals')
sns.heatmap(thermo_intersect_matrix.iloc[:, 0:6], ax = ax, rasterized = True, annot=True, fmt="f", cbar = False, cmap = cmr.lavender)
ax.set_xlabel('Hops')

# photo
ax = axs[6, 1]
ax.get_xaxis().set_visible(False)
ax.set_title('Number of Neurons downstream of photo left signal')
sns.heatmap(photo_ic_left.iloc[:, 0:6], ax = ax, rasterized = True, annot=True, fmt="d", cbar = False)

ax = axs[7, 1]
ax.get_xaxis().set_visible(False)
ax.set_title('Number of Neurons downstream of photo right signal')
sns.heatmap(photo_ic_right.iloc[:, 0:6], ax = ax, rasterized = True, annot=True, fmt="d", cbar = False)

ax = axs[8, 1]
ax.set_title('Fraction of neurons receiving left and right signals')
sns.heatmap(photo_intersect_matrix.iloc[:, 0:6], ax = ax, rasterized = True, annot=True, fmt="f", cbar = False, cmap = cmr.lavender)
ax.set_xlabel('Hops')

plt.savefig('cascades/interhemisphere_plots/simulated_left-right_signals.pdf', format='pdf', bbox_inches='tight')

# %%
# signal to left and right descending
# not super obvious differences when all descending are lumped together
# signal clearly goes to both sides of the brain equally

#sns.heatmap([ORN_hit_hist_left[dVNC_indices_left].sum(axis = 0), ORN_hit_hist_left[dVNC_indices_right].sum(axis = 0)])
#sns.heatmap([(ORN_hit_hist_left[dVNC_indices_left]>50).sum(axis = 0), (ORN_hit_hist_left[dVNC_indices_right]>50).sum(axis = 0)])
fig, axs = plt.subplots(
    1, 1, figsize=(8, 8)
)
output_dVNC = pd.DataFrame([ORN_hit_hist_left[dVNC_indices_left].sum(axis = 0), 
                            ORN_hit_hist_left[dVNC_indices_right].sum(axis = 0),
                            ORN_hit_hist_right[dVNC_indices_left].sum(axis = 0), 
                            ORN_hit_hist_right[dVNC_indices_right].sum(axis = 0),
                            AN_hit_hist_left[dVNC_indices_left].sum(axis = 0), 
                            AN_hit_hist_left[dVNC_indices_right].sum(axis = 0),
                            AN_hit_hist_right[dVNC_indices_left].sum(axis = 0), 
                            AN_hit_hist_right[dVNC_indices_right].sum(axis = 0),
                            MN_hit_hist_left[dVNC_indices_left].sum(axis = 0), 
                            MN_hit_hist_left[dVNC_indices_right].sum(axis = 0),
                            MN_hit_hist_right[dVNC_indices_left].sum(axis = 0), 
                            MN_hit_hist_right[dVNC_indices_right].sum(axis = 0),
                            A00c_hit_hist_left[dVNC_indices_left].sum(axis = 0), 
                            A00c_hit_hist_left[dVNC_indices_right].sum(axis = 0),
                            A00c_hit_hist_right[dVNC_indices_left].sum(axis = 0), 
                            A00c_hit_hist_right[dVNC_indices_right].sum(axis = 0),
                            thermo_hit_hist_left[dVNC_indices_left].sum(axis = 0), 
                            thermo_hit_hist_left[dVNC_indices_right].sum(axis = 0),
                            thermo_hit_hist_right[dVNC_indices_left].sum(axis = 0), 
                            thermo_hit_hist_right[dVNC_indices_right].sum(axis = 0),
                            photo_hit_hist_left[dVNC_indices_left].sum(axis = 0), 
                            photo_hit_hist_left[dVNC_indices_right].sum(axis = 0),
                            photo_hit_hist_right[dVNC_indices_left].sum(axis = 0), 
                            photo_hit_hist_right[dVNC_indices_right].sum(axis = 0)], 
                            index = ['ORN_left -> dVNC_left', 'ORN_left -> dVNC_right', 
                                    'ORN_right -> dVNC_left', 'ORN_right -> dVNC_right',
                                    'AN_left -> dVNC_left', 'AN_left -> dVNC_right', 
                                    'AN_right -> dVNC_left', 'AN_right -> dVNC_right',
                                    'MN_left -> dVNC_left', 'MN_left -> dVNC_right', 
                                    'MN_right -> dVNC_left', 'MN_right -> dVNC_right',
                                    'A00c_left -> dVNC_left', 'A00c_left -> dVNC_right', 
                                    'A00c_right -> dVNC_left', 'A00c_right -> dVNC_right',
                                    'thermo_left -> dVNC_left', 'thermo_left -> dVNC_right', 
                                    'thermo_right -> dVNC_left', 'thermo_right -> dVNC_right',
                                    'photo_left -> dVNC_left', 'photo_left -> dVNC_right', 
                                    'photo_right -> dVNC_left', 'photo_right -> dVNC_right'])

sns.heatmap(output_dVNC, ax = axs, rasterized = True)

# %%
fig, axs = plt.subplots(
    1, 1, figsize=(8, 8)
)
output_dSEZ = pd.DataFrame([ORN_hit_hist_left[dSEZ_indices_left].sum(axis = 0), 
                            ORN_hit_hist_left[dSEZ_indices_right].sum(axis = 0),
                            ORN_hit_hist_right[dSEZ_indices_left].sum(axis = 0), 
                            ORN_hit_hist_right[dSEZ_indices_right].sum(axis = 0),
                            AN_hit_hist_left[dSEZ_indices_left].sum(axis = 0), 
                            AN_hit_hist_left[dSEZ_indices_right].sum(axis = 0),
                            AN_hit_hist_right[dSEZ_indices_left].sum(axis = 0), 
                            AN_hit_hist_right[dSEZ_indices_right].sum(axis = 0),
                            MN_hit_hist_left[dSEZ_indices_left].sum(axis = 0), 
                            MN_hit_hist_left[dSEZ_indices_right].sum(axis = 0),
                            MN_hit_hist_right[dSEZ_indices_left].sum(axis = 0), 
                            MN_hit_hist_right[dSEZ_indices_right].sum(axis = 0),
                            A00c_hit_hist_left[dSEZ_indices_left].sum(axis = 0), 
                            A00c_hit_hist_left[dSEZ_indices_right].sum(axis = 0),
                            A00c_hit_hist_right[dSEZ_indices_left].sum(axis = 0), 
                            A00c_hit_hist_right[dSEZ_indices_right].sum(axis = 0),
                            thermo_hit_hist_left[dSEZ_indices_left].sum(axis = 0), 
                            thermo_hit_hist_left[dSEZ_indices_right].sum(axis = 0),
                            thermo_hit_hist_right[dSEZ_indices_left].sum(axis = 0), 
                            thermo_hit_hist_right[dSEZ_indices_right].sum(axis = 0),
                            photo_hit_hist_left[dSEZ_indices_left].sum(axis = 0), 
                            photo_hit_hist_left[dSEZ_indices_right].sum(axis = 0),
                            photo_hit_hist_right[dSEZ_indices_left].sum(axis = 0), 
                            photo_hit_hist_right[dSEZ_indices_right].sum(axis = 0)], 
                            index = ['ORN_left -> dSEZ_left', 'ORN_left -> dSEZ_right', 
                                    'ORN_right -> dSEZ_left', 'ORN_right -> dSEZ_right',
                                    'AN_left -> dSEZ_left', 'AN_left -> dSEZ_right', 
                                    'AN_right -> dSEZ_left', 'AN_right -> dSEZ_right',
                                    'MN_left -> dSEZ_left', 'MN_left -> dSEZ_right', 
                                    'MN_right -> dSEZ_left', 'MN_right -> dSEZ_right',
                                    'A00c_left -> dSEZ_left', 'A00c_left -> dSEZ_right', 
                                    'A00c_right -> dSEZ_left', 'A00c_right -> dSEZ_right',
                                    'thermo_left -> dSEZ_left', 'thermo_left -> dSEZ_right', 
                                    'thermo_right -> dSEZ_left', 'thermo_right -> dSEZ_right',
                                    'photo_left -> dSEZ_left', 'photo_left -> dSEZ_right', 
                                    'photo_right -> dSEZ_left', 'photo_right -> dSEZ_right'])

sns.heatmap(output_dSEZ, ax = axs, rasterized = True)

# %%
fig, axs = plt.subplots(
    1, 1, figsize=(8, 8)
)
output_RG = pd.DataFrame([ORN_hit_hist_left[RG_indices_left].sum(axis = 0), 
                            ORN_hit_hist_left[RG_indices_right].sum(axis = 0),
                            ORN_hit_hist_right[RG_indices_left].sum(axis = 0), 
                            ORN_hit_hist_right[RG_indices_right].sum(axis = 0),
                            AN_hit_hist_left[RG_indices_left].sum(axis = 0), 
                            AN_hit_hist_left[RG_indices_right].sum(axis = 0),
                            AN_hit_hist_right[RG_indices_left].sum(axis = 0), 
                            AN_hit_hist_right[RG_indices_right].sum(axis = 0),
                            MN_hit_hist_left[RG_indices_left].sum(axis = 0), 
                            MN_hit_hist_left[RG_indices_right].sum(axis = 0),
                            MN_hit_hist_right[RG_indices_left].sum(axis = 0), 
                            MN_hit_hist_right[RG_indices_right].sum(axis = 0),
                            A00c_hit_hist_left[RG_indices_left].sum(axis = 0), 
                            A00c_hit_hist_left[RG_indices_right].sum(axis = 0),
                            A00c_hit_hist_right[RG_indices_left].sum(axis = 0), 
                            A00c_hit_hist_right[RG_indices_right].sum(axis = 0),
                            thermo_hit_hist_left[RG_indices_left].sum(axis = 0), 
                            thermo_hit_hist_left[RG_indices_right].sum(axis = 0),
                            thermo_hit_hist_right[RG_indices_left].sum(axis = 0), 
                            thermo_hit_hist_right[RG_indices_right].sum(axis = 0),
                            photo_hit_hist_left[RG_indices_left].sum(axis = 0), 
                            photo_hit_hist_left[RG_indices_right].sum(axis = 0),
                            photo_hit_hist_right[RG_indices_left].sum(axis = 0), 
                            photo_hit_hist_right[RG_indices_right].sum(axis = 0)], 
                            index = ['ORN_left -> RG_left', 'ORN_left -> RG_right', 
                                    'ORN_right -> RG_left', 'ORN_right -> RG_right',
                                    'AN_left -> RG_left', 'AN_left -> RG_right', 
                                    'AN_right -> RG_left', 'AN_right -> RG_right',
                                    'MN_left -> RG_left', 'MN_left -> RG_right', 
                                    'MN_right -> RG_left', 'MN_right -> RG_right',
                                    'A00c_left -> RG_left', 'A00c_left -> RG_right', 
                                    'A00c_right -> RG_left', 'A00c_right -> RG_right',
                                    'thermo_left -> RG_left', 'thermo_left -> RG_right', 
                                    'thermo_right -> RG_left', 'thermo_right -> RG_right',
                                    'photo_left -> RG_left', 'photo_left -> RG_right', 
                                    'photo_right -> RG_left', 'photo_right -> RG_right'])

sns.heatmap(output_RG, ax = axs, rasterized = True)

# %%
#
x_axis_labels = [0,1,2,3,4,5,6]
x_label = 'Hops from Sensory'

fig, axs = plt.subplots(
    4, 2, figsize=(5, 8)
)

fig.tight_layout(pad=2.5)
#cbar_ax = axs.add_axes([3, 7, .1, .75])

ax = axs[0, 0] 
ax.set_title('Signal from ORN')
ax.set(xticks=[0, 1, 2, 3, 4, 5])
sns.heatmap(ORN_profile.T.iloc[:,0:7], ax = ax, cbar=False, xticklabels = x_axis_labels, rasterized=True)

ax = axs[1, 0]
ax.set_title('Signal from AN')
sns.heatmap(AN_profile.T.iloc[:,0:7], ax = ax, cbar=False, xticklabels = x_axis_labels, rasterized=True)

ax = axs[2, 0]
ax.set_title('Signal from MN')
sns.heatmap(MN_profile.T.iloc[:,0:7], ax = ax, cbar=False, xticklabels = x_axis_labels, rasterized=True)

ax = axs[3, 0]
ax.set_title('Signal from A00c')
sns.heatmap(A00c_profile.T.iloc[:,0:7], ax = ax, cbar=False, xticklabels = x_axis_labels, rasterized=True)
ax.set_xlabel(x_label)

ax = axs[0, 1]
ax.set_title('Signal from vtd')
sns.heatmap(vtd_profile.T.iloc[:,0:7], ax = ax, cbar=False, xticklabels = x_axis_labels, rasterized=True)

ax = axs[1, 1]
ax.set_title('Signal from thermo')
sns.heatmap(thermo_profile.T.iloc[:,0:7], ax = ax, cbar=False, xticklabels = x_axis_labels, rasterized=True)

ax = axs[2, 1]
ax.set_title('Signal from photo')
sns.heatmap(photo_profile.T.iloc[:,0:7], ax = ax, cbar_ax = axs[3, 1], xticklabels = x_axis_labels, rasterized=True)
ax.set_xlabel(x_label)

ax = axs[3, 1]
ax.set_xlabel('Number of Visits\nfrom Sensory Signal')
#ax.axis("off")

plt.savefig('cascades/plots/sensory_integration_per_hop.pdf', format='pdf', bbox_inches='tight')

# %%
# visits in ipsi or contra neurons per hop
# initial exploratory chunk
fig, axs = plt.subplots(
    1, 1, figsize=(8, 8)
)
ipsi_contra_flow = pd.DataFrame([ORN_hit_hist_left[ipsi_indices_left].sum(axis = 0), 
                            ORN_hit_hist_left[contra_indices_left].sum(axis = 0),
                            ORN_hit_hist_left[ipsi_indices_right].sum(axis = 0), 
                            ORN_hit_hist_left[contra_indices_right].sum(axis = 0),
                            AN_hit_hist_left[ipsi_indices_left].sum(axis = 0), 
                            AN_hit_hist_left[contra_indices_left].sum(axis = 0),
                            AN_hit_hist_left[ipsi_indices_right].sum(axis = 0), 
                            AN_hit_hist_left[contra_indices_right].sum(axis = 0),
                            MN_hit_hist_left[ipsi_indices_left].sum(axis = 0), 
                            MN_hit_hist_left[contra_indices_left].sum(axis = 0),
                            MN_hit_hist_left[ipsi_indices_right].sum(axis = 0), 
                            MN_hit_hist_left[contra_indices_right].sum(axis = 0),
                            A00c_hit_hist_left[ipsi_indices_left].sum(axis = 0), 
                            A00c_hit_hist_left[contra_indices_left].sum(axis = 0),
                            A00c_hit_hist_left[ipsi_indices_right].sum(axis = 0), 
                            A00c_hit_hist_left[contra_indices_right].sum(axis = 0),
                            vtd_hit_hist_left[ipsi_indices_left].sum(axis = 0), 
                            vtd_hit_hist_left[contra_indices_left].sum(axis = 0),
                            vtd_hit_hist_left[ipsi_indices_right].sum(axis = 0), 
                            vtd_hit_hist_left[contra_indices_right].sum(axis = 0),
                            thermo_hit_hist_left[ipsi_indices_left].sum(axis = 0), 
                            thermo_hit_hist_left[contra_indices_left].sum(axis = 0),
                            thermo_hit_hist_left[ipsi_indices_right].sum(axis = 0), 
                            thermo_hit_hist_left[contra_indices_right].sum(axis = 0),
                            photo_hit_hist_left[ipsi_indices_left].sum(axis = 0), 
                            photo_hit_hist_left[contra_indices_left].sum(axis = 0),
                            photo_hit_hist_left[ipsi_indices_right].sum(axis = 0), 
                            photo_hit_hist_left[contra_indices_right].sum(axis = 0)], 
                            index = ['ORN_left -> ipsi_indices_left', 'ORN_left -> contra_indices_left', 
                                    'ORN_left -> ipsi_indices_right', 'ORN_left -> contra_indices_right',
                                    'AN_left -> ipsi_indices_left', 'AN_left -> contra_indices_left', 
                                    'AN_left -> ipsi_indices_right', 'AN_left -> contra_indices_right',
                                    'MN_left -> ipsi_indices_left', 'MN_left -> contra_indices_left', 
                                    'MN_left -> ipsi_indices_right', 'MN_left -> contra_indices_right',
                                    'A00c_left -> ipsi_indices_left', 'A00c_left -> contra_indices_left', 
                                    'A00c_left -> ipsi_indices_right', 'A00c_left -> contra_indices_right',
                                    'vtd_left -> ipsi_indices_left', 'vtd_left -> contra_indices_left', 
                                    'vtd_left -> ipsi_indices_right', 'vtd_left -> contra_indices_right',
                                    'thermo_left -> ipsi_indices_left', 'thermo_left -> contra_indices_left', 
                                    'thermo_left -> ipsi_indices_right', 'thermo_left -> contra_indices_right',
                                    'photo_left -> ipsi_indices_left', 'photo_left -> contra_indices_left', 
                                    'photo_left -> ipsi_indices_right', 'photo_left -> contra_indices_right'])

sns.heatmap(ipsi_contra_flow, ax = axs, rasterized = True)

# %%
# number of ipsi or contra neurons visited >50 per hop
# initial exploratory chunk
fig, axs = plt.subplots(
    1, 1, figsize=(8, 8)
)
threshold = 50

ipsi_contra_flow_num_neurons = pd.DataFrame([(ORN_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (ORN_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0),
                            (ORN_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0), 
                            (ORN_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0),
                            (AN_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (AN_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0),
                            (AN_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0), 
                            (AN_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0),
                            (MN_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (MN_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0),
                            (MN_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0), 
                            (MN_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0),
                            (A00c_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (A00c_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0),
                            (A00c_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0), 
                            (A00c_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0),
                            (vtd_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (vtd_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0),
                            (vtd_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0), 
                            (vtd_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0),
                            (thermo_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (thermo_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0),
                            (thermo_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0), 
                            (thermo_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0),
                            (photo_hit_hist_left[ipsi_indices_left]>threshold).sum(axis = 0), 
                            (photo_hit_hist_left[contra_indices_left]>threshold).sum(axis = 0),
                            (photo_hit_hist_left[ipsi_indices_right]>threshold).sum(axis = 0), 
                            (photo_hit_hist_left[contra_indices_right]>threshold).sum(axis = 0)], 
                            index = ['ORN_left -> ipsi_indices_left', 'ORN_left -> contra_indices_left', 
                                    'ORN_left -> ipsi_indices_right', 'ORN_left -> contra_indices_right',
                                    'AN_left -> ipsi_indices_left', 'AN_left -> contra_indices_left', 
                                    'AN_left -> ipsi_indices_right', 'AN_left -> contra_indices_right',
                                    'MN_left -> ipsi_indices_left', 'MN_left -> contra_indices_left', 
                                    'MN_left -> ipsi_indices_right', 'MN_left -> contra_indices_right',
                                    'A00c_left -> ipsi_indices_left', 'A00c_left -> contra_indices_left', 
                                    'A00c_left -> ipsi_indices_right', 'A00c_left -> contra_indices_right',
                                    'vtd_left -> ipsi_indices_left', 'vtd_left -> contra_indices_left', 
                                    'vtd_left -> ipsi_indices_right', 'vtd_left -> contra_indices_right',
                                    'thermo_left -> ipsi_indices_left', 'thermo_left -> contra_indices_left', 
                                    'thermo_left -> ipsi_indices_right', 'thermo_left -> contra_indices_right',
                                    'photo_left -> ipsi_indices_left', 'photo_left -> contra_indices_left', 
                                    'photo_left -> ipsi_indices_right', 'photo_left -> contra_indices_right'])

sns.heatmap(ipsi_contra_flow_num_neurons, ax = axs, rasterized = True)

