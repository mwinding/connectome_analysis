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

from graspy.plot import gridplot, heatmap
from graspy.utils import binarize, pass_to_ranks
from src.data import load_metagraph
from src.visualization import CLASS_COLOR_DICT, adjplot

import cmasher as cmr

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

rm = pymaid.CatmaidInstance(url, name, password, token)

mg = load_metagraph("Gad", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
#mg = load_metagraph("G", version="2020-06-10", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg.calculate_degrees(inplace=True)

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

#%%
# pull brain input skids and then divide into left and right hemisphere

input_names = list(pymaid.get_annotated('mw brain inputs and ascending').name)
brain_inputs_list = list(map(pymaid.get_skids_by_annotation, input_names))
input_skids = [val for sublist in brain_inputs_list for val in sublist]
input_names_format = ['ORN', 'thermo', 'visual', 'AN', 'MN', 'vtd', 'asc-proprio', 'asc-mechano', 'asc-classII_III', 'asc-noci']

left_annot = pymaid.get_skids_by_annotation('mw left')
right_annot = pymaid.get_skids_by_annotation('mw right')

# need to switch several ascending neurons because they ascending contralateral
# including those in annotations: ['mw A1 ascending noci', 'mw A1 ascending proprio', 'mw A1 ascending mechano']
#   excluding: [2123422, 2784471]
neurons_to_flip = list(map(pymaid.get_skids_by_annotation, ['mw A1 ascending noci', 'mw A1 ascending proprio', 'mw A1 ascending mechano']))
neurons_to_flip = [x for sublist in neurons_to_flip for x in sublist]
neurons_to_flip = list(np.setdiff1d(neurons_to_flip, [2123422, 2784471]))
neurons_to_flip_left = [skid for skid in neurons_to_flip if skid in left_annot]
neurons_to_flip_right = [skid for skid in neurons_to_flip if skid in right_annot]

# removing neurons_to_flip and adding to the other side
left = list(np.setdiff1d(left_annot, neurons_to_flip_left)) + neurons_to_flip_right
right = list(np.setdiff1d(right_annot, neurons_to_flip_right)) + neurons_to_flip_left

# loading output neurons
output_names = list(pymaid.get_annotated('mw brain outputs').name)
brain_outputs_list = list(map(pymaid.get_skids_by_annotation, output_names))
output_skids = [val for sublist in brain_outputs_list for val in sublist]

# identify left and right side for each skid category
def split_hemilateral_to_indices(skids, left, right, skids_order):
    intersect_left = np.intersect1d(skids, left)
    indices_left = np.where([x in intersect_left for x in skids_order])[0]
    intersect_right = np.intersect1d(skids, right)
    indices_right = np.where([x in intersect_right for x in skids_order])[0]

    return(indices_left, indices_right, intersect_left, intersect_right)

# split according to left/right input type and identify indices of adj for cascade
inputs_split = [split_hemilateral_to_indices(skids, left, right, adj.index) for skids in brain_inputs_list]
outputs_split = [split_hemilateral_to_indices(skids, left, right, adj.index) for skids in brain_outputs_list]

#ORN_indices_left, ORN_indices_right, ORN_left, ORN_right = split_hemilateral_to_indices(ORN_skids, left, right, mg)
input_indices_left, input_indices_right, input_left, input_right = split_hemilateral_to_indices(input_skids, left, right, adj.index)
output_indices = np.where([x in output_skids for x in adj.index])[0]

#%%
# cascades from left and right sensories
from src.traverse import Cascade, to_transmission_matrix
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot
from joblib import Parallel, delayed

def run_cascade(i, cdispatch):
    return(cdispatch.multistart(start_nodes = i))

def excise_cascade(excised_skids, input_skids, output_skids, brain_inputs_list, left, right, adj, process_all_sens=False):
    adj_excised = adj.loc[np.setdiff1d(adj.index, excised_skids), np.setdiff1d(adj.index, excised_skids)]
    excised_input_indices_left, excised_input_indices_right, excised_input_left, excised_input_right = split_hemilateral_to_indices(input_skids, left, right, adj_excised.index)
    excised_output_indices = np.where([x in output_skids for x in adj_excised.index])[0]
    excised_inputs_split = [split_hemilateral_to_indices(skids, left, right, adj_excised.index) for skids in brain_inputs_list]

    p = 0.05
    max_hops = 20
    n_init = 10
    simultaneous = True

    transition_probs = to_transmission_matrix(adj_excised.values, p)
    cdispatch_excised = TraverseDispatcher(
        Cascade,
        transition_probs,
        stop_nodes = excised_output_indices,
        max_hops=max_hops,
        allow_loops = True,
        n_init=n_init,
        simultaneous=simultaneous,
    )

    excised_all_inputs_hit_hist_left, excised_all_inputs_hit_hist_right = Parallel(n_jobs=-1)(delayed(run_cascade)(i, cdispatch_excised) for i in [excised_input_indices_left, excised_input_indices_right])
    
    excised_inputs_hit_hist_list_left=[]
    excised_inputs_hit_hist_list_right=[]

    if(process_all_sens):
        excised_inputs_hit_hist_list_left = Parallel(n_jobs=-1)(delayed(run_cascade)(i[0], cdispatch_excised) for i in excised_inputs_split)
        excised_inputs_hit_hist_list_right = Parallel(n_jobs=-1)(delayed(run_cascade)(i[1], cdispatch_excised) for i in excised_inputs_split)

    return(excised_all_inputs_hit_hist_left, excised_all_inputs_hit_hist_right, excised_inputs_hit_hist_list_left, excised_inputs_hit_hist_list_right, adj_excised)

contra = pymaid.get_skids_by_annotation('mw contralateral axon')
bilateral = pymaid.get_skids_by_annotation('mw bilateral axon')
dVNC = pymaid.get_skids_by_annotation('mw dVNC')

contra_br = list(np.setdiff1d(contra, input_skids + output_skids))
bilateral_br = list(np.setdiff1d(bilateral, input_skids + output_skids))

# random set of bilaterals to match contra
np.random.seed(0)
random_nums = np.random.choice(len(bilateral_br), len(contra_br), replace = False)
random_bilateral_set = list(np.array(bilateral_br)[random_nums])

# random set of neurons corresponding to number of contra/bi removed
br_interneurons = np.setdiff1d(adj.index, input_skids + output_skids + contra_br + bilateral_br)
np.random.seed(0)
random_nums = np.random.choice(len(br_interneurons), len(contra_br), replace = False)
random_set270 = list(br_interneurons[random_nums])

np.random.seed(1)
random_nums = np.random.choice(len(br_interneurons), len(bilateral_br), replace = False)
random_set544 = list(br_interneurons[random_nums])

dC_all_inputs_hit_hist_left, dC_all_inputs_hit_hist_right, _, _, adj_dContra = excise_cascade(contra_br, input_skids, output_skids, brain_inputs_list, left, right, adj, process_all_sens=False)
dB_all_inputs_hit_hist_left, dB_all_inputs_hit_hist_right, _, _, adj_dBi = excise_cascade(bilateral_br, input_skids, output_skids, brain_inputs_list, left, right, adj, process_all_sens=False)
dB_dC_all_inputs_hit_hist_left, dB_dC_all_inputs_hit_hist_right, _, _, adj_dB_dC = excise_cascade(contra_br + bilateral_br, input_skids, output_skids, brain_inputs_list, left, right, adj, process_all_sens=False)
control_all_inputs_hit_hist_left, control_all_inputs_hit_hist_right, _, _, _ = excise_cascade([], input_skids, output_skids, brain_inputs_list, left, right, adj, process_all_sens=False)

con_dC_all_inputs_hit_hist_left, con_dC_all_inputs_hit_hist_right, _, _, con_adj_dContra = excise_cascade(random_set270, input_skids, output_skids, brain_inputs_list, left, right, adj, process_all_sens=False)
con_dB_all_inputs_hit_hist_left, con_dB_all_inputs_hit_hist_right, _, _, con_adj_dBi = excise_cascade(random_set544, input_skids, output_skids, brain_inputs_list, left, right, adj, process_all_sens=False)
con_dB_dC_all_inputs_hit_hist_left, con_dB_dC_all_inputs_hit_hist_right, _, _, con_adj_dB_dC = excise_cascade(random_set270 + random_set544, input_skids, output_skids, brain_inputs_list, left, right, adj, process_all_sens=False)

cut_Lhemi_left, cut_Lhemi_right, _, _, cut_Lhemi_adj = excise_cascade(left, input_skids, output_skids, brain_inputs_list, left, right, adj, process_all_sens=False)
# %%
import connectome_tools.cascade_analysis as casc

control_casc = casc.Cascade_Analyzer(control_all_inputs_hit_hist_left, adj.index, pairs)
dB_casc = casc.Cascade_Analyzer(dB_all_inputs_hit_hist_left, adj_dBi.index, pairs)
dC_casc = casc.Cascade_Analyzer(dC_all_inputs_hit_hist_left, adj_dContra.index, pairs)
dB_dC_casc = casc.Cascade_Analyzer(dB_dC_all_inputs_hit_hist_left, adj_dB_dC.index, pairs)

control_dB_casc = casc.Cascade_Analyzer(con_dB_all_inputs_hit_hist_left, con_adj_dBi.index, pairs)
control_dC_casc = casc.Cascade_Analyzer(con_dC_all_inputs_hit_hist_left, con_adj_dContra.index, pairs)
control_dB_dC_casc = casc.Cascade_Analyzer(con_dB_dC_all_inputs_hit_hist_left, con_adj_dB_dC.index, pairs)

cut_Lhemi_casc = casc.Cascade_Analyzer(cut_Lhemi_right, cut_Lhemi_adj.index, pairs)

control_casc.get_skid_hit_hist().loc[dVNC, :].sum(axis=0)
dB_casc.get_skid_hit_hist().loc[dVNC, :].sum(axis=0)
dC_casc.get_skid_hit_hist().loc[dVNC, :].sum(axis=0)
dB_dC_casc.get_skid_hit_hist().loc[dVNC, :].sum(axis=0)
control_dB_casc.get_skid_hit_hist().loc[dVNC, :].sum(axis=0)
control_dC_casc.get_skid_hit_hist().loc[dVNC, :].sum(axis=0)
control_dB_dC_casc.get_skid_hit_hist().loc[dVNC, :].sum(axis=0)

control_casc.get_skid_hit_hist().loc[np.intersect1d(dVNC, right), :].sum(axis=0)
cut_Lhemi_casc.get_skid_hit_hist().loc[np.intersect1d(dVNC, right), :].sum(axis=0)


