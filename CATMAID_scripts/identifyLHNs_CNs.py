import connectome_tools.process_matrix as promat
import pandas as pd
import glob as gl

# load whole brain connectivity matrix
matrix = pd.read_csv('data/Gadn-pair-sorted.csv', header=0, index_col=0)

# import pair list CSV, manually generated
pairs = pd.read_csv('data/bp-pairs-2020-01-28.csv', header = 0)

# identify list of neuron-groups to import
neuron_groups = gl.glob('neuron_groups_data/*.json')

# load skids of each neuron class
MBONs = pd.read_json(neuron_groups[0])['skeleton_id'].values
mPNs = pd.read_json(neuron_groups[1])['skeleton_id'].values
tPNs = pd.read_json(neuron_groups[2])['skeleton_id'].values
uPNs = pd.read_json(neuron_groups[3])['skeleton_id'].values
vPNs = pd.read_json(neuron_groups[4])['skeleton_id'].values

# trim out neurons that aren't in the matrix (to avoid code breaking)
mPNs = promat.trim_missing(mPNs, matrix)
tPNs = promat.trim_missing(tPNs, matrix)
uPNs = promat.trim_missing(uPNs, matrix)
vPNs = promat.trim_missing(vPNs, matrix)
MBONs = promat.trim_missing(MBONs, matrix)

# generating whole brain matrices with summed input from selected neuron type
sum_vPNs = promat.summed_input(vPNs, matrix, pairs)
sum_mPNs = promat.summed_input(mPNs, matrix, pairs)
sum_tPNs = promat.summed_input(tPNs, matrix, pairs)
sum_uPNs = promat.summed_input(uPNs, matrix, pairs)
sum_MBONs = promat.summed_input(MBONs, matrix, pairs)

