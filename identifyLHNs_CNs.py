import connectome_tools.process_matrix as promat
import pandas as pd
import glob as gl

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

# sort each list into paired neurons


#binaryBrain = promat.binary_matrix('data/Gadn-pair-sorted.csv', 0.01)

