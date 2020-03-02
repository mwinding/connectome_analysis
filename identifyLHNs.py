import connectome_tools.process_matrix as promat
import pandas as pd
import glob as gl

# load whole brain connectivity matrix
matrix = pd.read_csv('data/Gadn-pair-sorted.csv', header=0, index_col=0)

# import pair list CSV, manually generated
pairs = pd.read_csv('data/bp-pairs-2020-01-28.csv', header = 0)

# identify list of neuron-groups to import
neuron_groups = gl.glob('neuron_groups_data/*.json')

print(neuron_groups)


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

# thresholds to identify each cell type
uLHNs = promat.identify_downstream(sum_uPNs, 0.1, 0.00001)
vLHNs = promat.identify_downstream(sum_vPNs, 0.1, 0.00001)
tLHNs = promat.identify_downstream(sum_tPNs, 0.1, 0.00001)
mLHNs = promat.identify_downstream(sum_mPNs, 0.1, 0.00001)
MB2ONs = promat.identify_downstream(sum_MBONs, 0.1, 0.00001)

uLHNs.to_csv("outputs/uLHNs_0-1_0-00001_thresholds_2020-27-2.csv")
vLHNs.to_csv("outputs/vLHNs_0-1_0-00001_thresholds_2020-27-2.csv")
tLHNs.to_csv("outputs/tLHNs_0-1_0-00001_thresholds_2020-27-2.csv")
mLHNs.to_csv("outputs/mLHNs_0-1_0-00001_thresholds_2020-27-2.csv")







