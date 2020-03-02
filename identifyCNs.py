import connectome_tools.process_matrix as promat
import pandas as pd
import glob as gl
import numpy as np

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
LHNs = pd.read_json(neuron_groups[5])['skeleton_id'].values
LHNs2 = pd.read_json(neuron_groups[6])['skeleton_id'].values

# trim out neurons that aren't in the matrix (to avoid code breaking)
mPNs = promat.trim_missing(mPNs, matrix)
tPNs = promat.trim_missing(tPNs, matrix)
uPNs = promat.trim_missing(uPNs, matrix)
vPNs = promat.trim_missing(vPNs, matrix)
MBONs = promat.trim_missing(MBONs, matrix)
LHNs = promat.trim_missing(LHNs, matrix)
LHNs2 = promat.trim_missing(LHNs2, matrix)

# generating whole brain matrices with summed input from selected neuron type
sum_vPNs = promat.summed_input(vPNs, matrix, pairs)
sum_mPNs = promat.summed_input(mPNs, matrix, pairs)
sum_tPNs = promat.summed_input(tPNs, matrix, pairs)
sum_uPNs = promat.summed_input(uPNs, matrix, pairs)
sum_MBONs = promat.summed_input(MBONs, matrix, pairs)
sum_LHNs = promat.summed_input(LHNs, matrix, pairs)
sum_LHNs2 = promat.summed_input(LHNs2, matrix, pairs)

# thresholds to identify each cell type
uLHNs = promat.identify_downstream(sum_uPNs, 0.1, 0.00001)
vLHNs = promat.identify_downstream(sum_vPNs, 0.1, 0.00001)
tLHNs = promat.identify_downstream(sum_tPNs, 0.1, 0.00001)
mLHNs = promat.identify_downstream(sum_mPNs, 0.1, 0.00001)
MB2ONs = promat.identify_downstream(sum_MBONs, 0.1, 0.00001)
LH2Ns = promat.identify_downstream(sum_LHNs, 0.1, 0.00001)
LH2Ns2 = promat.identify_downstream(sum_LHNs2, 0.1, 0.00001)

#print(MB2ONs)
#print(LH2Ns)

#print(MB2ONs)

MB2ONs = np.concatenate([MB2ONs['leftid'].values, MB2ONs['rightid'].values])
LH2Ns = np.concatenate([LH2Ns['leftid'].values, LH2Ns['rightid'].values])
LH2Ns2 = np.concatenate([LH2Ns2['leftid'].values, LH2Ns2['rightid'].values])

for i in range(0, len(MB2ONs)): 
    MB2ONs[i] = int(MB2ONs[i])

for i in range(0, len(LH2Ns)): 
    LH2Ns[i] = int(LH2Ns[i])

for i in range(0, len(LH2Ns2)): 
    LH2Ns2[i] = int(LH2Ns2[i])

CNs = np.intersect1d(MB2ONs, LH2Ns)
CNs2 = np.intersect1d(MB2ONs, LH2Ns2)
LHN_CNs = np.intersect1d(MB2ONs, LHNs)
LHN2_CNs = np.intersect1d(MB2ONs, LHNs2)

print(len(CNs))
print(len(CNs2))
print(len(LHN_CNs))
print(len(LHN2_CNs))

CNs = np.union1d(CNs, LHN_CNs)
CNs2 = np.union1d(CNs2, LHN2_CNs)

print(len(CNs))
print(len(CNs2))

CNs = pd.DataFrame(CNs)
CNs2 = pd.DataFrame(CNs2)

CNs.to_csv("outputs/CNs_0-1_0-00001_thresholds_2020-2-3.csv")
CNs2.to_csv("outputs/CNs2_0-1_0-00001_thresholds_2020-2-3.csv")

'''
CNs = np.intersect1d(MB2ONs['leftid'], LH2Ns['leftid'])
CNs2 = np.intersect1d(MB2ONs['leftid'], LH2Ns2['leftid'])

#print(len(np.union1d(CNs, CNs2)))
LHNs = pd.read_json(neuron_groups[5])['skeleton_id']

for i in range(0, len(MB2ONs['leftid'])): 
    MB2ONs['leftid'].iloc[i] = int(MB2ONs['leftid'].iloc[i]) 

LHN_CNs = np.intersect1d(MB2ONs['leftid'], LHNs)
LHN2_CNs = np.intersect1d(MB2ONs['leftid'], LHNs2)


print(CNs)
print(CNs2)
print(LHN_CNs)
print(LHN2_CNs)

CNs.to_csv("outputs/CNs_traditional_0-1_0-00001_thresholds_2020-2-3.csv")
CNs2.to_csv("outputs/CNs2_0-1_0-00001_thresholds_2020-2-3.csv")
LHN_CNs.to_csv("outputs/LHN_CNs_0-1_0-00001_thresholds_2020-27-2.csv")
LHN2_CNs.to_csv("outputs/LHN2_CNs_0-1_0-00001_thresholds_2020-27-2.csv")
'''