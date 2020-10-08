#%%
import os

try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:

    pass

#%%
import sys
sys.path.append("/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/")

import pandas as pd
import numpy as np
import connectome_tools.process_matrix as promat
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pymaid
from pymaid_creds import url, name, password, token

# convert pair-sorted brain/sensories matrix to binary matrix based on synapse threshold
matrix_ad = pd.read_csv('data/axon-dendrite.csv', header=0, index_col=0)
matrix_dd = pd.read_csv('data/dendrite-dendrite.csv', header=0, index_col=0)
matrix_aa = pd.read_csv('data/axon-axon.csv', header=0, index_col=0)
matrix_da = pd.read_csv('data/dendrite-axon.csv', header=0, index_col=0)

matrix = matrix_ad + matrix_dd + matrix_aa + matrix_da

# the columns are string by default and the indices int; now both are int
matrix_ad.columns = pd.to_numeric(matrix_ad.columns)
matrix_dd.columns = pd.to_numeric(matrix_dd.columns)
matrix_aa.columns = pd.to_numeric(matrix_aa.columns)
matrix_da.columns = pd.to_numeric(matrix_da.columns)
matrix.columns = pd.to_numeric(matrix.columns)

# import pair list CSV, manually generated
pairs = pd.read_csv('data/pairs-2020-05-08.csv', header = 0)
paired = pairs.values.flatten()

# %%
rm = pymaid.CatmaidInstance(url, name, password, token)

# pull sensory annotations and then pull associated skids
inputs = pymaid.get_annotated('mw brain inputs')
order2 = pymaid.get_annotated('mw brain inputs 2nd_order')
order3 = pymaid.get_annotated('mw brain inputs 3rd_order')
order4 = pymaid.get_annotated('mw brain inputs 4th_order')

inputs = inputs['name'].sort_values()
order2 = order2['name'].sort_values()
order3 = order3['name'].sort_values()
order4 = order4['name'].sort_values()

inputs.index = np.arange(0, len(inputs), 1) # reset indices
order2.index = np.arange(0, len(order2), 1) # reset indices
order3.index = np.arange(0, len(order3), 1) # reset indices
order4.index = np.arange(0, len(order4), 1) # reset indices

input_skids = []
for i in np.arange(0, len(inputs), 1):
    input_skid = inputs[i]
    input_skid = pymaid.get_skids_by_annotation(input_skid)
    input_skids.append(input_skid)

order2_skids = []
for i in np.arange(0, len(order2), 1):
    order2_skid = order2[i]
    order2_skid = pymaid.get_skids_by_annotation(order2_skid)
    order2_skids.append(order2_skid)

order3_skids = []
for i in np.arange(0, len(order3), 1):
    order3_skid = order3[i]
    order3_skid = pymaid.get_skids_by_annotation(order3_skid)
    order3_skids.append(order3_skid)

order4_skids = []
for i in np.arange(0, len(order4), 1):
    order4_skid = order4[i]
    order4_skid = pymaid.get_skids_by_annotation(order4_skid)
    order4_skids.append(order4_skid)
    
# %%
def membership(list1, list2):
    set1 = set(list1)
    return [item in set1 for item in list2]
# returns boolean of list2

from upsetplot import plot
from upsetplot import from_contents
import itertools

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

#fig, ax = plt.subplots(1,1,figsize=(8,4))
input_dict = dict(AN=input_skids[0], MN = input_skids[1], ORN = input_skids[2], photo = input_skids[6], vtd = input_skids[4], thermo = input_skids[3], A00c = input_skids[5])
plot(from_contents(input_dict), sort_by = 'cardinality', sort_categories_by = None)
plt.savefig('identify_neuron_classes/plots/input.pdf', format='pdf', bbox_inches='tight')

#fig, ax = plt.subplots(1,1,figsize=(8,4))
order2_dict = dict(AN=order2_skids[0], MN = order2_skids[1], ORN = order2_skids[2], photo = order2_skids[6], vtd = order2_skids[4], thermo = order2_skids[3], A00c = order2_skids[5])
plot(from_contents(order2_dict), sort_by = 'cardinality', sort_categories_by = None)
plt.savefig('identify_neuron_classes/plots/input2.pdf', format='pdf', bbox_inches='tight')

#fig, ax = plt.subplots(1,1,figsize=(8,4))
order3_dict = dict(AN=order3_skids[0], MN = order3_skids[1], ORN = order3_skids[2], photo = order3_skids[6], vtd = order3_skids[4], thermo = order3_skids[3], A00c = order3_skids[5])
plot(from_contents(order3_dict), sort_by = 'cardinality', sort_categories_by = None)
plt.savefig('identify_neuron_classes/plots/input3.pdf', format='pdf', bbox_inches='tight')

#fig, ax = plt.subplots(1,1,figsize=(8,4))
order4_dict = dict(AN=order4_skids[0], MN = order4_skids[1], ORN = order4_skids[2], photo = order4_skids[6], vtd = order4_skids[4], thermo = order4_skids[3], A00c = order4_skids[5])
plot(from_contents(order4_dict), sort_by = 'cardinality', sort_categories_by = None)
plt.savefig('identify_neuron_classes/plots/input4.pdf', format='pdf', bbox_inches='tight')

# %%
# number of LNs, PNs, descending each layer

# import different PNs
order2PN = pymaid.get_annotated('mw brain inputs 2nd_order PN')
order3PN = pymaid.get_annotated('mw brain inputs 3rd_order PN')
# order4PN

order2PN = order2PN['name'].sort_values()
order3PN = order3PN['name'].sort_values()

order2PN.index = np.arange(0, len(order2PN), 1) # reset indices
order3PN.index = np.arange(0, len(order3PN), 1) # reset indices

order2PN_skids = []
for i in np.arange(0, len(order2PN), 1):
    order2PN_skid = order2PN[i]
    order2PN_skid = pymaid.get_skids_by_annotation(order2PN_skid)
    order2PN_skids.append(order2PN_skid)

order3PN_skids = []
for i in np.arange(0, len(order3PN), 1):
    order3PN_skid = order3PN[i]
    order3PN_skid = pymaid.get_skids_by_annotation(order3PN_skid)
    order3PN_skids.append(order3PN_skid)

# import different LNs
ORN2LN = pymaid.get_skids_by_annotation('mw ORN 2nd_order LN')
MN2LN = pymaid.get_skids_by_annotation('mw MN 2nd_order LN')
photo2LN = pymaid.get_skids_by_annotation('mw photo 2nd_order LN')

ORN3LN = pymaid.get_skids_by_annotation('mw ORN 3rd_order LN')
AN3LN = pymaid.get_skids_by_annotation('mw AN 3rd_order LN')
thermo3LN = pymaid.get_skids_by_annotation('mw thermo 3rd_order LN')

A00c4LN = pymaid.get_skids_by_annotation('mw A00c 4th_order LN')
AN4LN = pymaid.get_skids_by_annotation('mw AN 4th_order LN')
MN4LN = pymaid.get_skids_by_annotation('mw MN 4th_order LN')
ORN4LN = pymaid.get_skids_by_annotation('mw ORN 4th_order LN')
thermo4LN = pymaid.get_skids_by_annotation('mw thermo 4th_order LN')

# import output types
oVNC = pymaid.get_skids_by_annotation('mw dVNC')
oSEZ = pymaid.get_skids_by_annotation('mw dSEZ')
oRG = pymaid.get_skids_by_annotation('mw RG')
# %%

A00c2 = len(order2_skids[0])
LN_A00c2 = 0
oVNC_A00c2 = len(np.intersect1d(oVNC, order2_skids[0]))
oSEZ_A00c2 = len(np.intersect1d(oSEZ, order2_skids[0]))
oRG_A00c2 = len(np.intersect1d(oRG, order2_skids[0]))

A00c2_counts = [A00c2, LN_A00c2, oVNC_A00c2, oSEZ_A00c2, oRG_A00c2]

A00c3 = len(order3_skids[0])
LN_A00c3 = 0
oVNC_A00c3 = len(np.intersect1d(oVNC, order3_skids[0]))
oSEZ_A00c3 = len(np.intersect1d(oSEZ, order3_skids[0]))
oRG_A00c3 = len(np.intersect1d(oRG, order3_skids[0]))

A00c3_counts = [A00c3, LN_A00c3, oVNC_A00c3, oSEZ_A00c3, oRG_A00c3]

A00c4 = len(order4_skids[0])
LN_A00c4 = len(A00c4LN)
oVNC_A00c4 = len(np.intersect1d(oVNC, order4_skids[0]))
oSEZ_A00c4 = len(np.intersect1d(oSEZ, order4_skids[0]))
oRG_A00c4 = len(np.intersect1d(oRG, order4_skids[0]))

A00c4_counts = [A00c4, LN_A00c4, oVNC_A00c4, oSEZ_A00c4, oRG_A00c4]
# %%
