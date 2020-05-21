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

inputs = inputs.iloc[[3,4,0,1,5,6,2],]
order3 = order3.iloc[[6, 5, 4, 1, 3, 0, 2, 7, 8, 9],]
order4 = order4.iloc[[1, 2, 3, 5, 6, 0, 4, 7],]

inputs.index = np.arange(0, len(inputs['name']), 1) # reset indices
order3.index = np.arange(0, len(order3['name']), 1) # reset indices
order4.index = np.arange(0, len(order4['name']), 1) # reset indices

input_skids = []
for i in np.arange(0, len(inputs), 1):
    input_skid = inputs['name'][i]
    input_skid = pymaid.get_skids_by_annotation(input_skid)
    input_skids.append(input_skid)

order2_skids = []
for i in np.arange(0, len(order2), 1):
    order2_skid = order2['name'][i]
    order2_skid = pymaid.get_skids_by_annotation(order2_skid)
    order2_skids.append(order2_skid)

order3_skids = []
for i in np.arange(0, len(order3), 1):
    order3_skid = order3['name'][i]
    order3_skid = pymaid.get_skids_by_annotation(order3_skid)
    order3_skids.append(order3_skid)

order4_skids = []
for i in np.arange(0, len(order4), 1):
    order4_skid = order4['name'][i]
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

fig, ax = plt.subplots(1,1,figsize=(8,4))
input_dict = dict(AN=input_skids[0], MN = input_skids[1], ORN = input_skids[2], photo = input_skids[6], vtd = input_skids[4], thermo = input_skids[3], A00c = input_skids[5])
plot(from_contents(input_dict), sort_by = 'cardinality', sort_categories_by = None)
plt.savefig('identify_neuron_classes/plots/input.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1,figsize=(8,4))
order2_dict = dict(AN=order2_skids[0], MN = order2_skids[1], ORN = order2_skids[2], photo = order2_skids[6], vtd = order2_skids[4], thermo = order2_skids[3], A00c = order2_skids[5])
plot(from_contents(order2_dict), sort_by = 'cardinality', sort_categories_by = None)
plt.savefig('identify_neuron_classes/plots/input2.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1,figsize=(8,4))
order3_dict = dict(AN=order3_skids[0], MN = order3_skids[1], ORN = order3_skids[2], photo = order3_skids[6], vtd = order3_skids[4], thermo = order3_skids[3], A00c = order3_skids[5])
plot(from_contents(order3_dict), sort_by = 'cardinality', sort_categories_by = None)
plt.savefig('identify_neuron_classes/plots/input3.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1,figsize=(8,4))
order4_dict = dict(AN=order4_skids[0], MN = order4_skids[1], ORN = order4_skids[2], photo = order4_skids[6], vtd = order4_skids[4], thermo = order4_skids[3], A00c = order4_skids[5])
plot(from_contents(order4_dict), sort_by = 'cardinality', sort_categories_by = None)
plt.savefig('identify_neuron_classes/plots/input4.pdf', format='pdf', bbox_inches='tight')

# %%
