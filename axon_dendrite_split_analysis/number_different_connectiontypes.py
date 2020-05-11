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
import math

aa = pd.read_csv('data/axon-axon.csv', header = 0, index_col = 0)
ad = pd.read_csv('data/axon-dendrite.csv', header = 0, index_col = 0)
dd = pd.read_csv('data/dendrite-dendrite.csv', header = 0, index_col = 0)
da = pd.read_csv('data/dendrite-axon.csv', header = 0, index_col = 0)

# %%
# synaptic numbers

aa_syn = np.matrix(aa).sum(axis=None)
ad_syn = np.matrix(ad).sum(axis=None)
dd_syn = np.matrix(dd).sum(axis=None)
da_syn = np.matrix(da).sum(axis=None)

total_syn = aa_syn + ad_syn + dd_syn + da_syn

print(ad_syn)
print(aa_syn)
print(dd_syn)
print(da_syn)

print(ad_syn/total_syn)
print(aa_syn/total_syn)
print(dd_syn/total_syn)
print(da_syn/total_syn)


# %%
# edge numbers
aa_bin = aa
ad_bin = ad
dd_bin = dd
da_bin = da


aa_bin[aa > 0] = 1
ad_bin[ad > 0] = 1
dd_bin[dd > 0] = 1
da_bin[da > 0] = 1

aa_edge = np.matrix(aa_bin).sum(axis=None)
ad_edge = np.matrix(ad_bin).sum(axis=None)
dd_edge = np.matrix(dd_bin).sum(axis=None)
da_edge = np.matrix(da_bin).sum(axis=None)

total_edge = aa_edge + ad_edge + dd_edge + da_edge

print(ad_edge)
print(aa_edge)
print(dd_edge)
print(da_edge)

print(ad_edge/total_edge)
print(aa_edge/total_edge)
print(dd_edge/total_edge)
print(da_edge/total_edge)

# %%
