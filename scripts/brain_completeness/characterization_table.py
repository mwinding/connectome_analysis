#%%
import os
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:
    pass

#%%
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

character = pd.read_csv('data/characterization_table.csv')

fragments = character.loc[character['category']=='fragment']
external = character.loc[character['category']=='external_object']
other_cell = character.loc[character['category']=='other_cell']


# %%
# fragment categorization

#fragments[fragments['cable_length'] < 1000.0]['cable_length']
frag1_count = len(fragments[fragments['cable_length'] < 1000.0]['cable_length'])
frag1_pre = sum(fragments[fragments['cable_length'] < 1000.0]['n_presynaptic_sites_in_brain'])
frag1_post = sum(fragments[fragments['cable_length'] < 1000.0]['n_postsynaptic_sites_in_brain'])

#fragments[(fragments['cable_length'] >= 1000.0) & (fragments['cable_length'] < 10000)]['cable_length']
frag1_10_count = len(fragments[(fragments['cable_length'] >= 1000.0) & (fragments['cable_length'] < 10000)]['cable_length'])
frag1_10_pre = sum(fragments[(fragments['cable_length'] >= 1000.0) & (fragments['cable_length'] < 10000)]['n_presynaptic_sites_in_brain'])
frag1_10_post = sum(fragments[(fragments['cable_length'] >= 1000.0) & (fragments['cable_length'] < 10000)]['n_postsynaptic_sites_in_brain'])

#fragments[fragments['cable_length'] >= 10000.0]['cable_length']
frag10_count = len(fragments[fragments['cable_length'] >= 10000.0]['cable_length'])
frag10_pre = sum(fragments[fragments['cable_length'] >= 10000.0]['n_presynaptic_sites_in_brain'])
frag10_post = sum(fragments[fragments['cable_length'] >= 10000.0]['n_postsynaptic_sites_in_brain'])

print('Fragments <1um\nNumber: %i\nPresynaptic Sites: %i\nPostsynaptic Sites: %i\n' %(frag1_count, frag1_pre, frag1_post))
print('Fragments >1um\nNumber: %i\nPresynaptic Sites: %i\nPostsynaptic Sites: %i\n' %(frag1_10_count, frag1_10_pre, frag1_10_post))
print('Fragments >10um\nNumber: %i\nPresynaptic Sites: %i\nPostsynaptic Sites: %i\n' %(frag10_count, frag10_pre, frag10_post))

# %%
# external object categorization

external_count = len(external)
external_pre = sum(external['n_presynaptic_sites_in_brain'])
external_post = sum(external['n_postsynaptic_sites_in_brain'])

print('External Objects\nNumber: %i\nPresynaptic Sites: %i\nPostsynaptic Sites: %i\n' %(external_count, external_pre, external_post))

# %%
# total complete
# brain neuron inputs/outputs + external inputs/outputs

brain_pre = 140122
brain_post = 388699

complete_pre = external_pre + brain_pre
complete_post = external_post + brain_post

total_pre = complete_pre + sum(fragments['n_presynaptic_sites_in_brain'])
total_post = complete_post + sum(fragments['n_postsynaptic_sites_in_brain'])

print('Presynaptic Completeness: %f' %(complete_pre/total_pre))
print('Presynaptic Completeness: %f' %(complete_post/total_post))

# %%
# other cells
len(other_cell['n_presynaptic_sites_in_brain'])
sum(other_cell['n_presynaptic_sites_in_brain'])
sum(other_cell['n_postsynaptic_sites_in_brain'])


# %%
