# %%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import connectome_tools.process_matrix as pm
import connectome_tools.celltype as ct
import navis

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

adj_ad = pm.Promat.pull_adj(type_adj='ad', subgraph='brain and accesssory')

# %%
# signal expansion from uPNs

# load neurons
KCs = list(np.setdiff1d(pymaid.get_skids_by_annotation('mw KC'), pymaid.get_skids_by_annotation('mw partially differentiated')))
LHNs = pymaid.get_skids_by_annotation('mw LHN')

uPNs = pymaid.get_skids_by_annotation('mw uPN')

# KC, LHNs from uPNs
KC_counts = (adj_ad.loc[uPNs, KCs]>0).sum(axis=0)
LHN_counts = (adj_ad.loc[uPNs, LHNs]>0).sum(axis=0)

print(f'KCs receive from: {KC_counts.mean():.2f} +/- {KC_counts.std():.2f} PNs')
print(f'LHNs receive from: {LHN_counts.mean():.2f} +/- {LHN_counts.std():.2f} PNs')

# uPN to KCs, LHNs
uPN_KC_outputs = (adj_ad.loc[uPNs, KCs]>0).sum(axis=1)
uPN_LHN_outputs = (adj_ad.loc[uPNs, LHNs]>0).sum(axis=1)

print(f'uPNs output to: {uPN_KC_outputs.mean():.2f} +/- {uPN_KC_outputs.std():.2f} KCs')
print(f'uPNs output to: {uPN_LHN_outputs.mean():.2f} +/- {uPN_LHN_outputs.std():.2f} LHNs')

# %%
# signal expansion per modality PNs

order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-warm', 'thermo-cold', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']
PNs_annot = ['mw ' + name + ' 2nd_order PN' for name in order]
o3_annot = ['mw ' + name + ' 3rd_order' for name in order]

PNs = [pymaid.get_skids_by_annotation(annot) for annot in PNs_annot]
o3 = [pymaid.get_skids_by_annotation(annot) for annot in o3_annot]


KC_o3 = [list(np.intersect1d(x, KCs)) for x in o3]
nonKC_o3 = [list(np.setdiff1d(x, KCs)) for x in o3]

data = []
for i in range(len(PNs)):
    up = PNs[i]
    KC_down = KC_o3[i]
    nonKC_down = nonKC_o3[i]

    if(len(KC_down)>0):
        KC_type1 = (adj_ad.loc[up, KC_down]>0).sum(axis=0)
        KC_type2 = (adj_ad.loc[up, KC_down]>0).sum(axis=1)

    else: # if there are no KCs in 3rd-order modality, generate Series with counts of 0 to avoid NaN
        KC_type1 = pd.Series([0,0])
        KC_type2 = pd.Series([0,0])

    nonKC_type1 = (adj_ad.loc[up, nonKC_down]>0).sum(axis=0)
    nonKC_type2 = (adj_ad.loc[up, nonKC_down]>0).sum(axis=1)

    data_temp = [f'{KC_type1.mean():.2f}+/-{KC_type1.std():.2f}', f'{nonKC_type1.mean():.2f}+/-{nonKC_type1.std():.2f}', 
                    f'{KC_type2.mean():.2f}+/-{KC_type2.std():.2f}', f'{nonKC_type2.mean():.2f}+/-{nonKC_type2.std():.2f}']
    #data_temp = [KC_type1.mean(), nonKC_type1.mean(), KC_type2.mean(), nonKC_type2.mean()]

    data.append(data_temp)

data = pd.DataFrame(data, index=order)

# %%
