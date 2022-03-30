#%%
from pymaid_creds import url, name, password, token
from data_settings import pairs_path, data_date_A1_brain
import pymaid
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date

from contools import Promat, Celltype, Celltype_Analyzer

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

rm = pymaid.CatmaidInstance(url, token, name, password)
pairs = Promat.get_pairs(pairs_path=pairs_path)

ascending = pymaid.get_skids_by_annotation('mw ascending temp')
ascending_A1 = Celltype_Analyzer.get_skids_from_meta_annotation('mw A1 ascending')
ascending_nonA1 = list(np.setdiff1d(ascending, ascending_A1))
dVNC = pymaid.get_skids_by_annotation('mw dVNC')
dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')
dVNC_to_A1 = pymaid.get_skids_by_annotation('mw dVNC to A1')

# %%
# output to dVNC/ascending

# dVNC to ascending
dVNC_partners = pymaid.get_partners(dVNC)
dVNC_partners = dVNC_partners[dVNC_partners.num_nodes>1000]
ds_partners = dVNC_partners[dVNC_partners.loc[:, 'relation']=='downstream']

ds_partners.index = [int(x) for x in ds_partners.skeleton_id]

output_to_asc = ds_partners.loc[np.intersect1d(ds_partners.index, ascending), :].sum()
output_to_asc = output_to_asc.iloc[4:len(output_to_asc)-1]
output_counts = ds_partners.sum(axis=0)
output_counts = output_counts.iloc[4:len(output_to_asc)-1]

output_counts[output_counts==0]=1
np.mean(output_to_asc/output_counts)
np.std(output_to_asc/output_counts)

print(f'dVNCs output {np.mean(output_to_asc/output_counts)*100:.1f}+/-{np.std(output_to_asc/output_counts)*100:.1f}% to ascendings')

# ascending to dVNC
asc_partners = pymaid.get_partners(ascending)
asc_partners = asc_partners[asc_partners.num_nodes>1000]
ds_partners = asc_partners[asc_partners.loc[:, 'relation']=='downstream']

ds_partners.index = [int(x) for x in ds_partners.skeleton_id]

output_to_dVNC = ds_partners.loc[np.intersect1d(ds_partners.index, dVNC), :].sum()
output_to_dVNC = output_to_dVNC.iloc[4:len(output_to_dVNC)-1]
output_counts_asc = ds_partners.sum(axis=0)
output_counts_asc = output_counts_asc.iloc[4:len(output_to_dVNC)-1]

output_counts_asc[output_counts_asc==0]=1
np.mean(output_to_dVNC/output_counts_asc)
np.std(output_to_dVNC/output_counts_asc)

print(f'Ascendings output {np.mean(output_to_dVNC/output_counts_asc)*100:.1f}+/-{np.std(output_to_dVNC/output_counts_asc)*100:.1f}% to dVNCs')

# %%
# input from dVNC/ascending

# dVNC from ascending
dVNC_partners = pymaid.get_partners(dVNC)
us_partners = dVNC_partners[dVNC_partners.loc[:, 'relation']=='upstream']

us_partners.index = [int(x) for x in us_partners.skeleton_id]

input_from_asc = us_partners.loc[np.intersect1d(us_partners.index, ascending), :].sum()
input_from_asc = input_from_asc.iloc[4:len(input_from_asc)-1]
input_counts_dVNC = us_partners.sum(axis=0)
input_counts_dVNC = input_counts_dVNC.iloc[4:len(input_from_asc)-1]

input_counts_dVNC[input_counts_dVNC==0]=1

print(f'dVNCs receive {np.mean(input_from_asc/input_counts_dVNC)*100:.1f}+/-{np.std(input_from_asc/input_counts_dVNC)*100:.1f}% input from ascendings')

# dSEZ from ascending
dSEZ_partners = pymaid.get_partners(dSEZ)
us_partners = dSEZ_partners[dSEZ_partners.loc[:, 'relation']=='upstream']

us_partners.index = [int(x) for x in us_partners.skeleton_id]

input_dSEZ_from_asc = us_partners.loc[np.intersect1d(us_partners.index, ascending), :].sum()
input_dSEZ_from_asc = input_dSEZ_from_asc.iloc[4:len(input_dSEZ_from_asc)-1]
input_counts_dSEZ = us_partners.sum(axis=0)
input_counts_dSEZ = input_counts_dSEZ.iloc[4:len(input_dSEZ_from_asc)-1]

input_counts_dSEZ[input_counts_dSEZ==0]=1

print(f'dSEZs receive {np.mean(input_dSEZ_from_asc/input_counts_dSEZ)*100:.1f}+/-{np.std(input_dSEZ_from_asc/input_counts_dSEZ)*100:.1f}% input from ascendings')

# ascending from dVNC
asc_partners = pymaid.get_partners(ascending)
us_partners = asc_partners[asc_partners.loc[:, 'relation']=='upstream']

us_partners.index = [int(x) for x in us_partners.skeleton_id]

input_from_dVNC = us_partners.loc[np.intersect1d(us_partners.index, dVNC), :].sum()
input_from_dVNC = input_from_dVNC.iloc[4:len(input_from_dVNC)-1]
input_counts_asc = us_partners.sum(axis=0)
input_counts_asc = input_counts_asc.iloc[4:len(input_from_dVNC)-1]

input_counts_asc[input_counts_asc==0]=1

print(f'Ascendings receive {np.mean(input_from_dVNC/input_counts_asc)*100:.1f}+/-{np.std(input_from_dVNC/input_counts_asc)*100:.1f}% input from dVNCs')

# %%
# details on dVNC from ascending

# dVNC from ascending
dVNC_partners = pymaid.get_partners(dVNC)
us_partners = dVNC_partners[dVNC_partners.loc[:, 'relation']=='upstream']

us_partners.index = [int(x) for x in us_partners.skeleton_id]

input_from_asc_A1 = us_partners.loc[np.intersect1d(us_partners.index, ascending_A1), :].sum()
input_from_asc_A1 = input_from_asc_A1.iloc[4:len(input_from_asc_A1)-1]
input_counts_dVNC = us_partners.sum(axis=0)
input_counts_dVNC = input_counts_dVNC.iloc[4:len(input_from_asc_A1)-1]

input_counts_dVNC[input_counts_dVNC==0]=1

print(f'dVNCs receive {np.mean(input_from_asc_A1/input_counts_dVNC)*100:.1f}+/-{np.std(input_from_asc_A1/input_counts_dVNC)*100:.1f}% input from ascendings-A1')

# dVNC from ascending
dVNC_partners = pymaid.get_partners(dVNC)
us_partners = dVNC_partners[dVNC_partners.loc[:, 'relation']=='upstream']

us_partners.index = [int(x) for x in us_partners.skeleton_id]

input_from_asc = us_partners.loc[np.intersect1d(us_partners.index, ascending_nonA1), :].sum()
input_from_asc = input_from_asc.iloc[4:len(input_from_asc)-1]
input_counts_dVNC = us_partners.sum(axis=0)
input_counts_dVNC = input_counts_dVNC.iloc[4:len(input_from_asc)-1]

input_counts_dVNC[input_counts_dVNC==0]=1

print(f'dVNCs receive {np.mean(input_from_asc/input_counts_dVNC)*100:.1f}+/-{np.std(input_from_asc/input_counts_dVNC)*100:.1f}% input from ascendings-nonA1')

# %%