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
from pymaid_creds import url, name, password, token
import pymaid

rm = pymaid.CatmaidInstance(url, name, password, token)
completeness = pd.read_csv('data/brain_partner_completeness_2020-04-28.csv')

# %%
fig, ax = plt.subplots(1,1,figsize=(1,2))

sns.distplot(completeness['ppn_pre'], bins = 20, ax = ax, hist = True, kde = False, hist_kws=dict(edgecolor=(0, 0, 0, 1), linewidth=1), norm_hist=False)
#sns.distplot(values, ax = ax, hist = True, kde_kws = {'shade': True})

#ax.set(xlim = (0, 10))
ax.set_ylabel('Number of Neurons')
ax.set_xlabel('Presynaptic Completeness')
plt.axvline(np.mean(completeness['ppn_pre']), 0, 1)

plt.savefig('brain_completeness/plots/presynaptic_completeness.svg', format='svg')

# %%
fig, ax = plt.subplots(1,1,figsize=(1,2))

sns.distplot(completeness['ppn_post'], bins = 20, ax = ax, hist = True, kde = False, hist_kws=dict(edgecolor=(0, 0, 0, 1), linewidth=1), norm_hist=False)
#sns.distplot(values, ax = ax, hist = True, kde_kws = {'shade': True})

#ax.set(xlim = (0, 10))
#ax.set(xticks=np.arange(1,11,1))
ax.set_ylabel('Number of Neurons')
ax.set_xlabel('Postsynaptic Completeness')
plt.axvline(np.mean(completeness['ppn_post']), 0, 1)

plt.savefig('brain_completeness/plots/postsynaptic_completeness.svg', format='svg')

# %%
# removing the output neurons from analysis
# turns out it doesn't help (many of the descending are actually ~1 completeness)

dVNC = pymaid.get_skids_by_annotation('mw dVNC')
dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')
RG = pymaid.get_skids_by_annotation('mw RG')
outputs = dVNC + dSEZ + RG

post_complete = []
pre_complete = []
for i in np.arange(0, len(completeness['skeleton']), 1):
    if (completeness['skeleton'][i] not in outputs):
        post_complete.append(completeness['ppn_post'][i])
        pre_complete.append(completeness['ppn_pre'][i])
    if(completeness['skeleton'][i] in outputs):
        print('%i ignored' %i)

# %%
fig, ax = plt.subplots(1,1,figsize=(3,6))

sns.distplot(post_complete, bins = 20, ax = ax, hist = True, kde = False, hist_kws=dict(edgecolor=(0, 0, 0, 1), linewidth=1), norm_hist=False)
sns.distplot(completeness['ppn_post'], bins = 20, ax = ax, hist = True, kde = False, hist_kws=dict(edgecolor=(0, 0, 0, 1), linewidth=1), norm_hist=False)
#sns.distplot(values, ax = ax, hist = True, kde_kws = {'shade': True})

#ax.set(xlim = (0, 10))
#ax.set(xticks=np.arange(1,11,1))
ax.set_ylabel('Number of Neurons')
ax.set_xlabel('Output Completeness')
plt.axvline(np.mean(post_complete), 0, 1)


# %%
