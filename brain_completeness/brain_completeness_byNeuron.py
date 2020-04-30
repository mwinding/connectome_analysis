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
fig, ax = plt.subplots(1,1,figsize=(.75,1.3))

sns.distplot(completeness['ppn_pre'], color = 'royalblue',bins = 20, ax = ax, hist = True, kde = False, hist_kws=dict(edgecolor=(0, 0, 0, 1), linewidth=1), norm_hist=False)

#sns.distplot(values, ax = ax, hist = True, kde_kws = {'shade': True})

ax.set(xlim = (0, 1))
plt.axvline(np.mean(completeness['ppn_pre']), 0, 1, linewidth = 0.5, color = 'royalblue')

ax.set(xticks=np.arange(0,1.5,0.5))
ax.xaxis.set_tick_params(width=0.5)
ax.yaxis.set_tick_params(width=0.5)
ax.tick_params(labelsize=6)
ax.set_ylabel('Number of Neurons', fontname="Arial", fontsize = 6)
ax.set_xlabel('Input Completeness', fontname="Arial", fontsize = 6)

for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(0.5)

for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")

plt.savefig('brain_completeness/plots/presynaptic_completeness.pdf', format='pdf', bbox_inches='tight')

# %%
fig, ax = plt.subplots(1,1,figsize=(.75,1.3))

sns.distplot(completeness['ppn_post'], color = 'crimson',bins = 20, ax = ax, hist = True, kde = False, hist_kws=dict(edgecolor=(0, 0, 0, 1), linewidth=1), norm_hist=False)
#sns.distplot(values, ax = ax, hist = True, kde_kws = {'shade': True})

ax.set(xlim = (0, 1))
#ax.set(xticks=np.arange(1,11,1))
plt.axvline(np.mean(completeness['ppn_post']), 0, 1, linewidth = 0.5, color = 'crimson')

ax.set(xticks=np.arange(0,1.5,0.5))
ax.xaxis.set_tick_params(width=0.5)
ax.yaxis.set_tick_params(width=0.5)
ax.tick_params(labelsize=6)
ax.set_ylabel('Number of Neurons', fontname="Arial", fontsize = 6)
ax.set_xlabel('Output Completeness', fontname="Arial", fontsize = 6)

for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(0.5)

for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")

plt.savefig('brain_completeness/plots/postsynaptic_completeness.pdf', format='pdf', bbox_inches='tight')


# %%
# combined completeness metrics
fig, ax = plt.subplots(1,1,figsize=(.75,1.3))

sns.distplot(completeness['ppn_post'], bins = 20, ax = ax, hist = True, kde = True, norm_hist=True)
sns.distplot(completeness['ppn_pre'], bins = 20, ax = ax, hist = True, kde = True, norm_hist=True)

#ax.set(xlim = (0, 10))
#ax.set(xticks=np.arange(1,11,1))
ax.set_ylabel('Number of Neurons')
ax.set_xlabel('Per Neuron Completeness')
plt.axvline(np.mean(completeness['ppn_post']), 0, 1, linewidth = 0.5)
plt.axvline(np.mean(completeness['ppn_pre']), 0, 1, linewidth = 0.5, color='orange')

#plt.savefig('brain_completeness/plots/postsynaptic_completeness.svg', format='svg')

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
