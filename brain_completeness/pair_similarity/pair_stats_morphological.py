#%%
import os
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:
    pass

#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import connectome_tools.process_matrix as promat
import pymaid
from pymaid_creds import url, name, password, token

#rm = pymaid.CatmaidInstance(url, name, password, token)

pairs = pd.read_csv('data/bp-pairs-2020-01-28.csv', header = 0)
morph_stats = pd.read_csv('data/brain_skeleton_measurements.csv', header = 0)
morph_stats_pub = pd.read_csv('data/published_skeleton_measurements.csv', header = 0)


# %%
cable_diff = []
input_diff = []
output_diff = []
pair_ids = []

for i in np.arange(0, len(pairs.rightid), 1):
    if (sum(morph_stats['Skeleton']==pairs.leftid[i])+sum(morph_stats['Skeleton']==pairs.rightid[i]) > 1 ):

        neuron_index = np.where(morph_stats['Skeleton']==pairs.rightid[i])[0]
        partner_index = np.where(morph_stats['Skeleton']==pairs.leftid[i])[0]

        average_outputs = (morph_stats['N outputs'][neuron_index].values + morph_stats['N outputs'][partner_index].values)/2
        average_inputs = (morph_stats['N inputs'][neuron_index].values + morph_stats['N inputs'][partner_index].values)/2
        average_cable = (morph_stats['Raw cable (nm)'][neuron_index].values + morph_stats['Raw cable (nm)'][partner_index].values)/2

        # percent difference between pair neurons (left-right)/left
        out_diff = (morph_stats['N outputs'][neuron_index].values 
                    - morph_stats['N outputs'][partner_index].values)/average_outputs
        in_diff = (morph_stats['N inputs'][neuron_index].values 
                    - morph_stats['N inputs'][partner_index].values)/average_inputs
        cab_diff = (morph_stats['Raw cable (nm)'][neuron_index].values 
                    - morph_stats['Raw cable (nm)'][partner_index].values)/average_cable

        cable_diff.append(cab_diff[0])
        input_diff.append(in_diff[0])
        output_diff.append(out_diff[0])
        pair_ids.append([pairs.rightid[i], pairs.leftid[i]])

# clean up the values with inf and -inf
output_diff_cleaned = []
for i in np.arange(0, len(output_diff), 1):
    if(output_diff[i] != float("inf") and output_diff[i] != float("-inf") and output_diff[i] != "nan"):
        output_diff_cleaned.append(output_diff[i])

output_diff_cleaned = np.array(output_diff_cleaned)
input_diff = np.array(input_diff)
cable_diff = np.array(cable_diff)
# %%
# generate cable_diff plot
fig, ax = plt.subplots(1,1,figsize=(.75,1))

sns.distplot(cable_diff, ax = ax, hist = True, kde = False, hist_kws=dict(edgecolor=(0, 0, 0, 1), linewidth=0.25), norm_hist=True)

ax.set(xlim = (-1, 1))
#ax.set(ylim = (0, 0.8))
ax.set(xticks=np.arange(-1,1.5,0.5))
ax.xaxis.set_tick_params(width=0.5)
ax.yaxis.set_tick_params(width=0.5)
ax.tick_params(labelsize=6)
ax.set_ylabel('Neuron Pairs', fontname="Arial", fontsize = 6)
ax.set_xlabel('Percent difference between pairs', fontname="Arial", fontsize = 6)

for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(0.5)

for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")

plt.savefig('brain_completeness/plots/cable_diff.pdf', bbox_inches='tight', transparent = True)


# %%
fig, ax = plt.subplots(1,1,figsize=(.75,1))

sns.distplot(cable_diff, color = 'gray', ax = ax, hist = False, kde = True, kde_kws=dict(linewidth=0.5), norm_hist=True)
sns.distplot(input_diff[np.where(input_diff>-2)[0]], ax = ax, hist = False, kde = True, kde_kws=dict(linewidth=0.5), norm_hist=True)
sns.distplot(output_diff_cleaned[np.where(output_diff_cleaned>-2)[0]], ax = ax, hist = False, kde = True, kde_kws=dict(linewidth=0.5), norm_hist=True)
plt.axvline(np.mean(cable_diff), 0, 1, color = 'gray', linewidth = 0.5)
#plt.axvline(np.mean(input_diff[np.where(input_diff>-2)[0]]), 0, 1, color = 'blue', linewidth = 0.25)
#plt.axvline(np.mean(output_diff_cleaned), 0, 1, color = 'orange', linewidth = 0.25)

ax.set(xlim = (-1, 1))
#ax.set(ylim = (0, 0.8))
ax.set(xticks=np.arange(-1,1.5,0.5))
ax.xaxis.set_tick_params(width=0.5)
ax.yaxis.set_tick_params(width=0.5)
ax.tick_params(labelsize=6)
ax.set_ylabel('Neuron Pairs', fontname="Arial", fontsize = 6)
ax.set_xlabel('Percent difference between pairs', fontname="Arial", fontsize = 6)

for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(0.5)

for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")

plt.savefig('brain_completeness/plots/pairs_diff.pdf', bbox_inches='tight', transparent = True)


# %%
# difference in these metrics for published neurons
cable_diff_pub = []
input_diff_pub = []
output_diff_pub = []
pair_ids_pub = []

for i in np.arange(0, len(pairs.rightid), 1):
    if (sum(morph_stats_pub['Skeleton']==pairs.leftid[i])+sum(morph_stats_pub['Skeleton']==pairs.rightid[i]) > 1 ):

        neuron_index = np.where(morph_stats_pub['Skeleton']==pairs.rightid[i])[0]
        partner_index = np.where(morph_stats_pub['Skeleton']==pairs.leftid[i])[0]

        # percent difference between pair neurons (left-right)/left
        out_diff = (morph_stats_pub['N outputs'][neuron_index].values 
                    - morph_stats_pub['N outputs'][partner_index].values)/morph_stats_pub['N outputs'][neuron_index].values
        in_diff = (morph_stats_pub['N inputs'][neuron_index].values 
                    - morph_stats_pub['N inputs'][partner_index].values)/morph_stats_pub['N inputs'][neuron_index].values
        cab_diff = (morph_stats_pub['Raw cable (nm)'][neuron_index].values 
                    - morph_stats_pub['Raw cable (nm)'][partner_index].values)/morph_stats_pub['Raw cable (nm)'][neuron_index].values

        cable_diff_pub.append(cab_diff[0])
        input_diff_pub.append(in_diff[0])
        output_diff_pub.append(out_diff[0])
        pair_ids_pub.append([pairs.rightid[i], pairs.leftid[i]])


#%%

fig, ax = plt.subplots(1,1,figsize=(6,6))

sns.distplot(cable_diff_pub, ax = ax, hist = True, kde = False, hist_kws=dict(edgecolor=(0, 0, 0, 1), linewidth=1), norm_hist=True)
sns.distplot(cable_diff, ax = ax, hist = True, kde = False, hist_kws=dict(edgecolor=(0, 0, 0, 1), linewidth=1), norm_hist=True)

#%%

fig, ax = plt.subplots(1,1,figsize=(6,6))

sns.distplot(input_diff_pub, ax = ax, hist = True, kde = False, hist_kws=dict(edgecolor=(0, 0, 0, 1), linewidth=1), norm_hist=True)
sns.distplot(input_diff, ax = ax, hist = True, kde = False, hist_kws=dict(edgecolor=(0, 0, 0, 1), linewidth=1), norm_hist=True)
# %%
# selecting neurons for review

for i in np.arange(0, len(input_diff), 1):
    input_diff[i] = abs(input_diff[i])

for i in np.arange(0, len(output_diff), 1):
    output_diff[i] = abs(output_diff[i])

for i in np.arange(0, len(cable_diff), 1):
    cable_diff[i] = abs(cable_diff[i])

input_diff = np.array(input_diff)
output_diff = np.array(output_diff)
cable_diff = np.array(cable_diff)
pair_ids = np.array(pair_ids)

# different threshold for review
threshold_cable = 2*np.std(cable_diff_pub)
threshold_input = 2*np.std(input_diff_pub)

cable_bool = np.array(cable_diff) > threshold_cable
output_bool = np.array(output_diff) > threshold_input
input_bool = np.array(input_diff) > threshold_input

total_bool = cable_bool + output_bool + input_bool


print(len(cable_diff[cable_bool]))
print(len(output_diff[output_bool]))
print(len(input_diff[input_bool]))
print(sum(total_bool))
# %%
# identify the weak/less constructed partner of each pair


# %%
# export as CSV
pairs_toexport = pd.DataFrame(pair_ids[total_bool], columns = ['leftid', 'rightid'])

pd.DataFrame(pair_ids[total_bool]).to_csv('brain_completeness/to_review.csv')

# %%
