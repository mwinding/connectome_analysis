#%%
from pymaid_creds import url, name, password, token
from data_settings import pairs_path, data_date
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

from contools import Celltype, Celltype_Analyzer, Promat, Cascade_Analyzer
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed

# load cascades from pickle object; generated in generate_data/cascades_all-modalities.py
n_init = 1000
input_hit_hist_list = pickle.load(open(f'data/cascades/all-sensory-modalities_processed-cascades_{n_init}-n_init_{data_date}.p', 'rb'))

# load pairs
pairs = Promat.get_pairs(pairs_path=pairs_path)

# %%
# pairwise thresholding of hit_hists
input_skids = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain inputs')

threshold = n_init/2
hops = 8

sens_types = [Celltype(hit_hist.get_name(), list(hit_hist.pairwise_threshold(hh_pairwise=hit_hist.hh_pairwise, pairs=pairs, threshold=threshold, hops=hops))) for hit_hist in input_hit_hist_list.cascade_objs]
sens_types_analyzer = Celltype_Analyzer(sens_types)

upset_threshold = 30 #(use 30 for main figure)
upset_threshold_dual_cats = 20 #(use 20 for main figure)
path = f'plots/sens-cascades_upset'
upset_members, members_selected, skids_excluded = sens_types_analyzer.upset_members(threshold=upset_threshold, path=path, plot_upset=True, 
                                                                                    exclude_singletons_from_threshold=True, exclude_skids=input_skids, threshold_dual_cats=upset_threshold_dual_cats)
# check what's in the skids_excluded group
# this group is assorted integrative that were excluded from the plot for simplicity; added manually as "other combinations" at far right of plot
_, celltypes = Celltype_Analyzer.default_celltypes()
test_excluded = Celltype_Analyzer([Celltype('excluded', skids_excluded)])
test_excluded.set_known_types(celltypes)
excluded_data = test_excluded.memberships(raw_num=True).drop(['sensories', 'ascendings'])

# cell identity of all of these categories
upset_analyzer = Celltype_Analyzer(members_selected)
upset_analyzer.set_known_types(celltypes)
upset_data = upset_analyzer.memberships(raw_num=True) # the data is all out of order compared to upset plot

# make annotations for labeled line vs integrative cells
names_combos = [x.name for x in upset_analyzer.Celltypes]
labeled_line = ['and' not in x for x in names_combos]
integrative = ['and' in x for x in names_combos]

labeled_line_skids = [upset_analyzer.Celltypes[i].skids for i,x in enumerate(labeled_line) if x==True]
integrative_skids = [upset_analyzer.Celltypes[i].skids for i,x in enumerate(integrative) if x==True] + [skids_excluded]
labeled_line_skids = [x for sublist in labeled_line_skids for x in sublist]
integrative_skids = [x for sublist in integrative_skids for x in sublist]

#pymaid.add_annotations(labeled_line_skids, f'mw cascade-{hops}-hop labeled-line {data_date}')
#pymaid.add_annotations(integrative_skids, f'mw cascade-{hops}-hop integrative {data_date}')

# %%
# integrative from most modalities
dVNC = pymaid.get_skids_by_annotation('mw dVNC')
sum(upset_data.loc['dVNCs', np.array([x.count('and') + 1 for x in upset_data.columns])>6])/len(dVNC)

dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')
sum(upset_data.loc['dSEZs', np.array([x.count('and') + 1 for x in upset_data.columns])>6])/len(dSEZ)

RGN = pymaid.get_skids_by_annotation('mw RGN')
sum(upset_data.loc['RGNs', np.array([x.count('and') + 1 for x in upset_data.columns])>6])/len(RGN)

# all brain neurons
all_upset_data = upset_data.sum(axis=0)
sum(all_upset_data.loc[np.array([x.count('and') + 1 for x in all_upset_data.index])>6])/sum(all_upset_data)

# %%
# matched the order of the upset manually (couldn't find out quickly how to pull the data from the plot)
# *** MANUAL INPUT REQUIRED HERE ***
col_order = [10, 8, 7, 6, 5, 4, 3, 2, 1, 0, 11, 9, 12, 13, 14, 15, 16, 17]
col_name_order = [upset_data.columns[i] for i in col_order]

# %%
# plot upset cell lines
upset_data = upset_data.drop(['sensories', 'ascendings'])
upset_data = upset_data.loc[:, col_name_order]

# add set of miscellaneous integrations that is plotted as the final bar in the upset plot
upset_data['and all other combinations'] = excluded_data

# identify columns of labelled line or integrative categories
labelled_line = ['and' not in x for x in upset_data.columns]
integrative = ['and' in x for x in upset_data.columns]
vmax = 100

#######
# plot labelled line
vmax=100
cell_width=0.25
# set up annotations to put on plot
data = upset_data.loc[:, labelled_line]
annotations = data.astype(str)
annotations[annotations=='0']=''
# generate heatmap
fig, ax = plt.subplots(1,1, figsize=(cell_width * data.shape[0], cell_width * data.shape[1]))
sns.heatmap(data, annot=annotations, fmt='s', square=True, cmap='Blues', vmax=vmax, ax=ax)
plt.savefig(f'plots/sens-cascades-{hops}hops_upset_celltype-members_labelled-line.pdf', format='pdf', bbox_inches='tight')

#######
# plot integrative
# set up annotations to put on plot
data = upset_data.loc[:, integrative]
annotations = data.astype(str)
annotations[annotations=='0']=''
# generate heatmap
fig, ax = plt.subplots(1,1, figsize=(cell_width * data.shape[0], cell_width * data.shape[1]))
sns.heatmap(data, annot=annotations, fmt='s', square=True, cmap='Reds', vmax=vmax, ax=ax)
plt.savefig(f'plots/sens-cascades-{hops}hops_upset_celltype-members_integrative.pdf', format='pdf', bbox_inches='tight')

# %%
# plot hops from input labelled line vs integrative
all_upset_cats = [members_selected[i] for i in col_order] + [Celltype(name='and all other combinations', skids=skids_excluded)]

ll_upset_cats = [all_upset_cats[i] for i, x in enumerate([cat.name for cat in all_upset_cats]) if 'and' not in x]
int_upset_cats = [all_upset_cats[i] for i, x in enumerate([cat.name for cat in all_upset_cats]) if 'and' in x]

hops = 8
# determine median hops from sensory for labelled-line neurons
neuron_data_ll = []
for cat in ll_upset_cats:
    sens_type = cat.get_name()
    modality = [hit_hist for hit_hist in input_hit_hist_list.cascade_objs if sens_type==hit_hist.get_name()][0]
    skid_hit_hist = modality.get_skid_hit_hist()
    for skid in cat.get_skids():
        data = skid_hit_hist.loc[skid, :]

        # convert to counts
        count=[]
        for i in data.index:
            num = int(data.iloc[i])
            if(num>0): count = count + [i]*num

        neuron_data_ll.append([skid, np.median(count), sens_type, 'labelled-line'])

# determine median hops from sensory for integrative neurons
neuron_data_int = []
for cat in int_upset_cats:
    for hit_hist in input_hit_hist_list.cascade_objs:
        skid_hit_hist = hit_hist.get_skid_hit_hist()
        for skid in cat.get_skids():

            # identify pair to check pairwise threshold for each sensory modality
            partner = Promat.identify_pair(skid, pairs)

            # for paired neurons (partner skid != skid for paired neurons)
            if(partner!=skid):
                data = skid_hit_hist.loc[skid, 1:hops+1]
                partner_data = skid_hit_hist.loc[partner, 1:hops+1]
                if((sum(data)+sum(partner_data))>=n_init): # check pairwise threshold again here
                    # convert to counts
                    count=[]
                    for i in data.index:
                        num = int(data.loc[i])
                        if(num>0): count = count + [i]*num

            # for nonpaired neurons
            if(partner==skid):
                data = skid_hit_hist.loc[skid, 1:hops+1]
                if(sum(data)>=n_init/2): # check threshold again here for unpaired
                    # convert to counts
                    count=[]
                    for i in data.index:
                        num = int(data.loc[i])
                        if(num>0): count = count + [i]*num

            neuron_data_int.append([skid, np.median(count), hit_hist.get_name(), 'integrative'])        

df_hops = pd.DataFrame(neuron_data_ll + neuron_data_int, columns=['skid', 'hops', 'sensory', 'type'])
#df_hops = df_hops.set_index(['type', 'skid', 'sensory'])

# how many neurons of each type are there? labelled line vs integrative
ll_count = len(np.unique(df_hops[df_hops.type=='labelled-line'].skid))
int_count = len(np.unique(df_hops[df_hops.type=='integrative'].skid))
identified = list(np.unique(df_hops[df_hops.type=='labelled-line'].skid)) + list(np.unique(df_hops[df_hops.type=='integrative'].skid))

# any missing neurons? who are they?
brain = np.setdiff1d(pymaid.get_skids_by_annotation('mw brain paper clustered neurons'), Celltype_Analyzer.get_skids_from_meta_meta_annotation('mw brain sensory modalities'))
left_over = list(np.setdiff1d(brain, identified))
left_over_analyze = Celltype_Analyzer([Celltype('left_over', left_over)])
left_over_analyze.set_known_types(celltypes)
left_over_data = left_over_analyze.memberships(raw_num=True)

# boxenplot of hops from sens/ascending neurons for integrative vs. labelled line
#df_hops['hops'] = df_hops['hops']-0.5 # makes the boxenplot line up with each hop level
fig, ax = plt.subplots(1,1, figsize=(1,1.5))
sns.boxenplot(data=df_hops, x='type', y='hops', k_depth='full', orient='v', linewidth=0, showfliers=False, ax=ax)
ax.set(ylim=(1, 8), yticks=[1,2,3,4,5,6,7,8])
ax.set_xlabel([f'Labeled Line (N={ll_count})', f'Integrative (N={int_count})'])
plt.savefig('plots/sens-cascades-{hops}_hops-plot.pdf', format='pdf', bbox_inches='tight')
#df_hops['hops'] = df_hops['hops']+0.5 # revert back to normal

# %%
# take the upset categories and make labelled-line / integrative plot of clusters

# %%
# labelled-line / integrative cells in sensory circuits (2nd-order, 3rd-order, 4th-order, etc.)

hop_name = '5-hop'
ll_cells = pymaid.get_skids_by_annotation(f'mw cascade-{hop_name} labeled-line {data_date}')
int_cells = pymaid.get_skids_by_annotation(f'mw cascade-{hop_name} integrative {data_date}')

annots = ['2nd_order', '3rd_order', '4th_order', '5th_order']
cells = [Celltype_Analyzer.get_skids_from_meta_annotation('mw brain inputs ' + annot) for annot in annots]

for i in range(len(annots)):
    print(f'{(len(np.intersect1d(cells[i], int_cells))/len(cells[i]))*100:.1f}% of {annots[i]} cells are integrative')

for i in range(len(annots)):
    print(f'{(len(np.intersect1d(cells[i], ll_cells))/len(cells[i]))*100:.1f}% of {annots[i]} cells are labeled line')

for i in range(len(annots)):
    print(f'{(len(np.setdiff1d(cells[i], ll_cells + int_cells))/len(cells[i]))*100:.1f}% of {annots[i]} cells were neither labeled line nor integrative')

# %%
# labelled-line / integrative cells in 2nd-order sensory circuits by modality

cells_2ndorder = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain inputs 2nd_order', split=True, return_celltypes=True)
for i in range(len(cells_2ndorder)):
    fraction_int = (len(np.intersect1d(cells_2ndorder[i].skids, int_cells))/len(cells_2ndorder[i].skids))
    fraction_ll = (len(np.intersect1d(cells_2ndorder[i].skids, ll_cells))/len(cells_2ndorder[i].skids))
    fraction_neither = (len(np.setdiff1d(cells_2ndorder[i].skids, ll_cells + int_cells))/len(cells_2ndorder[i].skids))

    #print(f'{fraction_int*100:.1f}% of {cells_2ndorder[i].name} were integrative')
    print(f'{fraction_ll*100:.1f}% of {cells_2ndorder[i].name} were labeled line')
    #print(f'{fraction_neither*100:.1f}% of {cells_2ndorder[i].name} were neither')

# %%
# unimodal and multimodal output neurons; generated manually from plot

DN_VNC = [6/172, 166/172]#, 146/172]
DN_SEZ = [48/122, 74/122]#, 44/122]
RGN = [4/32, 28/32]#, 24/32]

DN_VNC = [6/182, 166/182]#, 146/172]
DN_SEZ = [48/184, 74/184]#, 44/122]
RGN = [4/56, 28/56]#, 24/32]

fig, ax = plt.subplots(1,1,figsize=(1,1))
sns.barplot(x=['unimodal', 'multimodal'], y=DN_VNC, ax=ax)
ax.set(ylim=(0,1))
plt.savefig('plots/unimodal-multimodal_DN-VNCs.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1,figsize=(1,1))
sns.barplot(x=['unimodal', 'multimodal'], y=DN_SEZ, ax=ax)
ax.set(ylim=(0,1))
plt.savefig('plots/unimodal-multimodal_DN-SEZs.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1,figsize=(1,1))
sns.barplot(x=['unimodal', 'multimodal'], y=RGN, ax=ax)
ax.set(ylim=(0,1))
plt.savefig('plots/unimodal-multimodal_RGNs.pdf', format='pdf', bbox_inches='tight')
# %%
