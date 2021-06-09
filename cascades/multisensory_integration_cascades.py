#%%
import sys
import os

os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory
sys.path.append('/Users/mwinding/repos/maggot_models')

from pymaid_creds import url, name, password, token
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

from src.data import load_metagraph
from src.visualization import CLASS_COLOR_DICT, adjplot
from src.traverse import Cascade, to_transmission_matrix
from src.traverse import TraverseDispatcher
from src.visualization import matrixplot

import connectome_tools.cascade_analysis as casc
import connectome_tools.celltype as ct
import connectome_tools.process_matrix as pm

adj_ad = pm.Promat.pull_adj(type_adj='ad', subgraph='brain')

#%%
# pull sensory annotations and then pull associated skids
order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-warm', 'thermo-cold', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']
sens = [ct.Celltype(name, ct.Celltype_Analyzer.get_skids_from_meta_annotation(f'mw {name}')) for name in order]
input_skids_list = [x.get_skids() for x in sens]
input_skids = [val for sublist in input_skids_list for val in sublist]

output_names = pymaid.get_annotated('mw brain outputs').name
output_skids_list = list(map(pymaid.get_skids_by_annotation, pymaid.get_annotated('mw brain outputs').name))
output_skids = [val for sublist in output_skids_list for val in sublist]

#%%
# cascades from each sensory modality
# save as pickle to use later because cascades are stochastic; prevents the need to remake plots everytime
import pickle

p = 0.05
max_hops = 10
n_init = 1000
simultaneous = True
adj=adj_ad
'''
input_hit_hist_list = casc.Cascade_Analyzer.run_cascades_parallel(source_skids_list=input_skids_list, source_names = order, stop_skids=output_skids, 
                                                                    adj=adj_ad, p=p, max_hops=max_hops, n_init=n_init, simultaneous=simultaneous)

pickle.dump(input_hit_hist_list, open('data/cascades/sensory-modality-cascades_1000-n_init.p', 'wb'))
'''
input_hit_hist_list = pickle.load(open('data/cascades/sensory-modality-cascades_1000-n_init.p', 'rb'))
# %%
# pairwise thresholding of hit_hists

threshold = n_init/2
hops = 8

sens_types = [ct.Celltype(hit_hist.get_name(), list(hit_hist.pairwise_threshold(threshold=threshold, hops=hops))) for hit_hist in input_hit_hist_list]
sens_types_analyzer = ct.Celltype_Analyzer(sens_types)

upset_threshold = 30
upset_threshold_dual_cats = 10
path = f'cascades/plots/sens-cascades_upset'
upset_members, members_selected, skids_excluded = sens_types_analyzer.upset_members(threshold=upset_threshold, path=path, plot_upset=True, 
                                                                                    exclude_singletons_from_threshold=True, exclude_skids=input_skids, threshold_dual_cats=upset_threshold_dual_cats)
# check what's in the skids_excluded group
# this group is assorted integrative that were excluded from the plot for simplicity; added manually as "other combinations" at far right of plot
_, celltypes = ct.Celltype_Analyzer.default_celltypes()
test_excluded = ct.Celltype_Analyzer([ct.Celltype('excluded', skids_excluded)])
test_excluded.set_known_types(celltypes)
excluded_data = test_excluded.memberships(raw_num=True).drop(['sensories', 'ascendings'])

# cell identity of all of these categories
upset_analyzer = ct.Celltype_Analyzer(members_selected)
upset_analyzer.set_known_types(celltypes)
upset_data = upset_analyzer.memberships(raw_num=True) # the data is all out of order compared to upset plot

# %%
# matched the order of the upset manually (couldn't find out quickly how to pull the data from the plot)
# *** MANUAL INPUT REQUIRED HERE ***
col_order = [10, 8, 7, 6, 5, 4, 3, 2, 1, 0, 11, 9, 12, 13, 14, 15, 16]
col_name_order = [upset_data.columns[i] for i in col_order]

# %%
# plot upset cell lines
upset_data = upset_data.loc[:, col_name_order].drop(['sensories', 'ascendings'])

# add set of miscellaneous integrations that is plotted as the final bar in the upset plot
upset_data['+ all other combinations'] = excluded_data

# identify columns of labelled line or integrative categories
labelled_line = ['+' not in x for x in upset_data.columns]
integrative = ['+' in x for x in upset_data.columns]
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
plt.savefig('cascades/plots/sens-cascades_upset_celltype-members_labelled-line.pdf', format='pdf', bbox_inches='tight')

#######
# plot integrative
# set up annotations to put on plot
data = upset_data.loc[:, integrative]
annotations = data.astype(str)
annotations[annotations=='0']=''
# generate heatmap
fig, ax = plt.subplots(1,1, figsize=(cell_width * data.shape[0], cell_width * data.shape[1]))
sns.heatmap(data, annot=annotations, fmt='s', square=True, cmap='Reds', vmax=vmax, ax=ax)
plt.savefig('cascades/plots/sens-cascades_upset_celltype-members_integrative.pdf', format='pdf', bbox_inches='tight')

# %%
# plot hops from input labelled line vs integrative
all_upset_cats = [members_selected[i] for i in col_order] + [ct.Celltype(name='+ all other combinations', skids=skids_excluded)]

ll_upset_cats = [all_upset_cats[i] for i, x in enumerate(labelled_line) if x==True]
int_upset_cats = [all_upset_cats[i] for i, x in enumerate(integrative) if x==True]

hops = 8
# determine median hops from sensory for labelled-line neurons
neuron_data_ll = []
for cat in ll_upset_cats:
    sens_type = cat.get_name()
    modality = [hit_hist for hit_hist in input_hit_hist_list if sens_type==hit_hist.get_name()][0]
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
pairList = pm.Promat.get_pairs()
for cat in int_upset_cats:
    for hit_hist in input_hit_hist_list:
        skid_hit_hist = hit_hist.get_skid_hit_hist()
        for skid in cat.get_skids():

            # identify pair to check pairwise threshold for each sensory modality
            partner = pm.Promat.identify_pair(skid, pairList)

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
brain = np.setdiff1d(pymaid.get_skids_by_annotation('mw brain paper clustered neurons'), ct.Celltype_Analyzer.get_skids_from_meta_meta_annotation('mw brain sensory modalities'))
left_over = list(np.setdiff1d(brain, identified))
left_over_analyze = ct.Celltype_Analyzer([ct.Celltype('left_over', left_over)])
left_over_analyze.set_known_types(celltypes)
left_over_data = left_over_analyze.memberships(raw_num=True)

# boxenplot of hops from sens/ascending neurons for integrative vs. labelled line
#df_hops['hops'] = df_hops['hops']-0.5 # makes the boxenplot line up with each hop level
fig, ax = plt.subplots(1,1, figsize=(1,1.5))
sns.boxenplot(data=df_hops, x='type', y='hops', k_depth='full', orient='v', linewidth=0, showfliers=False, ax=ax)
ax.set(ylim=(1, 9), yticks=[1,2,3,4,5,6,7,8])
ax.set_xlabel([f'Labeled Line (N={ll_count})', f'Integrative (N={int_count})'])
plt.savefig('cascades/plots/sens-cascades_hops-plot.pdf', format='pdf', bbox_inches='tight')
#df_hops['hops'] = df_hops['hops']+0.5 # revert back to normal

# %%
# take the upset categories and make labelled-line / integrative plot of clusters

# %%
# labelled-line / integrative plot of sensory neuropils