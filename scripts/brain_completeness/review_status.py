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

rm = pymaid.CatmaidInstance(url, token, name, password)


# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

brain = pymaid.get_skids_by_annotation('mw brain neurons')
review = pymaid.get_review_details(brain)
review_percent = pymaid.get_review(brain)
morphologies = pymaid.get_treenode_table(brain)
#connectors_details = pymaid.get_connector_details(brain)
connectors = pymaid.get_connector_links(brain)
# %%
review.to_csv('brain_completeness/data/review_details.csv')
review_percent.to_csv('brain_completeness/data/review_percent.csv')
morphologies.to_csv('brain_completeness/data/morphologies.csv')
connectors.to_csv('brain_completeness/data/connectors.csv')
#connectors_details.to_csv('brain_completeness/data/connectors_details_2020_6_16.csv')

# %%
# plot review status of all neurons

fig, ax = plt.subplots(1,1,figsize=(6,6))

sns.distplot(review_percent.percent_reviewed.sort_values(ascending=False), bins = 100, ax = ax, hist = True, kde = False, hist_kws=dict(edgecolor=(0, 0, 0, 1), linewidth=1), norm_hist=True)


# %%
# select 50 random 0% reviewed neurons
immature = pymaid.get_skids_by_annotation('mw brain few synapses')
dVNC = pymaid.get_skids_by_annotation('mw dVNC')
dSEZ = pymaid.get_skids_by_annotation('mw dSEZ')

unreviewed = review_percent.skeleton_id[review_percent.percent_reviewed == 0]
unreviewed = list(map(int, unreviewed))
unreviewed = np.setdiff1d(unreviewed, immature)
unreviewed = np.setdiff1d(unreviewed, dVNC)
unreviewed = np.setdiff1d(unreviewed, dSEZ)

random_index = np.random.choice(range(len(unreviewed)), 50, replace=False)
unreviewed_skids = pd.DataFrame(unreviewed[random_index], columns = ['skeleton_id'])

unreviewed_skids[0:10].to_csv('brain_completeness/data/random_skids_10_for_review_.csv')
unreviewed_skids[10:20].to_csv('brain_completeness/data/random_skids_20_for_review_.csv')
unreviewed_skids[20:30].to_csv('brain_completeness/data/random_skids_30_for_review_.csv')
unreviewed_skids[30:40].to_csv('brain_completeness/data/random_skids_40_for_review_.csv')
unreviewed_skids[40:50].to_csv('brain_completeness/data/random_skids_50_for_review_.csv')

# %%
# stats on random neurons
# taken on 2020-6-16
review_50 = pymaid.get_review_details(unreviewed_skids.skeleton_id)
review_percent_50 = pymaid.get_review(unreviewed_skids.skeleton_id)
morphologies_50 = pymaid.get_treenode_table(unreviewed_skids.skeleton_id)
connectors_50 = pymaid.get_connector_links(unreviewed_skids.skeleton_id)
connectors_details_50 = pymaid.get_connector_details(connectors_50.connector_id)

review_50.to_csv('brain_completeness/data/review_50_.csv')
review_percent_50.to_csv('brain_completeness/data/review_percent_50_.csv')
morphologies_50.to_csv('brain_completeness/data/morphologies_50_.csv')
connectors_50.to_csv('brain_completeness/data/connectors_50_.csv')
connectors_details_50.to_csv('brain_completeness/data/connectors_details_50_.csv')
# %%
# stats for reviewed neurons
# add 17380319 next time
skids = [7025831, 17654645, 4901087, 9085808, 10355356, 12928099, 16714214, 9034378, 17384874, 10872426]

review_2 = pymaid.get_review_details(skids)
review_percent_2 = pymaid.get_review(skids)
morphologies_2 = pymaid.get_treenode_table(skids)
connectors_2 = pymaid.get_connector_links(skids)
connectors_details_ = pymaid.get_connector_details(connectors_2)

# %%
# comparison

aa = pd.read_csv('brain_completeness/data/2020_6_16/axon-axon.csv', header = 0, index_col = 0)
ad = pd.read_csv('brain_completeness/data/2020_6_16/axon-dendrite.csv', header = 0, index_col = 0)
dd = pd.read_csv('brain_completeness/data/2020_6_16/dendrite-dendrite.csv', header = 0, index_col = 0)
da = pd.read_csv('brain_completeness/data/2020_6_16/dendrite-axon.csv', header = 0, index_col = 0)
inputs = pd.read_csv('brain_completeness/data/2020_6_16/input_counts.csv', header = 0, index_col = 0)
all_all = aa + ad + dd + da
all_all.columns = list(map(int, all_all.columns))
all_all = all_all.sort_index(axis = 0)
all_all = all_all.sort_index(axis = 1)
'''
all_all_fract = all_all.copy()
for column in all_all_fract.columns:
    input_all = inputs.loc[column, ' axon_inputs'] + inputs.loc[column, ' dendrite_inputs']
    if(input_all == 0):
        all_all_fract.loc[:, column] = 0
    if(input_all > 0):
        all_all_fract.loc[:, column] = all_all_fract.loc[:, column]/input_all
'''
aa2 = pd.read_csv('brain_completeness/data/2021_02_22_reviewed/axon-axon.csv', header = 0, index_col = 0)
ad2 = pd.read_csv('brain_completeness/data/2021_02_22_reviewed/axon-dendrite.csv', header = 0, index_col = 0)
dd2 = pd.read_csv('brain_completeness/data/2021_02_22_reviewed/dendrite-dendrite.csv', header = 0, index_col = 0)
da2 = pd.read_csv('brain_completeness/data/2021_02_22_reviewed/dendrite-axon.csv', header = 0, index_col = 0)
inputs2 = pd.read_csv('brain_completeness/data/2021_02_22_reviewed/input_counts.csv', header = 0, index_col = 0)
all_all2 = aa2 + ad2 + dd2 + da2
all_all2.columns = list(map(int, all_all2.columns))
all_all2 = all_all2.sort_index(axis = 0)
all_all2 = all_all2.sort_index(axis = 1)
'''
all_all2_fract = all_all2.copy()
for column in all_all2_fract.columns:
    input_all = inputs2.loc[column, ' axon_inputs'] + inputs2.loc[column, ' dendrite_inputs']
    if(input_all == 0):
        all_all2_fract.loc[:, column] = 0
    if(input_all > 0):
        all_all2_fract.loc[:, column] = all_all2_fract.loc[:, column]/input_all
'''
intersect = np.intersect1d(all_all.columns, all_all2.columns)

test_before_pre = all_all.loc[skids, intersect]
test_after_pre = all_all2.loc[skids, intersect]

test_before_post = all_all.loc[intersect, skids]
test_after_post = all_all2.loc[intersect, skids]

test_before_pre = np.concatenate(test_before_pre.values)
test_after_pre = np.concatenate(test_after_pre.values)

test_before_post = np.concatenate(test_before_post.values)
test_after_post = np.concatenate(test_after_post.values)

'''
# for input fraction version
test_before_pre_fract = all_all_fract.loc[skids, intersect]
test_after_pre_fract = all_all2_fract.loc[skids, intersect]

test_before_post_fract = all_all_fract.loc[intersect, skids]
test_after_post_fract = all_all2_fract.loc[intersect, skids]

test_before_pre_fract = np.concatenate(test_before_pre_fract.values)
test_after_pre_fract = np.concatenate(test_after_pre_fract.values)

test_before_post_fract = np.concatenate(test_before_post_fract.values)
test_after_post_fract = np.concatenate(test_after_post_fract.values)
'''
# %%
fig, ax = plt.subplots(1,1,figsize=(4, 4))

rand_x = np.random.random(size = len(test_before_post))/4
rand_y = np.random.random(size = len(test_before_post))/4

ax.set(xlim = [0, 10], ylim = [0, 10])
sns.scatterplot(x = test_before_post + rand_x, y = test_after_post + rand_y, alpha = 1, edgecolor="none", s = 4)

# plot input fraction change
fig, ax = plt.subplots(1,1,figsize=(4, 4))
ax.set(xlim = [0, .2], ylim = [0, .2])
sns.scatterplot(x = test_before_post_fract, y = test_after_post_fract, alpha = 1, edgecolor="none", s = 4)
# %%
fig, ax = plt.subplots(1,1,figsize=(4, 4))

ax.set(xlim = [0, 10], ylim = [0, 10])
sns.scatterplot(x = test_before_pre + rand_x, y = test_after_pre + rand_y, alpha = 1, edgecolor="none", s = 4)

# plot input fraction change
fig, ax = plt.subplots(1,1,figsize=(4, 4))
ax.set(xlim = [0, .2], ylim = [0, .2])
sns.scatterplot(x = test_before_pre_fract, y = test_after_pre_fract, alpha = 1, edgecolor="none", s = 4)
# %%
# percent edges that changed/stayed same

def test_edges(before, after, edge_type):
    test = after-before
    edges = []
    for i, value in enumerate(test):
        if((after[i]>0) | (before[i]>0)):
            new_edge = 0
            false_edge = 0
            if((after[i]>0) & (before[i]==0)):
                new_edge = 1
            if((after[i]==0) & (before[i]>0)):
                false_edge = 1

            if(value==0):
                edges.append([edge_type, before[i], after[i], value, value/before[i],'no change', new_edge, false_edge])
            if(value>0):
                edges.append([edge_type, before[i], after[i], value, value/before[i], 'increase', new_edge, false_edge])
            if(value<0):
                edges.append([edge_type, before[i], after[i], value, value/before[i], 'decrease', new_edge, false_edge])

    return(edges)

post = test_edges(test_before_post, test_after_post, 'post')
pre = test_edges(test_before_pre, test_after_pre, 'pre')

post = pd.DataFrame(post, columns = ['edge_type', 'before', 'after', 'diff', 'fold_change', 'review_effect', 'new_edge', 'false_edge'])
pre = pd.DataFrame(pre, columns = ['edge_type', 'before', 'after', 'diff', 'fold_change', 'review_effect', 'new_edge', 'false_edge'])

sum(post.new_edge)/len(post.new_edge)
sum(post.false_edge)/len(post.false_edge)

edges = pd.concat([post, pre], axis=0)
edges.groupby(['edge_type', 'review_effect']).count()

# plots
palette ={"increase": sns.color_palette()[1], "decrease": sns.color_palette()[2], "no change": sns.color_palette()[0]}
alpha = 1
size = 2

edges_plot = edges[edges.edge_type=='post']
fig, ax = plt.subplots(1,1,figsize=(4, 4))
rand_x = np.random.random(size = len(edges_plot.before))/4
rand_y = np.random.random(size = len(edges_plot.before))/4
sns.scatterplot(data = edges_plot, x=edges_plot.before + rand_x, y=edges_plot.after + rand_y, hue='review_effect', palette=palette, ax=ax, alpha = alpha, edgecolor="none", s = size, legend=False)
ax.set(xlim = [-0.25, 15], ylim = [-0.25, 15], xticks=([0,5,10,15]), yticks=([0,5,10,15]))
plt.savefig('brain_completeness/plots/review_post-comparison.pdf', format='pdf', bbox_inches = 'tight')

edges_plot = edges[edges.edge_type=='pre']
fig, ax = plt.subplots(1,1,figsize=(4, 4))
rand_x = np.random.random(size = len(edges_plot.before))/4
rand_y = np.random.random(size = len(edges_plot.before))/4
sns.scatterplot(data = edges_plot, x=edges_plot.before + rand_x, y=edges_plot.after + rand_y, hue='review_effect', palette=palette, ax=ax, alpha = alpha, edgecolor="none", s = size, legend=False)
ax.set(xlim = [-0.25, 15], ylim = [-0.25, 15], xticks=([0,5,10,15]), yticks=([0,5,10,15]))
plt.savefig('brain_completeness/plots/review_pre-comparison.pdf', format='pdf', bbox_inches = 'tight')

edges_plot = edges
fig, ax = plt.subplots(1,1,figsize=(4, 4))
rand_x = np.random.random(size = len(edges_plot.before))/4
rand_y = np.random.random(size = len(edges_plot.before))/4
sns.scatterplot(data = edges_plot, x=edges_plot.before + rand_x, y=edges_plot.after + rand_y, hue='review_effect', palette=palette, ax=ax, alpha = alpha, edgecolor="none", s = size, legend=False)
ax.set(xlim = [-0.25, 15], ylim = [-0.25, 15], xticks=([0,5,10,15]), yticks=([0,5,10,15]))
plt.savefig('brain_completeness/plots/review_both-comparison.pdf', format='pdf', bbox_inches = 'tight')


effect = pd.DataFrame(list(zip(edges.groupby(['edge_type', 'review_effect']).count().iloc[:, 0], [x[0] for x in edges.groupby(['edge_type', 'review_effect']).count().index], [x[1] for x in edges.groupby(['edge_type', 'review_effect']).count().index])),
                columns = ['counts', 'edge_type', 'change'])

fig, ax = plt.subplots(1,1,figsize=(4, 4))
sns.barplot(data = effect, x = 'edge_type', y='counts', hue='change', palette=palette, ax=ax)
plt.savefig('brain_completeness/plots/review_edges_new-and-false.pdf', format='pdf', bbox_inches = 'tight')

#sns.distplot(edges[edges.new_edge!=1].fold_change, kde=0, bins=40)
# %%
