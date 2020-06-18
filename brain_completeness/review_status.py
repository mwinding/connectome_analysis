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

rm = pymaid.CatmaidInstance(url, name, password, token)

brain = pymaid.get_skids_by_annotation('mw brain neurons')
review = pymaid.get_review_details(brain)
review_percent = pymaid.get_review(brain)
morphologies = pymaid.get_treenode_table(brain)
#connectors_details = pymaid.get_connector_details(brain)
connectors = pymaid.get_connector_links(brain)
# %%
review.to_csv('brain_completeness/data/review_details_2020_6_16.csv')
review_percent.to_csv('brain_completeness/data/review_percent_2020_6_16.csv')
morphologies.to_csv('brain_completeness/data/morphologies_2020_6_16.csv')
connectors.to_csv('brain_completeness/data/connectors_2020_6_16.csv')
#connectors_details.to_csv('brain_completeness/data/connectors_details_2020_6_16.csv')

# %%
# plot review status of all neurons

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

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
# stats for first reviewed neurons
# 17654645
# 7025831

skids = [7025831, 17654645]

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
all_all = aa + ad + dd + da
all_all.columns = list(map(int, all_all.columns))
all_all = all_all.sort_index(axis = 0)
all_all = all_all.sort_index(axis = 1)

aa2 = pd.read_csv('brain_completeness/data/2020_6_16_2reviewed/axon-axon.csv', header = 0, index_col = 0)
ad2 = pd.read_csv('brain_completeness/data/2020_6_16_2reviewed/axon-dendrite.csv', header = 0, index_col = 0)
dd2 = pd.read_csv('brain_completeness/data/2020_6_16_2reviewed/dendrite-dendrite.csv', header = 0, index_col = 0)
da2 = pd.read_csv('brain_completeness/data/2020_6_16_2reviewed/dendrite-axon.csv', header = 0, index_col = 0)
all_all2 = aa2 + ad2 + dd2 + da2
all_all2.columns = list(map(int, all_all2.columns))
all_all2 = all_all2.sort_index(axis = 0)
all_all2 = all_all2.sort_index(axis = 1)

test_before_pre = all_all.loc[skids, :]
test_after_pre = all_all2.loc[skids, :]

test_before_post = all_all.loc[:, skids]
test_after_post = all_all2.loc[:, skids]

test_before_pre = np.concatenate((np.array(test_before_pre.iloc[0, :]), np.array(test_before_pre.iloc[1, :])))
test_after_pre = np.concatenate((np.array(test_after_pre.iloc[0, :]), np.array(test_after_pre.iloc[1, :])))

test_before_post = np.concatenate((np.array(test_before_post.iloc[:, 0]), np.array(test_before_post.iloc[:, 1])))
test_after_post = np.concatenate((np.array(test_after_post.iloc[:, 0]), np.array(test_after_post.iloc[:, 1])))

# %%
fig, ax = plt.subplots(1,1,figsize=(4, 4))

rand_x = np.random.random(size = len(test_before_post))/4
rand_y = np.random.random(size = len(test_before_post))/4

ax.set(xlim = [0, 10], ylim = [0, 10])
sns.scatterplot(x = test_before_post + rand_x, y = test_after_post + rand_y, alpha = 1, edgecolor="none", s = 4)
# %%
fig, ax = plt.subplots(1,1,figsize=(4, 4))

ax.set(xlim = [0, 10], ylim = [0, 10])
sns.scatterplot(x = test_before_pre + rand_x, y = test_after_pre + rand_y, alpha = 1, edgecolor="none", s = 4)

# %%
