#%%
import os
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/maggot_models/')
    print(os.getcwd())
except:
    pass

# %%
import matplotlib.pyplot as plt
import seaborn as sns

from graspy.plot import gridplot, heatmap
from graspy.utils import binarize, pass_to_ranks
from src.data import load_metagraph
from src.visualization import CLASS_COLOR_DICT, adjplot

sns.set_context("talk")

mg = load_metagraph("G", version="2020-04-23", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg.calculate_degrees(inplace=True)

adj = mg.adj  # adjacency matrix from the "mg" object
meta = mg.meta  # dataframe of node metadata


mg_ad = load_metagraph("Gad", version="2020-04-23", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg_aa = load_metagraph("Gaa", version="2020-04-23", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg_dd = load_metagraph("Gdd", version="2020-04-23", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg_da = load_metagraph("Gda", version="2020-04-23", path = '/Volumes/GoogleDrive/My Drive/python_code/maggot_models/data/processed/')
mg_ad.calculate_degrees(inplace=True)
mg_aa.calculate_degrees(inplace=True)
mg_dd.calculate_degrees(inplace=True)
mg_da.calculate_degrees(inplace=True)


adj_ad = mg_ad.adj  # adjacency matrix from the "mg" object
adj_aa = mg_aa.adj
adj_dd = mg_dd.adj
adj_da = mg_da.adj

meta_ad = mg_ad.meta  # dataframe of node metadata
meta_aa = mg_aa.meta
meta_dd = mg_dd.meta
meta_da = mg_da.meta
# %%
# ipsilateral / contralateral plot

fig, ax = plt.subplots(1, 1, figsize=(15, 15))
adjplot(
    adj,
    meta=meta,
    sort_class="hemisphere",  # group by hemisphere, this is a key for column in "meta"
    plot_type="scattermap",  # plot dots instead of a heatmap
    sizes=(1, 1),  # min and max sizes for dots, so this is effectively binarizing
    item_order="Pair ID",  # order by pairs (some have no pair here so don't look same)
    ax=ax,
)

# %%
# 4-color matrix
#sns.set_palette(sns.color_palette('bright'))

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

fig, ax = plt.subplots(1, 1, figsize=(15, 15))
adjplot(
    adj_ad,
    meta=meta_ad,
    sort_class="hemisphere",  # group by hemisphere, this is a key for column in "meta"
    plot_type="scattermap",  # plot dots instead of a heatmap
    sizes=(1, 1),  # min and max sizes for dots, so this is effectively binarizing
    item_order="Pair ID",  # order by pairs (some have no pair here so don't look same)
    ax=ax
)

adjplot(
    adj_aa,
    meta=meta_aa,
    sort_class="hemisphere",  # group by hemisphere, this is a key for column in "meta"
    plot_type="scattermap",  # plot dots instead of a heatmap
    sizes=(1, 1),  # min and max sizes for dots, so this is effectively binarizing
    item_order="Pair ID",  # order by pairs (some have no pair here so don't look same)
    ax=ax,
    color = sns.color_palette('bright')[1]
)

adjplot(
    adj_dd,
    meta=meta_dd,
    sort_class="hemisphere",  # group by hemisphere, this is a key for column in "meta"
    plot_type="scattermap",  # plot dots instead of a heatmap
    sizes=(1, 1),  # min and max sizes for dots, so this is effectively binarizing
    item_order="Pair ID",  # order by pairs (some have no pair here so don't look same)
    ax=ax,
    color = sns.color_palette('bright')[2]
)

adjplot(
    adj_da,
    meta=meta_da,
    sort_class="hemisphere",  # group by hemisphere, this is a key for column in "meta"
    plot_type="scattermap",  # plot dots instead of a heatmap
    sizes=(1, 1),  # min and max sizes for dots, so this is effectively binarizing
    item_order="Pair ID",  # order by pairs (some have no pair here so don't look same)
    ax=ax,
    color = sns.color_palette('bright')[3]
)

plt.savefig('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/adjacency_matrix_plots/plots/4color.pdf', format='pdf', bbox_inches='tight')

# %%
# ad
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
adjplot(
    adj_ad,
    meta=meta_ad,
    sort_class="hemisphere",  # group by hemisphere, this is a key for column in "meta"
    plot_type="scattermap",  # plot dots instead of a heatmap
    sizes=(1, 1),  # min and max sizes for dots, so this is effectively binarizing
    item_order="Pair ID",  # order by pairs (some have no pair here so don't look same)
    ax=ax,
    color = sns.color_palette()[0]
)

plt.savefig('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/adjacency_matrix_plots/plots/ad.pdf', format='pdf', bbox_inches='tight')

# %%
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
adjplot(
    adj_aa,
    meta=meta_aa,
    sort_class="hemisphere",  # group by hemisphere, this is a key for column in "meta"
    plot_type="scattermap",  # plot dots instead of a heatmap
    sizes=(1, 1),  # min and max sizes for dots, so this is effectively binarizing
    item_order="Pair ID",  # order by pairs (some have no pair here so don't look same)
    ax=ax,
    color = sns.color_palette()[1]
)

plt.savefig('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/adjacency_matrix_plots/plots/aa.pdf', format='pdf', bbox_inches='tight')

# %%
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
adjplot(
    adj_dd,
    meta=meta_dd,
    sort_class="hemisphere",  # group by hemisphere, this is a key for column in "meta"
    plot_type="scattermap",  # plot dots instead of a heatmap
    sizes=(1, 1),  # min and max sizes for dots, so this is effectively binarizing
    item_order="Pair ID",  # order by pairs (some have no pair here so don't look same)
    ax=ax,
    color = sns.color_palette()[2]
)

plt.savefig('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/adjacency_matrix_plots/plots/dd.pdf', format='pdf', bbox_inches='tight')


# %%
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
adjplot(
    adj_da,
    meta=meta_da,
    sort_class="hemisphere",  # group by hemisphere, this is a key for column in "meta"
    plot_type="scattermap",  # plot dots instead of a heatmap
    sizes=(1, 1),  # min and max sizes for dots, so this is effectively binarizing
    item_order="Pair ID",  # order by pairs (some have no pair here so don't look same)
    ax=ax,
    color = sns.color_palette()[3]
)

plt.savefig('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/adjacency_matrix_plots/plots/da.pdf', format='pdf', bbox_inches='tight')

# %%
