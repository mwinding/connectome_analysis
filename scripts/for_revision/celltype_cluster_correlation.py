# %%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from pymaid_creds import url, name, password, token
from data_settings import data_date, pairs_path
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

from contools import Celltype, Celltype_Analyzer, Promat

_, celltypes = Celltype_Analyzer.default_celltypes()

cluster_level = 7
clusters_cts = Celltype_Analyzer.get_skids_from_meta_annotation(f'mw brain clusters level {cluster_level}', split=True, return_celltypes=True)
clusters_cta = Celltype_Analyzer(clusters_cts)

# %%
# determine number of each celltype in each cluster

data = []
for celltype in celltypes:
    for skid in celltype.skids:
        for cluster in clusters_cts:
            if(skid in cluster.skids):
                data.append([skid, celltype.name, cluster.name])
            
df = pd.DataFrame(data, columns=['skid', 'celltype', 'cluster'])
crosstab = pd.crosstab(df.cluster, df.celltype)


# %%
# Cramer's V determining correlation between cell type labels and cluster labels

import scipy.stats as stats
import numpy as np
from natsort import natsort_keygen
import natsort as ns

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorical-categorical association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

cramers_v = cramers_corrected_stat(crosstab)

# sort cluster names naturally
index = ns.natsorted(crosstab.index)

fig, ax = plt.subplots(1,1, figsize=(8,8))
sns.heatmap(crosstab.loc[index, [x.name for x in celltypes]], ax=ax, square=True)
plt.savefig('plots/correlation_cluster-celltype.pdf', format='pdf', bbox_inches='tight')

print(f'Correlation between cluster and celltype category is {cramers_v:.3f} (Cramers V correlation)')

# %%
# correlation of SN modality and cluster identity

SN_cts = Celltype_Analyzer.get_skids_from_meta_meta_annotation('mw brain sensory modalities', split=True, return_celltypes=True)
for ct in SN_cts:
    ct.name = ct.name.replace('mw ', '')

SN = np.unique([x for sublist in [ct.get_skids() for ct in SN_cts] for x in sublist])
clusters_cta = Celltype_Analyzer(clusters_cts)
clusters_cta.set_known_types([Celltype('SN', SN)])

SN_membership = clusters_cta.memberships(raw_num=True)
SN_membership.columns = [x.name for x in clusters_cta.Celltypes]

data = []
for SN_ct in SN_cts:
    for skid in SN_ct.skids:
        for cluster in clusters_cts:
            if(skid in cluster.skids):
                data.append([skid, SN_ct.name, cluster.name])
            
df = pd.DataFrame(data, columns=['skid', 'modality', 'cluster'])
crosstab = pd.crosstab(df.cluster, df.modality)

cramers_v = cramers_corrected_stat(crosstab)

# normalize by number of SNs in each category for plotting purposes
counts_per_cluster = SN_membership.loc[:, crosstab.index].loc['SN']

for i, idx in enumerate(crosstab.index):
    crosstab.loc[idx] = crosstab.loc[idx]/counts_per_cluster.loc[idx]

# sort cluster names naturally
index = ns.natsorted(crosstab.index)

fig, ax = plt.subplots(1,1, figsize=(8,8))
sns.heatmap(crosstab.loc[index, ['visual', 'olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'mechano-Ch', 'noci', 'respiratory', 'mechano-II/III', 'proprio', 'thermo-warm', 'thermo-cold']], ax=ax, square=True)
plt.savefig('plots/correlation_cluster-modality.pdf', format='pdf', bbox_inches='tight')

print(f'Correlation between cluster and SN modality category is {cramers_v:.3f} (Cramers V correlation)')

# %%
# same for PNs

SN_cts = Celltype_Analyzer.get_skids_from_meta_annotation('mw brain inputs 2nd_order PN', split=True, return_celltypes=True)
for ct in SN_cts:
    ct.name = ct.name.replace('mw ', '')
    ct.name = ct.name.replace(' 2nd_order PN', '')

SN = np.unique([x for sublist in [ct.get_skids() for ct in SN_cts] for x in sublist])
clusters_cta = Celltype_Analyzer(clusters_cts)
clusters_cta.set_known_types([Celltype('SN', SN)])

SN_membership = clusters_cta.memberships(raw_num=True)
SN_membership.columns = [x.name for x in clusters_cta.Celltypes]

data = []
for SN_ct in SN_cts:
    for skid in SN_ct.skids:
        for cluster in clusters_cts:
            if(skid in cluster.skids):
                data.append([skid, SN_ct.name, cluster.name])
            
df = pd.DataFrame(data, columns=['skid', 'modality', 'cluster'])
crosstab = pd.crosstab(df.cluster, df.modality)

cramers_v = cramers_corrected_stat(crosstab)

# normalize by number of SNs in each category for plotting purposes
counts_per_cluster = SN_membership.loc[:, crosstab.index].loc['SN']

for i, idx in enumerate(crosstab.index):
    crosstab.loc[idx] = crosstab.loc[idx]/counts_per_cluster.loc[idx]

# sort cluster names naturally
index = ns.natsorted(crosstab.index)

fig, ax = plt.subplots(1,1, figsize=(8,8))
sns.heatmap(crosstab.loc[index, ['gustatory-external', 'gustatory-pharyngeal', 'thermo-warm', 'olfactory', 'mechano-Ch', 'noci', 'thermo-cold', 'visual','enteric', 'mechano-II/III', 'respiratory', 'proprio']], ax=ax, square=True)
plt.savefig('plots/correlation_cluster-modality-PNs.pdf', format='pdf', bbox_inches='tight')

print(f'Correlation between cluster and PN modality category is {cramers_v:.3f} (Cramers V correlation)')

# %%
# correlation between KC claw-classification and clusters

cts = Celltype_Analyzer.get_skids_from_meta_annotation('mw KC claws', split=True, return_celltypes=True)
for ct in cts:
    ct.name = ct.name.replace('cutoff', 'KC')

KC = np.unique([x for sublist in [ct.get_skids() for ct in cts] for x in sublist])
clusters_cta = Celltype_Analyzer(clusters_cts)
clusters_cta.set_known_types([Celltype('KC', KC)])

KC_membership = clusters_cta.memberships(raw_num=True)
KC_membership.columns = [x.name for x in clusters_cta.Celltypes]

data = []
for ct in cts:
    for skid in ct.skids:
        for cluster in clusters_cts:
            if(skid in cluster.skids):
                data.append([skid, ct.name, cluster.name])
            
df = pd.DataFrame(data, columns=['skid', 'modality', 'cluster'])
crosstab = pd.crosstab(df.cluster, df.modality)

cramers_v = cramers_corrected_stat(crosstab)

# normalize by number of SNs in each category for plotting purposes
counts_per_cluster = KC_membership.loc[:, crosstab.index].loc['KC']

for i, idx in enumerate(crosstab.index):
    crosstab.loc[idx] = crosstab.loc[idx]/counts_per_cluster.loc[idx]

# sort cluster names naturally
index = ns.natsorted(crosstab.index)

fig, ax = plt.subplots(1,1, figsize=(8,8))
sns.heatmap(crosstab.loc[index, :], ax=ax, square=True)
plt.savefig('plots/correlation_cluster-KC-claws.pdf', format='pdf', bbox_inches='tight')

print(f'Correlation between cluster and KC claw category is {cramers_v:.3f} (Cramers V correlation)')

# %%
