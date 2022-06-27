# %%

from pymaid_creds import url, name, password, token
from data_settings import pairs_path, data_date
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import cmasher as cmr

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

n_init = 1000
cascades_df = pickle.load(open(f'data/cascades/all-brain-pairs-nonpaired_inputs-interneurons-outputs_{n_init}-n_init_{data_date}.p', 'rb'))
cascades_df_with_inputs = pickle.load(open(f'data/cascades/all-brain-pairs-nonpaired_inputs-interneurons-outputs_{n_init}-n_init_{data_date}.p', 'rb'))

# exclude input neurons, include only brain and accessory neurons
brain = pymaid.get_skids_by_annotation(['mw brain neurons', 'mw brain accessory neurons'])
brain = list(np.intersect1d(brain, cascades_df.index))
cascades_df = cascades_df.loc[brain, :]

pairs = Promat.get_pairs(pairs_path=pairs_path)

# %%
# how many partners are in recurrent loops?

def generate_us_df(partners_df, hops):
    skids = partners_df.index

    us_partners_list = []
    for source in skids:
        us_partners = []
        for skid in skids:
            ds = list(partners_df.loc[skid, f'ds_partners_{hops}hop'])
            if(source in ds):
                us_partners.append(skid)

        us_partners_list.append(us_partners)

    return(us_partners_list)

def identify_recurrent_partners(partners_df, hops):
    skids = partners_df.index

    recurrent_partners = []
    for skid in skids:
        ds = list(partners_df.loc[skid, f'ds_partners_{hops}hop'])
        us = list(partners_df.loc[skid, f'us_partners_{hops}hop'])
        recurrent = list(np.intersect1d(ds, us))
        recurrent = list(np.unique(recurrent))
        recurrent_partners.append(recurrent)

    return(recurrent_partners)

partners_df = cascades_df_with_inputs.loc[:, ['ds_partners_5hop']]

# convert to pair_ids; did this as a stop-gap because expanding the recurrent partners took too long on my current compute
partners_df['ds_partners_5hop'] = [list(np.intersect1d(x, partners_df.index)) for x in partners_df.ds_partners_5hop]

# identify upstream neurons, takes a few minutes
partners_df['us_partners_5hop'] = generate_us_df(partners_df, hops=5)

# identify recurrent partners
partners_df['recurrent_partners_5hop'] = identify_recurrent_partners(partners_df, hops=5)

# fraction of recurrent vs. nonrecurrent parents at 5-hop
hops = 5
frac_recurrent = [len(partners_df.loc[i, f'recurrent_partners_{hops}hop'])/len(partners_df.loc[i, f'ds_partners_{hops}hop']) if len(partners_df.loc[i, f'ds_partners_{hops}hop'])>0 else 0 for i in partners_df.index]
partners_df[f'fraction_recurrent_{hops}hop'] = frac_recurrent
partners_df[f'fraction_nonrecurrent_{hops}hop'] = 1-partners_df.loc[:, f'fraction_recurrent_{hops}hop']

# %%
# where does recurrent signal come from? when celltypes?

# manually added DANs, OANs, other-MBINs in alphabetical order (using pair_ids)
MBIN_order = [15592096, 3886356, 7901791, 4381377, 7057894, 4414163, 12871993, 17068730, 3813487, 7983899, 11525714, 8689674, 17295912, 14541927, 12475432, 4381129]

MBIN_partners = partners_df.loc[MBIN_order].copy()

# add back left/right neurons for recurrent_partners_5hop (currently pair_ids)
MBIN_partners.loc[:, 'recurrent_partners_5hop'] = [Promat.get_paired_skids(skids, pairs, unlist=True) for skids in list(MBIN_partners.recurrent_partners_5hop)]

MBIN_rec_partner_cts = [Celltype(f'{MBIN_partners.index[i]}_recurrent-partners', skids) for i, skids in enumerate(list(MBIN_partners.recurrent_partners_5hop))]
MBIN_rec_partner_cta = Celltype_Analyzer(MBIN_rec_partner_cts)

_, celltypes = Celltype_Analyzer.default_celltypes()
MBIN_rec_partner_cta.set_known_types(celltypes)
MBIN_rec_partner_cta.memberships()
MBIN_rec_partner_cta.plot_memberships(path='plots/MBIN_recurrent-partners-5hops.pdf', figsize=(3,1))
MBIN_rec_partner_cta.plot_memberships(path='plots/MBIN_recurrent-partners-5hops_raw-num.pdf', figsize=(3,1), raw_num=True)

# %%
# how many sensory neurons from each modality talk to MBINs within 5-hops?

sens_order = ['olfactory', 'gustatory-external', 'gustatory-pharyngeal', 'enteric', 'thermo-warm', 'thermo-cold', 'visual', 'noci', 'mechano-Ch', 'mechano-II/III', 'proprio', 'respiratory']
sens_color = ['#17A55C', '#8DC048', '#D0D823', '#DBA933', '#E92930', '#3BA6D3', '#5C62AC', '#E05E35', '#007D70', '#48B191', '#DF4F9B', '#652D90']
sens_modalities = [Celltype(name, Celltype_Analyzer.get_skids_from_meta_annotation(f'mw {name}'), sens_color[i]) for i, name in enumerate(sens_order)]

# add back left/right neurons for us_partners_5hop (currently pair_ids)
MBIN_partners.loc[:, 'us_partners_5hop'] = [Promat.get_paired_skids(skids, pairs, unlist=True) for skids in list(MBIN_partners.us_partners_5hop)]

MBIN_us_partner_cts = [Celltype(f'{MBIN_partners.index[i]}_us-partners', skids) for i, skids in enumerate(list(MBIN_partners.us_partners_5hop))]
MBIN_us_partner_cta = Celltype_Analyzer(MBIN_us_partner_cts)
MBIN_us_partner_cta.set_known_types(sens_modalities, unknown=False)
MBIN_us_partner_cta.memberships()
MBIN_us_partner_cta.plot_memberships(path='plots/MBIN_us-sens-partners-5hops.pdf', figsize=(3,1))
MBIN_us_partner_cta.plot_memberships(path='plots/MBIN_us-sens-partners-5hops_raw-num.pdf', figsize=(3,1), raw_num=True)

# convert to fraction of each sensory modality
fraction_sens = MBIN_us_partner_cta.memberships(raw_num=True)

for i in range(len(fraction_sens.index)):
    fraction_sens.iloc[i, :] = fraction_sens.iloc[i, :]/len(sens_modalities[i].skids)

# plot data as heatmap, number of upstream sens
fig, ax = plt.subplots(figsize=(2,2))
sns.heatmap(MBIN_us_partner_cta.memberships(raw_num=True), square=True, cmap='Blues', annot=True)
plt.savefig('plots/MBIN-input_fraction-sens-modality_heatmap.pdf', format='pdf', bbox_inches='tight')

# plot data as heatmap, fraction of each sensory type that is upstream
fig, ax = plt.subplots(figsize=(2,2))
sns.heatmap(fraction_sens, square=True)
plt.savefig('plots/MBIN-input_fraction-sens-modality_heatmap-fraction-sens.pdf', format='pdf', bbox_inches='tight')


# convert to more standard dataframe
df = []
for i in range(len(fraction_sens.columns)):
    df = df + list(zip(list(fraction_sens.iloc[:, i]), [fraction_sens.columns[i]]*len(fraction_sens.index), list(fraction_sens.index)))

df = pd.DataFrame(df, columns=['fraction_sens', 'MBIN', 'sens'])

# plot data as multi-stack barplot
fig, ax = plt.subplots(figsize=(6,2))
sns.barplot(data=df, x='MBIN', y='fraction_sens', hue='sens', palette=[x.color for x in sens_modalities], ax=ax)
plt.xticks(rotation=45, ha='right')
plt.savefig('plots/MBIN-input_fraction-sens-modality.pdf', format='pdf', bbox_inches='tight')

# %%
# cascades from sensory modalities (all neurons at once)

# load cascades from pickle object; generated in generate_data/cascades_all-modalities.py
n_init = 1000
all_data_df = pickle.load(open(f'data/cascades/all-sensory-modalities_processed-cascades_{n_init}-n_init_{data_date}.p', 'rb'))

# load pairs
pairs = Promat.get_pairs(pairs_path=pairs_path)

# find neurons downstream of each modality
threshold = n_init/2
hops = 5
ds_partners_5hop = Parallel(n_jobs=-1)(delayed(Cascade_Analyzer.pairwise_threshold)(hh_pairwise=all_data_df.iloc[i, 1].hh_pairwise, pairs=pairs, threshold=threshold, hops=hops) for i in tqdm(range(len(all_data_df.index))))

# %%
# make heatmap of sensory modalities to DANs

hops = 5

MBINs_sens_input = [list(all_data_df.cascade_objs[i].hh_pairwise.loc[(slice(None), MBIN_order), 0:hops].sum(axis=1)) for i in range(len(all_data_df.cascade_objs))]
MBINs_sens_input = pd.DataFrame(MBINs_sens_input, index=sens_order, columns = MBIN_order)

fig,ax = plt.subplots(figsize=(2,2))
sns.heatmap(MBINs_sens_input, square=True, ax=ax, vmin=0)
plt.savefig('plots/MBIN-input_casc-sens-modality.pdf', format='pdf', bbox_inches='tight')
