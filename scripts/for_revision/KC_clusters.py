# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
from data_settings import data_date, pairs_path
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

from contools import Celltype, Celltype_Analyzer, Promat, Analyze_Cluster

# KC clusters
annots = ['30_level-7_clusterID-152_signal-flow_0.347',
            '33_level-7_clusterID-154_signal-flow_0.277',
            '34_level-7_clusterID-153_signal-flow_0.239',
            '36_level-7_clusterID-151_signal-flow_0.202',
            '39_level-7_clusterID-47_signal-flow_0.181',
            '40_level-7_clusterID-91_signal-flow_0.179',
            '42_level-7_clusterID-94_signal-flow_0.154',
            '49_level-7_clusterID-50_signal-flow_-0.017']

# KC lineages
lineages = ['Lineage MBNB A', 'Lineage MBNB B', 'Lineage MBNB C', 'Lineage MBNB D']
lineages = [Celltype(annot, pymaid.get_skids_by_annotation(annot)) for annot in lineages]
KC_cts = [Celltype(annot, pymaid.get_skids_by_annotation(annot)) for annot in annots]

KC_cta = Celltype_Analyzer(KC_cts)
KC_cta.set_known_types(lineages)

KC_cta.memberships(raw_num=True)