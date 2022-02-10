# %%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymaid_creds import url, name, password, token
import pymaid
rm = pymaid.CatmaidInstance(url, token, name, password)

neurons_of_interest = pymaid.get_skids_by_annotation('mw MB ad in-out hubs')
neurons_of_interest = pd.DataFrame(neurons_of_interest, columns=['ad_in_out_hubs_of_interest'])
neurons_of_interest.to_csv('CATMAID_scripts/csv/ad_in-out-hubs_MB-related.csv', index=False)

