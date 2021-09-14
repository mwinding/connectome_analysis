#%%
import os
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory

import pandas as pd
import numpy as np
from pymaid_creds import url, name, password, token
import pymaid


rm = pymaid.CatmaidInstance(url, token, name, password)

# %%
# deannotation all sub-annotations under meta-annotation
## VERY DANGEROUS; BE CAREFUL!!! ##

meta = 'mw partner-loops-bilateral partners'
annots = [annot for annot in pymaid.get_annotated(meta).name]

[pymaid.remove_annotations(pymaid.get_skids_by_annotation(annot), annot) for annot in annots]
# %%
# dennotate single annotation

annot = 'mw partner loops'
pymaid.remove_annotations(pymaid.get_skids_by_annotation(annot), annot)