#%%

import pandas as pd
import numpy as np
from pymaid_creds import url, name, password, token
import pymaid


rm = pymaid.CatmaidInstance(url, token, name, password)

# %%
# dennotate / remove single annotation

annot = 'something'
pymaid.remove_annotations(pymaid.get_skids_by_annotation(annot), annot)

# %%
# deannotate / remove all sub-annotations under meta-annotation
## VERY DANGEROUS; BE CAREFUL!!! ##

meta = 'meta-something'
annots = [annot for annot in pymaid.get_annotated(meta).name]

[pymaid.remove_annotations(pymaid.get_skids_by_annotation(annot), annot) for annot in annots]

# %%
# deannotate / remove annotations all sub- and sub-sub-anotations under meta-meta-annotation
## EXTREMELY DANGEROUS; BE VERY CAREFUL!!!!!! ##

meta_meta = 'meta-meta-something'
meta_annots = [meta for meta in pymaid.get_annotated(meta_meta).name]
annots = [[annot for annot in pymaid.get_annotated(meta).name] for meta in meta_annots]
annots = [x for sublist in annots for x in sublist]

[pymaid.remove_annotations(pymaid.get_skids_by_annotation(annot), annot) for annot in annots]

# %%
