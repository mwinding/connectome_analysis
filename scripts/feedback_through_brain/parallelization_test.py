#%%
import os
import sys
try:
    os.chdir('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
    print(os.getcwd())
except:
    pass

# %%
#
from math import sqrt
[sqrt(i ** 2) for i in range(10)]

# %%
from joblib import Parallel, delayed