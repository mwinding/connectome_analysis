import connectome_tools.process_matrix as promat
import pandas as pd
import numpy as np

pairs = pd.read_csv('data/bp-pairs-2020-01-28.csv', header = 0)

skid_left = np.int64(1679034)
skid_right = promat.identify_pair(skid_left, pairs)

print("Left skid: %i, Right skid: %i" %(skid_left, skid_right))
print("Actual pair: %d, %d" % (pairs.iloc[4].values[0], pairs.iloc[4].values[1]))
