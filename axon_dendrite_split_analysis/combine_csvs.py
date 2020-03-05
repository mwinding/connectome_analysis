import csv
import pandas as pd

# I had to export these skeleton morphologies in groups of ~600 so CATMAID didn't crash
# this script combines them again into one CSV
dist_l1 = pd.read_csv('axon_dendrite_data/splittable_connectdists_left1_norm.csv', header=0, skipinitialspace=True, keep_default_na = False)
dist_l2 = pd.read_csv('axon_dendrite_data/splittable_connectordists_left2_norm.csv', header=0, skipinitialspace=True, keep_default_na = False)
dist_r1 = pd.read_csv('axon_dendrite_data/splittable_connectordists_right1_norm.csv', header=0, skipinitialspace=True, keep_default_na = False)
dist_r2 = pd.read_csv('axon_dendrite_data/splittable_connectordists_right2_norm.csv', header=0, skipinitialspace=True, keep_default_na = False)

connectors = pd.concat([dist_l1, dist_l2, dist_r1, dist_r2])

connectors.to_csv('axon_dendrite_data/splittable_connectordists_all_2020-3-5.csv')