import pandas as pd
import numpy as np
import connectome_tools.process_matrix as promat

pairs = pd.read_csv('data/pairs-2020-02-27.csv', header = 0)

sample_left = np.random.choice(pairs['leftid'].values, size = int(len(pairs['leftid'])/2), replace = False)
sample_right = np.random.choice(pairs['rightid'].values, size = int(len(pairs['rightid'])/2), replace = False)

sample_left = pd.DataFrame(sample_left, columns = ["leftid"])
sample_right = pd.DataFrame(sample_right, columns = ["rightid"])

sample_left.to_csv("outputs/sampled_leftbrain.csv")
sample_right.to_csv("outputs/sampled_rightbrain.csv")