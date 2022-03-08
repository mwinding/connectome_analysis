#%%
# identify users to acknowledge in paper for any tracing at all

import pandas as pd
import numpy as np

date = '2022-03-08'

# load contribution data, saved manually from CATMAID
nodes = pd.read_csv(f'data/users/nodes_contributed_{date}.csv')
post = pd.read_csv(f'data/users/post_contributed_{date}.csv')
pre = pd.read_csv(f'data/users/pre_contributed_{date}.csv')
reviews = pd.read_csv(f'data/users/reviews_contributed_{date}.csv')
user_data = pd.read_csv(f'data/users/user_data_{date}.csv')
user_data.set_index('user', inplace=True)

# %%
# check if nodes data is representative

if(len(np.setdiff1d(pre.user.values, nodes.user.values))==0):
    print(f'Presynaptic data add no new users')
else:
    print('Check presynaptic data!')

if(len(np.setdiff1d(post.user.values, nodes.user.values))==0):
    print(f'Postsynaptic data add no new users')
else:
    print('Check postsynaptic data!')

if(len(np.setdiff1d(reviews.user.values, nodes.user.values))==0):
    print(f'Review data adds no new users')
else:
    print('Check review data!')

# %%
# use just nodes data

name_data = user_data.loc[nodes.user]
contributors = [f'{name_data.loc[user, "first_name"]} {name_data.loc[user, "last_name"]}' for user in name_data.index]
', '.join(contributors)
# %%
