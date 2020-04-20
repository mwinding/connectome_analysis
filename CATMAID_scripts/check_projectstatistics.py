#%%
import sys
sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')

import pymaid
import pyoctree
from pymaid_creds import url, name, password, token
import numpy as np
import pandas as pd
import re
import natsort as ns
import datetime

pymaid.clear_cache()

rm = pymaid.CatmaidInstance(url, name, password, token)

# %%

def timeTracing(user, interval, start_date):
    time_interval = pd.Timestamp(year=2020, month=1, day=1, hour=12, minute=interval) - pd.Timestamp(year=2020, month=1, day=1, hour=12, minute=0)

    userstats = pymaid.get_user_actions(users=user, start_date=start_date)

    time_chunks = []
    chunk = []

    for i in range(1, len(userstats['timestamp'])):
        gap = userstats['timestamp'][i] - userstats['timestamp'][i-1]
        if(gap < time_interval):
            chunk.append(userstats['timestamp'][i-1])
        if(gap > time_interval):
            chunk.append(userstats['timestamp'][i-1])
            total_chunk = chunk[len(chunk)-1]-chunk[0]
            time_chunks.append(total_chunk)
            #print(total_chunk)
            chunk = []

    time_sum = time_chunks[0]

    for i in range(1, len(time_chunks)):
        time_sum = time_sum + time_chunks[i]
        #print(time_sum)

    return(time_sum)

# %%
users = ['ana', 'andrey', 'marc']
times = [timeTracing(user, 3, datetime.date(2020, 3, 27)) for user in users]

# %%
users2 = ['nadine', 'michael', 'keira', 'xinyu']
times2 = [timeTracing(users2, 3, datetime.date(2020, 3, 27)) for user in users2]
# %%
users_all = users + users2
times_all = times + times2
print(users_all)
print(times_all)

#%%
#stats = pymaid.get_user_stats()
#print(stats)
#import datetime
#current_stats = pymaid.get_user_stats(start_date=datetime.date(2020, 3, 27))
#print(current_stats)
