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

#pymaid.clear_cache()

rm = pymaid.CatmaidInstance(url, name, password, token)

# %%

def timeTracing(user, interval, start_date, end_date=None):
    time_interval = pd.Timestamp(year=2020, month=1, day=1, hour=12, minute=interval) - pd.Timestamp(year=2020, month=1, day=1, hour=12, minute=0)

    userstats = pymaid.get_user_actions(users=user, start_date=start_date, end_date=end_date)

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
#ana = ['ana', 'andrey', 'marc']
#ana_time = [timeTracing(user, 3, datetime.date(2020, 3, 27)) for user in users]
# %%
# stopped checking
#marc = timeTracing('marc', 3, datetime.date(2020, 11, 23))
#print(marc)
# %%
# stopped checking
#keira = timeTracing('keira', 3, datetime.date(2020, 11, 23))
#print(keira)
# %%
andy = timeTracing('andrey', 3, datetime.date(2020, 12, 15), datetime.date(2021, 1, 20))
print(andy)
# %%
# stopped checking
#xinyu = timeTracing('xinyu', 3, datetime.date(2020, 11, 23))
#print(xinyu)
# %%
michael = timeTracing('michael', 3, datetime.date(2020, 12, 15), datetime.date(2021, 1, 20))
print(michael)
# %%
nadine = timeTracing('nadine', 3, datetime.date(2020, 12, 15), datetime.date(2021, 1, 20))
print(nadine)
# %%
ana = timeTracing('ana', 3, datetime.date(2020, 12, 15), datetime.date(2021, 1, 20))
print(ana)

# %%
elizabeth = timeTracing('ebarsotti', 3, datetime.date(2020, 12, 15), datetime.date(2021, 1, 20))
print(elizabeth)
# %%
#correspondences = pd.DataFrame(data = correspondences, columns = ['old_name', 'new_name'])
#print(correspondences)

#correspondences.to_csv('CATMAID_scripts/brain_tracing_stats.csv')

#%%
#stats = pymaid.get_user_stats()
#print(stats)
#import datetime
#current_stats = pymaid.get_user_stats(start_date=datetime.date(2020, 3, 27))
#print(current_stats)
