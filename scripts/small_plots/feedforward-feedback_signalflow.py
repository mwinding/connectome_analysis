#%%

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# allows text to be editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# font settings
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'arial'

# data collected from Benjamin Pedigo's plot in S57
#data = [[.23, .77], [.46, .54], [.36, .64], [.67, .33]]
data = [[.13, .87], [.41, .59], [.31, .69], [.71, .29]]
df = pd.DataFrame(data, index=['a-d', 'a-a', 'd-d', 'd-a'], columns = ['feedback', 'feedforward'])
df['bias'] = df.feedforward-df.feedback

fig, ax = plt.subplots(1,1,figsize=(0.75,2))
plt.bar(x = range(len(df.index)), height = df.bias)
#plt.bar(x = range(len(df.index)), height = df.feedback, bottom=df.feedforward)
ax.set(ylim=(-1,1))
plt.savefig('plots/ff-fb_signal-flow-plots.pdf', format='pdf', bbox_inches='tight', ax=ax)

fig, ax = plt.subplots(1,1,figsize=(0.75,2))
plt.bar(x = range(len(df.index)), height = df.feedforward)
plt.bar(x = range(len(df.index)), height = -df.feedback)
ax.set(ylim=(-1,1))
plt.savefig('plots/ff-fb_signal-flow-plots-type2.pdf', format='pdf', bbox_inches='tight', ax=ax)

# %%
# seaborn barplots

'''
# old data
data = [[.77, 'feedforward', 'a-d'], [.23, 'feedback', 'a-d'], 
        [.54, 'feedforward', 'a-a'], [.46, 'feedback', 'a-a'], 
        [.64, 'feedforward', 'd-d'], [.36, 'feedback', 'd-d'], 
        [.33, 'feedforward', 'd-a'], [.67, 'feedback', 'd-a']]
'''
data = [[.87, 'feedforward', 'a-d'], [.13, 'feedback', 'a-d'], 
        [.59, 'feedforward', 'a-a'], [.41, 'feedback', 'a-a'], 
        [.69, 'feedforward', 'd-d'], [.31, 'feedback', 'd-d'], 
        [.29, 'feedforward', 'd-a'], [.71, 'feedback', 'd-a']]

df = pd.DataFrame(data, columns = ['fraction', 'direction', 'edge_type'])

fig, ax = plt.subplots(1,1,figsize=(1.5,1))
sns.barplot(data=df, x='edge_type', y='fraction', hue='direction')
ax.set(ylim=(0,1))
plt.savefig('plots/ff-fb_signal-flow-plots-type3.pdf', format='pdf', bbox_inches='tight')


# %%
