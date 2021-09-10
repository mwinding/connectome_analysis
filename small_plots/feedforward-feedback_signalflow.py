#%%
import os
import sys
os.chdir(os.path.dirname(os.getcwd())) # make directory one step up the current directory

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

data = [[.2, .8], [.47, .53], [.42, .58], [.62, .38]]
df = pd.DataFrame(data, index=['a-d', 'a-a', 'd-d', 'd-a'], columns = ['feedback', 'feedforward'])
df['bias'] = df.feedforward-df.feedback

fig, ax = plt.subplots(1,1,figsize=(0.75,2))
plt.bar(x = range(len(df.index)), height = df.bias)
#plt.bar(x = range(len(df.index)), height = df.feedback, bottom=df.feedforward)
ax.set(ylim=(-1,1))
plt.savefig('small_plots/plots/ff-fb_signal-flow-plots.pdf', format='pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1,figsize=(0.75,2))
plt.bar(x = range(len(df.index)), height = df.feedforward)
plt.bar(x = range(len(df.index)), height = -df.feedback)
ax.set(ylim=(-1,1))
plt.savefig('small_plots/plots/ff-fb_signal-flow-plots-type2.pdf', format='pdf', bbox_inches='tight')