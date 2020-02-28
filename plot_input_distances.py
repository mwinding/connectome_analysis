import networkx as nx 
from networkx.utils import pairwise
import pandas as pd
import numpy as np
import connectome_tools.process_skeletons as proskel
import math
import matplotlib.pyplot as plt
import seaborn as sns

connectors = pd.read_csv('outputs/connectdists.csv')

print(connectors)


inputs = []
outputs = []
for i in range(len(connectors)):
    if(connectors.iloc[i]['type']=='postsynaptic'):
        inputs.append(connectors.iloc[i]['distance_root'])
    if(connectors.iloc[i]['type']=='presynaptic'):
        outputs.append(connectors.iloc[i]['distance_root'])

#print(inputs)
fig, ax = plt.subplots(1,1,figsize=(8,4))
#sns.distplot(data = inputs, ax = ax, )
#sns.distplot(data = outputs, ax = ax, )

ax.hist(inputs)
ax.hist(outputs)
plt.show()