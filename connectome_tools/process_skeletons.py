import networkx as nx 
import pandas as pd

skeleton = pd.read_csv('data/test_skeleton.csv', header=0, skipinitialspace=True, keep_default_na = False)

G = nx.Graph()
rootnode = []

# adding edges from child-parent structure of skeleton to networkx object
# also identifies the rootnode, whose parent id is an empty cell or '' (due to import style)
for i in range(0, len(skeleton['treenode_id'])):
    if not(skeleton['parent_treenode_id'][i]==''):
        G.add_edge(int(skeleton['treenode_id'][i]), int(skeleton['parent_treenode_id'][i]))
    if(skeleton['parent_treenode_id'][i]==''):
        rootnode = skeleton['treenode_id'][i]

print(G.edges)
print(rootnode)
