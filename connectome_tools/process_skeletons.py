import networkx as nx 
import pandas as pd

def skid_as_networkx_graph(skeleton_path):

    skeleton = pd.read_csv(skeleton_path, header=0, skipinitialspace=True, keep_default_na = False)

    G = nx.Graph()

    # adding nodes from child-parent structure of skeleton to networkx object
    # x,y,z attributes are spatial coordinates in nm
    # identifies rootnode, whose parent id is an empty cell or '' (due to import style)
    for i in range(0, len(skeleton['treenode_id'])):
        if(skeleton['parent_treenode_id'][i]==''):
            G.add_node(skeleton['treenode_id'][i], x = skeleton['x'][i], y = skeleton['y'][i], z = skeleton['z'][i], root = True)

        if not(skeleton['parent_treenode_id'][i]==''):
            G.add_node(skeleton['treenode_id'][i], x = skeleton['x'][i], y = skeleton['y'][i], z = skeleton['z'][i], root = False)
            

    # adding edges from child-parent structure of skeleton to networkx object
    # also identifies the rootnode, whose parent id is an empty cell or '' (due to import style)
    for i in range(0, len(skeleton['treenode_id'])):
        if not(skeleton['parent_treenode_id'][i]==''):
            G.add_edge(int(skeleton['treenode_id'][i]), int(skeleton['parent_treenode_id'][i]))

    return(G)

# find rootnode id
def identify_root(G):
    root=nx.get_node_attributes(G,'root')

    rootnode = []
    for i in root:
        if(root[i]==True):
            rootnode = i

    return(rootnode)
