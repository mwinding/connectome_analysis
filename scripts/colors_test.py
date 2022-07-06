# %%
#
import matplotlib.pyplot as plt

colors = ['#00753F', '#1D79B7', '#5D8C90', '#D4E29E', '#FF8734', '#E55560', '#F9EB4D', '#C144BC', '#FF9CFF', '#8C7700', '#77CDFC', '#FFDAC7', '#E0B1AD', '#9467BD','#D88052', '#A52A2A', 'tab:grey']
cell_types = ['sens', 'PN', 'LN', 'LHN', 'MBIN', 'KC', 'MBON', 'MB-FBN', 'FFN', 'CN', 'ascending', 'pre-DN-SEZ', 'pre-DN-VNC', 'RGN', 'DN-SEZ', 'DN-VNC', 'unk']
cell_types = [x + ', ' + colors[i] for i, x in enumerate(cell_types)]
plt.bar(x=cell_types,height=[1]*len(colors),color=colors)
plt.xticks(rotation=45, ha='right')
plt.savefig('plots/test_colors.pdf', format='pdf', bbox_inches='tight')
# %%
