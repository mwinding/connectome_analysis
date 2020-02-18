import connectome_tools.process_matrix as promat
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns


print("running...")

matrix = pd.read_csv('data/CN_test_matrix_G-pair-sorted.csv', header=0, index_col=0)

#binMat = promat.binary_matrix('data/Gadn-pair-sorted.csv', 0.01)

sns.heatmap(matrix)
plt.show()

#file = 'plots/Gadn-pair-sorted-binaryMatrix.png'
#plt.savefig(file)