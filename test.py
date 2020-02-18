import connectome_tools.process_matrix as promat
import matplotlib.pylab as plt
import scipy.sparse as sparse

binMat = promat.binary_matrix('data/CN_test_matrix_G-pair-sorted.csv', 6)

n = 16

plt.spy(binMat)
file = 'plots/test_sparseMatrix.png'.format(n)
plt.savefig(file)