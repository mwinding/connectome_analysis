import connectome_tools.process_matrix as promat

binMat = promat.binary_matrix('data/Gadn-pair-sorted.csv', 0.05)

print(binMat)
