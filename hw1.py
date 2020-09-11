import numpy

mat = [[3, 1, 1], [2, 4, 2], [-1, -1, 1]]

#get eigendecomposition of mat
decomp = numpy.linalg.eig(mat)
print(decomp)
print("\nEigen vals: {0}, {1}, {2}".format(decomp[0][0], decomp[0][1], decomp[0][2]))
print("\nEigen vectors: \n {0}".format(decomp[1]))

#attempt to reconstruct mat uisng decomp
inv = numpy.linalg.inv(decomp[1])
vals = [[decomp[0][0], 0, 0], [0, decomp[0][1], 0], [0, 0, decomp[0][2]]]
prod = numpy.matmul(decomp[1], vals)
prod = numpy.matmul(prod, inv)
print("\nReconstruction:\n {0}".format(prod))

