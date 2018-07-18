import random
from Tensor import *

# Make a set of indices with random size between 1 and 10
ind = [Index(random.randint(1, 10)) for i in range(4)]
print('Initial Indices: ' + str(ind))
print()

# initialize tensors
A = Tensor(*ind[:2])
B = Tensor(*ind[1:3])
C = Tensor(*ind[2:])

print('Tensor A: ' + str(A))
print('Tensor B: ' + str(B))
print('Tensor C: ' + str(C))
print()

# create the function graph
graph = A @ B @ C
print('Graph: ' + str(graph))
print('Graph Indices: ' + str(graph.indices))
print()

# compute gradient wrt A, given indices
gradient = simplify(graph.gradient(B))
print('Gradient of graph wrt Tensor ' + str(B) + ': ' + str(gradient))
print('Gradient Indices: ' + str(gradient.indices))
