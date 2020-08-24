import numpy as np

foo = np.array([[1,2,3],[4, 5, 6], [7, 8, 9]])

print(foo[1:3, 1:3])
print(foo[(1, 2), (1,2)])