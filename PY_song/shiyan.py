import numpy as np

a = np.ones((2,3))
a[0][2] = -1
a[1][0] = 0.5
print(a.reshape(-1).exp())