import numpy as np

from util import *

a = np.array([[1, 2], [3, 4]], dtype=np.float64)
print(dumps_custom({'a': a}))