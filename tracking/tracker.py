
# class Tracker(object):
#     def __init__(self, min_hits = 3, threshold = 0.3):
#         ...

import numpy as np


cache = dict()
ref = tuple((1,4), (2,3), (3,2))
gtr = tuple((5,2), (4,3), (3,4))


a = cache[ref]
b = cache[gtr]
dist = np.linalg.norm(a - b)

print(dist)