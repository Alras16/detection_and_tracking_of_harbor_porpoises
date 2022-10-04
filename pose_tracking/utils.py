import numpy as np
import re
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise, Q_continuous_white_noise

def keypoint_distance(pred_keypoint, gt_keypoint):
    """
    Measures the similarity by looking at the distance between 
    keypoints of the harbor porpoises head.
    """
    return -np.linalg.norm(gt_keypoint - pred_keypoint)

def norm2squared_matrix(objs, hyps, max_d2=float('inf')):
    """
    Computes the Euclidean distance matrix between object and hypothesis points
    """
    objs = np.atleast_2d(objs).astype(float)
    hyps = np.atleast_2d(hyps).astype(float)

    if objs.size == 0 or hyps.size == 0:
        return np.empty((0, 0))

    assert hyps.shape[1] == objs.shape[1], "Dimension mismatch"

    delta = objs[:, np.newaxis] - hyps[np.newaxis, :]
    C = np.sqrt(np.sum(delta ** 2, axis=-1))

    C[C > max_d2] = np.nan
    return C

def hungarian_matching(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return list(zip(x, y))

def greedy_matching(cost_matrix):
    x, y = np.unravel_index(np.argsort(cost_matrix, axis=None), cost_matrix.shape)
    unassigned_detections = list(zip(x, y))

    assignments = []
    while len(unassigned_detections) > 0:
        x, y = unassigned_detections.pop(0)
        assignments.append((x, y))
        for i in range(max(cost_matrix.shape)):
            if (i, y) in unassigned_detections:
                unassigned_detections.remove((i, y))
            if (x, i) in unassigned_detections:
                unassigned_detections.remove((x, i))
    return assignments
   
# Python program to use
# main for function call.
if __name__ == "__main__":
    objs = np.array([[236, 134], [343, 149], [404, 206]])
    hyps = np.array([[360, 156], [400, 200], [224, 127], [200, 200]])
    dists = norm2squared_matrix(objs, hyps)

    ndists = np.array([[68, 31, 79, 6, 21, 37], [45, 27, 23, 66, 9, 17], [83, 59, 25, 38, 63, 25], [1, 37, 53, 100, 80, 51], [69, 72, 74, 32, 82, 31], [34, 95, 61, 64, 100, 82]])
    ndists1 = np.array([[68, 35, 37, 10, 47, 31], [10, 70, 26, 52, 58, 74], [71, 59, 86, 65, 84, 40], [65, 87, 53, 4, 69, 77], [33, 28, 31, 68, 67, 38], [5, 34, 72, 93, 95, 18]])
    print(ndists1.shape)

    print(ndists[0])

    hungarian_matches = hungarian_matching(ndists1)
    print(hungarian_matches)

    greedy_matches = greedy_matching(ndists1)
    print(greedy_matches)

    R_std = 0.35
    Q_std = 0.04
    dt = 1.0 # time step

    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
    kf.H = np.array([[1, 0, 0, 0],[0, 1, 0, 0]])

    kf.P[2:,2:] *= 100. #give high uncertainty to the unobservable initial velocities
    kf.P *= 10.
    print(kf.P)

    q = Q_discrete_white_noise(dim=2, dt=dt, var=pow(Q_std, 2), block_size=1, order_by_dim=False)
    kf.R = np.eye(2) * pow(R_std, 2)
    kf.Q = block_diag(q, q)

    print(f'Q = {kf.Q}')
    print(kf.R)
    print(pow(Q_std,2))

    print(np.empty((0, 2), dtype=int))
    print(np.arange(len(hyps)))
""" 
    detections = np.array([1691.97,381.048,152.23,352.6171])

#<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z> 

1233.55,467.507,133.65,218.985
108.484,461.531,97.759,297.453
256.996,420.694,101.497,296.434
19.5685,469.707,87.4595,343.434
1869.09,376.414,49.91,222.562
1252.28,509.199,59.49,133.608
0,265.374,109.767,381.325




2,-1,1691.29,382.432,187.78,363.197,0.99714,-1,-1,-1
2,-1,249.136,473.997,108.874,235.769,0.986837,-1,-1,-1
2,-1,1272.37,447.903,72.79,217.543,0.985943,-1,-1,-1
2,-1,88.626,445.029,104.591,328.988,0.967904,-1,-1,-1
2,-1,10.6657,486.485,106.665,375.935,0.812186,-1,-1,-1
2,-1,1875.74,375.006,43.26,203.969,0.749853,-1,-1,-1
2,-1,3.37989,285.912,96.0394,365.456,0.579795,-1,-1,-1 """
