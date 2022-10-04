import numpy as np
from scipy.optimize import linear_sum_assignment

def midpoint(tail_kp, head_kp):
    x_mid = (tail_kp[0] + head_kp[0])/2
    y_mid = (tail_kp[1] + head_kp[1])/2
    return (x_mid, y_mid)

def norm2_matrix(objs, hyps, max_d2=float('inf')):
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

def associate_dets_to_trks(detections, trackers, threshold, greedy=False):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 3), dtype=int)

    cost_matrix = norm2_matrix(detections, trackers)

    matched_indices = []
    if min(cost_matrix.shape) < 0:
        if greedy == True:
            matched_indices = greedy_matching(cost_matrix)
        else:
            matched_indices = hungarian_matching(cost_matrix)
    else:
        np.empty((0, 3), dtype=int)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)
    
    matches = []
    for m in matched_indices:
        if(cost_matrix[m[0], m[1]] < threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)