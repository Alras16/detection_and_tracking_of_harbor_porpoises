
import numpy as np
from scipy.optimize import linear_sum_assignment
from cost_functions import keypoint_distance

class Matching():
    def __init__(self):
        self.detections = None
        self.trackers = None
    
    def get_detections_and_trackers(self, detections, trackers):
        self.detections = detections
        self.trackers = trackers
    
    @staticmethod
    def hungarian_matching(cost_matrix):
        x, y= linear_sum_assignment(cost_matrix)
        return list(zip(x, y))

    @staticmethod
    def greedy_matching(cost_matrix):
        x, y = np.unravel_index(np.argsort(cost_matrix, axis=None), cost_matrix.shape)
        unassigned_detections = list(zip(x, y))

        assignments = []
        while len(unassigned_detections) > 0:
            x, y = unassigned_detections.pop(0)
            assignments.append((x, y))
            for i in range(len(unassigned_detections) - 1):
                if unassigned_detections[i] == (x, y):
                    del unassigned_detections[i]
        return assignments
    
    def associate_detections_to_trackers(self, detections, trackers, threshold, greedy_matching=False):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        cost_matrix = keypoint_distance(detections, trackers)
        if min(cost_matrix.shape) > 0:
            a=(cost_matrix > threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                if greedy_matching == True:
                    matched_indices = self.greedy_matching(cost_matrix)
                else:
                    matched_indices = self.hungarian_matching(cost_matrix)
        else:
            matched_indices = np.empty(shape=(0,2))

        unmatched_detections = []
        unmatched_trackers = []
        for d, det in enumerate(detections):
            if (d not in matched_indices[:,0]):
                unmatched_detections.append(d)
        for t, trk in enumerate(trackers):
            if (t not in matched_indices[:,1]):
                unmatched_trackers.append(t)
        
        # matches = []
        # for m in matched_indices:
        #     if (cost_matrix[m[0], m[1]] < threshold):
        #         unmatched_detections.append(m[0])
        #         unmatched_trackers.append(m[1])
        #     else:
        #         matches.append(m.reshape(1,2))
        
        if len(matches == 0):
            matches = np.empty((0,2), dtype=int)
        else:
             matches = np.concatenate(matched_indices, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
