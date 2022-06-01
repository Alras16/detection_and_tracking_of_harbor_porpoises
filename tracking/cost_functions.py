import numpy as np

def instance_similarity(pred_keypoints, gt_keypoints):
    """
    Measures the similarity by looking at the distance between 
    corresponding keypoints in the instances, normalized by 
    the number of valid nodes in the candidate instance.
    """
    num_valid_preds = ~(np.isnan(pred_keypoints).any(axis=1))
    keypoint_dists = np.sum((gt_keypoints - pred_keypoints) ** 2, axis=1)
    return np.nansum(np.exp(-keypoint_dists)) / np.sum(num_valid_preds)

def keypoint_distance(pred_keypoint, gt_keypoint):
    """
    Measures the similarity by looking at the distance between 
    the instance heads.
    """
    return -np.linalg.norm(gt_keypoint - pred_keypoint)

