import sleap
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.style.use("seaborn-deep")
sleap.versions()

# Load the train top-down or bottom-up model
# Top-down
predictor = sleap.load_model([
    "models/train_on_831_labeled_frames_in_11_videos220908_163341.centroid.n=831", 
    "models/train_on_831_labeled_frames_in_11_videos220908_171252.centered_instance.n=831"
    ])

# Bottom-up
# predictor = sleap.load_model("bu.210506_230852.multi_instance.n=1800.zip")

# Then load the ground truth (GT) labels and generate the predictions.
labels_gt = sleap.Labels.load_file("labels_packages/labels_with_images.pkg.slp")
labels_pr = predictor.predict(labels_gt)

#Generating another set of metrics can then be calculated with the pair of GT and predicted labels
metrics = sleap.nn.evals.evaluate(labels_gt, labels_pr)

# To start, let’s look at the summary of the localization errors:
print("Error distance (50%):", metrics["dist.p50"])
print("Error distance (90%):", metrics["dist.p90"])
print("Error distance (95%):", metrics["dist.p95"])

# These are the percentiles of the distribution of how far off the model was from the ground truth location.
# The entire distribution can also be vizualized as follows
plt.figure(figsize=(6, 3), dpi=150, facecolor="w")
sns.histplot(metrics["dist.dists"].flatten(), binrange=(0, 20), kde=True, kde_kws={"clip": (0, 20)}, stat="probability")
plt.xlabel("Localization error (px)")

# This metric is intuitive, but it does not incorporate other sources of error like 
# those stemming from poor instance detection and grouping, or missing points.

# The Object Keypoint Similarity (OKS) is a more holistic metric that takes factors such as 
# landmark visibility, animal size, and the difficulty in locating keypoints.

# The distribution of OKS scores can be plotted as follows
plt.figure(figsize=(6, 3), dpi=150, facecolor="w")
sns.histplot(metrics["oks_voc.match_scores"].flatten(), binrange=(0, 1), kde=True, kde_kws={"clip": (0, 1)}, stat="probability")
plt.xlabel("Object Keypoint Similarity");

# Another way to summarize this is through precision-recall curves, which evaluate how well the model does
# at different thresholds of OKS scores. The higher the threshold, the more stringent our criteria for 
# classifying a prediction as correct.

# Plot the different thresholds as follows:
plt.figure(figsize=(4, 4), dpi=150, facecolor="w")
for precision, thresh in zip(metrics["oks_voc.precisions"][::2], metrics["oks_voc.match_score_thresholds"][::2]):
    plt.plot(metrics["oks_voc.recall_thresholds"], precision, "-", label=f"OKS @ {thresh:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left");

# An easy way to summarize this analysis is to take the average over all of these thresholds to compute the mean 
# Average Precision (mAP) and mean Average Recall (mAR) which are widely used in the pose estimation literature.
print("mAP:", metrics["oks_voc.mAP"])
print("mAR:", metrics["oks_voc.mAR"])