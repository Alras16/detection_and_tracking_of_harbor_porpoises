#!/bin/bash
sleap-track --labels /home/alras16/sleap_training_data/831_labeled_frames_in_12_videos.slp --only-suggested-frames -m /home/alras16/sleap_training_data/models/train_on_65_labeled_frames_in_1_video220909_084323.centroid -m /home/alras16/sleap_training_data/models/train_on_65_labeled_frames_in_1_video220909_084323.centered_instance -o 831_labeled_frames_in_12_videos.slp.predictions.slp --verbosity json --no-empty-frames
