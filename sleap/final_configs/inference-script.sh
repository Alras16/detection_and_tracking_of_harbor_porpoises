#!/bin/bash
sleap-track --labels /home/alras16/sleap_training_data/final_configs/305_labeled_frames_in_4_videos.slp --only-suggested-frames -m /home/alras16/sleap_training_data/models/train_on_1017_labeled_frames_using_sgd_optimizer220930_101241.multi_instance -o 305_labeled_frames_in_4_videos.slp.predictions.slp --verbosity json --no-empty-frames
