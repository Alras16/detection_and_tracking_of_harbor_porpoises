from pickle import NONE
import cv2

class OpticalFlowTracker:
    def __init__(self):
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01) 
        self.window_size = (15, 15) 
        self.max_level = 3

    def track_objects(self, frame, matched_instances):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

