import numpy as np

from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

def convert_x_to_keypoint(x):
    return (x[0], x[1])

class KalmanFilter(object):
    count = 0
    def __init__(self, keypoint):
        self.R_std = 0.35
        self.Q_std = 0.04
        self.dt = 1.0 # time step

        # Define state transition matrix and measurement function
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1,self.dt,0,0],[0,1,0,self.dt],[0,0,1,0],[0,0,0,1]])
        self.kf.H = np.array([[1, 0, 0, 0],[0, 1, 0, 0]])

        # Define the measurement uncertainty matrix and process uncertainty matrix
        self.q = Q_discrete_white_noise(dim=2, dt=self.dt, var=pow(self.Q_std, 2))
        self.kf.R = np.eye(2) * pow(self.R_std, 2)
        self.kf.Q = block_diag(self.q, self.q)
        
        # Initial condition for state vector and covariance matrix
        self.P = np.eye(4) * 500.
        self.kf.x[:2] = keypoint
      
        # Setting tracking id and update time etc.
        self.id = KalmanFilter.count
        self.time_since_update = 0
        KalmanFilter.count += 1
        self.hit_streak = 0
        self.history = []
        self.hits = 0
        
    def update(self, keypoint):
        self.time_since_update = 0
        self.hit_streak += 1
        self.hits += 1
        self.history = []
        self.kf.update(keypoint)
    
    def predict(self):
        self.kf.predict()
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_keypoint(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_keypoint(self.kf.x)
