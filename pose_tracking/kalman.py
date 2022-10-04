import os
import glob
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage import io
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise, Q_continuous_white_noise
from tracking.components import associate_dets_to_trks

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
        self.q = Q_discrete_white_noise(dim=2, dt=self.dt, var=pow(self.Q_std, 2), order_by_dim=False)
        self.q1 = Q_continuous_white_noise(dim=2, dt=self.dt, spectral_density=pow(self.Q_std, 2), order_by_dim=False)
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
        self.age = 0
        
    def update(self, keypoint):
        self.time_since_update = 0
        self.hit_streak += 1
        self.hits += 1
        self.history = []
        self.kf.update(keypoint)
    
    def predict(self):
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x[0:2])
        return self.history[-1]

    def get_state(self):
        return self.kf.x[0:2]

class KalmanTracker(object):
    def __init__(self, max_age=1, min_hits=2, threshold=20):
        self.trackers = []
        self.frame_count = 0
        self.max_age = max_age
        self.min_hits = min_hits
        self.threshold = threshold

    def update(self, dets=np.empty((0,3))):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 3))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_dets_to_trks(dets, trks, self.threshold, greedy=False)

        # Update matched trackers with signed detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
        
        # Initialize new trackers for unmatched detections
        for d in unmatched_dets:
            trk = KalmanFilter(dets[d, :])
            self.trackers[trk]
        
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            # Remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,3)) 


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--threshold", help="Maximum euclidean distance threshold for match.", type=float, default=10)
    args = parser.parse_args()
    return args           

if __name__ == "__main__":
    total_time = 0.0
    total_frames = 0
    args = parse_args()
    colours = np.random.rand(32, 3) #used only for display

    if(args.display):
        if not os.path.exists('mot_benchmark'):
            print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')

    if not os.path.exists('output'):
        os.makedirs('output')
    pattern = os.path.join(args.seq_path, args.phase, '*', 'det', 'det.txt')
    for seq_dets_fn in glob.glob(pattern):  
        tracker = KalmanTracker(max_age=args.max_age, min_hits=args.min_hits, threshold=args.threshold) #create instance of the SORT tracker
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
        seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]

    with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:
        print("Processing %s."%(seq))
        for frame in range(int(seq_dets[:,0].max())):
            frame += 1 #detection and frame numbers begin at 1
            dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
            #dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
            total_frames += 1

            if(args.display):
                fn = os.path.join('mot_benchmark', args.phase, seq, 'img1', '%06d.jpg'%(frame))
                im =io.imread(fn)
                ax1.imshow(im)
                plt.title(seq + ' Tracked Targets')

        start_time = time.time()
        trackers = tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
            print('%d,%d,%.2f,%.2f,-1,-1,1,-1,-1,-1'%(frame,d[2],d[0],d[1]),file=out_file)
            if(args.display):
                d = d.astype(np.int32)
                ax1.add_patch(patches.Circle(d[0], d[1], radius=1))

            if(args.display):
                fig.canvas.flush_events()
                plt.draw()
                ax1.cla()

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))
