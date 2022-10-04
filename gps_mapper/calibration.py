from dis import dis
import os
import re
import cv2
import yaml
import numpy as np

class CalibrateCamera:
    def __init__(self):
        self.video_file = None
        self.board_size = None
        self.num_images = None
        self.obj_points_list = []
        self.img_points_list = []

    def set_calibration_parameters(self, video, board_size, num_images):
        self.video_file = video
        self.board_size = board_size
        self.num_images = num_images

    def detect_chessboard_corners(self, img, num_image):
        # Prepare object points
        obj_points = np.zeros((self.board_size[0]*self.board_size[1], 3), np.float32)
        obj_points[:,:2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        # Convert an RGB-image into a gray-scale image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Invert the gray-scale image
        invert_img = cv2.bitwise_not(gray_img)
        # Find the corners of the chessboard
        ret, corners = cv2.findChessboardCorners(invert_img, self.board_size, None, flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
        if ret == True: 
            # Refine the detected corners
            img_points = cv2.cornerSubPix(invert_img, corners, (11,11), (-1,-1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            # Vizualize the corners and save the image
            img = cv2.drawChessboardCorners(img, self.board_size, img_points,ret)
            cv2.imwrite('calibration_images/img' + str(num_image) + '.png',img)
            return obj_points, img_points 
        else:
            return None, None

    def calibrate_camera_from_video(self):
        img_size, num_image, count = None, 0, 0
        cap = cv2.VideoCapture(self.video_file)
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            ret_val, frame = cap.read()
            if ret_val:
                img_size = (frame.shape[1], frame.shape[0])
                obj_points, img_points = self.detect_chessboard_corners(frame, num_image)
                if obj_points is not None:
                    self.obj_points_list.append(obj_points)
                    self.img_points_list.append(img_points)
                    num_image += 1
            else:
                self.num_images += 1 if self.num_images < 30 else 30
            
            # Calculate next frame to extract
            count += int(num_frames / self.num_images)
            if (count > num_frames):
                break
        cap.release()
        if self.obj_points_list:
            _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points_list, self.img_points_list, img_size, None, None)
            return mtx, dist, rvecs, tvecs, img_size
        else: 
            return None, None, None, None, None
    
    def calculate_calibration_error(self, mtx, dist, rvecs, tvecs):
        mean_error = 0
        for i in range(len(self.obj_points_list)):
            new_img_points, _ = cv2.projectPoints(self.obj_points_list[i], rvecs[i], tvecs[i], mtx, dist)
            mean_error += cv2.norm(self.img_points_list[i], new_img_points, cv2.NORM_L2)/len(new_img_points)
        return mean_error/len(self.obj_points_list)

    def save_calibration_result(self, camera_mtx, dist_coeffs, fovs):
        np.savez('mapping/calibration_result', camera_mtx, dist_coeffs, fovs)

    def calculate_camera_fov(self, mtx, img_size):
        fov_x, fov_y, _, _, _ = cv2.calibrationMatrixValues(mtx, img_size, 1, 1)
        return np.array([fov_x, fov_y])

if __name__ == "__main__":
    calibration = CalibrateCamera()
    board_size, num_images = (9, 13), 30
    video_file = "data/videos/calibration.MOV"
    calibration.set_calibration_parameters(video_file, board_size, num_images)
    mtx, dist, rvecs, tvecs, img_size = calibration.calibrate_camera_from_video()
    total_error = calibration.calculate_calibration_error(mtx, dist, rvecs, tvecs)
    fovs = calibration.calculate_camera_fov(mtx, img_size)
    calibration.save_calibration_result(mtx, dist, fovs)

    print(f"Extracted images: {num_images}")
    print(f"Camera matrix: {mtx}")
    print(f"Distortion coefficients: {dist}")
    print(f"Field of view: {fovs}")
    print(f"Estimated total error: {total_error}")