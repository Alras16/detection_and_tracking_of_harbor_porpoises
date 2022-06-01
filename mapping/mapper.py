import os
import numpy as np

from gps import GPS
from kml import Kml
from data_logger import FlightLog
from argparse import ArgumentParser
from calibration import CalibrateCamera

parser = ArgumentParser()
parser.add_argument('flight_log', help='Decrypted flight log file for data logging')

args = parser.parse_args()

flight_log_file = "data/flight_logs/decrypted/" + args.flight_log
calibration_video_file = "data/videos/calibration.MOV"
if os.path.exists('mapping/calibration_result.npz'):
    numpy_file = np.load('mapping/calibration_result.npz')
    camera_mtx = numpy_file['arr_0']
    dist_coeffs = numpy_file['arr_1']
    fovs = numpy_file['arr_2']
else:
    calibration = CalibrateCamera()
    calibration.set_calibration_parameters(calibration_video_file, board_size=(9, 13), num_images=30)
    camera_mtx, dist_coeffs, _, _, img_size = calibration.calibrate_camera_from_video()
    fovs = calibration.calculate_camera_fov(camera_mtx, img_size)
    calibration.save_calibration_result(camera_mtx, dist_coeffs, fovs)

logging = FlightLog()
logging.get_flight_log_data(flight_log_file)
flight_time, position, altitude, rotation_uav, rotation_cam, is_video = logging.get_log_data()

image_point = (1200, 1500)

mapping = GPS()
mapping.set_image_size(img_size[0], img_size[1])
mapping.set_camera_parameters(camera_mtx, dist_coeffs, fovs)
gps_point = mapping.get_gps_point(image_point, altitude, rotation_uav, rotation_cam, position)




