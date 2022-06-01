import utm
import cv2
import numpy as np

class GPS:
    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_size = None
        self.fov = None

    def set_camera_parameters(self, camera_mtx, dist_coeffs, fov):
        self.fov = [angle * np.pi / 180 for angle in fov]
        self.camera_matrix = camera_mtx
        self.dist_coeffs = dist_coeffs

    def set_image_size(self, width, height):
        self.image_size = (width, height)

    def get_rotation_matrix(self, uav_euler_angles, cam_euler_angles):
        uav_yaw, uav_pitch, uav_roll = uav_euler_angles
        uav_rotate_x = np.array([[np.cos(uav_roll), 0, np.sin(uav_roll)], [0, 1, 0], [-np.sin(uav_roll), 0, np.cos(uav_roll)]]) 
        uav_rotate_y = np.array([[1, 0, 0], [0, np.cos(uav_pitch), -np.sin(uav_pitch)], [0, np.sin(uav_pitch), np.cos(uav_pitch)]]) 
        uav_rotate_z = np.array([[np.cos(uav_yaw), -np.sin(uav_yaw), 0], [np.sin(uav_yaw), np.cos(uav_yaw), 0], [0, 0, 1]])
        rotate_world_to_uav = np.matmul(uav_rotate_z, np.matmul(uav_rotate_y, uav_rotate_x))
        cam_yaw, cam_pitch, cam_roll = cam_euler_angles
        cam_rotate_x = np.array([[1, 0, 0], [0, np.cos(cam_roll), -np.sin(cam_roll)], [0, np.sin(cam_roll), np.cos(cam_roll)]]) 
        cam_rotate_y = np.array([[np.cos(cam_pitch), 0, np.sin(cam_pitch)], [0, 1, 0], [-np.sin(cam_pitch), 0, np.cos(cam_pitch)]]) 
        cam_rotate_z = np.array([[np.cos(cam_yaw), -np.sin(cam_yaw), 0], [np.sin(cam_yaw), np.cos(cam_yaw), 0], [0, 0, 1]])
        rotate_uav_to_camera = np.matmul(cam_rotate_z, np.matmul(cam_rotate_y, cam_rotate_x))
        return np.matmul(rotate_world_to_uav, rotate_uav_to_camera)

    def get_unit_vector(self, image_point):
        undistort_point = cv2.undistortPoints(np.array([[image_point]], dtype=np.float32), self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)[0][0]
        image_point_from_center = undistort_point - np.array([self.image_size[0]/2, self.image_size[1]/2])
        image_plane_size_in_meters = np.array([np.tan(self.fov[0] / 2) * 2, np.tan(self.fov[1] / 2) * 2])
        x = image_point_from_center[0] / self.image_size[0] * image_plane_size_in_meters[0]
        y = 1
        z = - image_point_from_center[1] / self.image_size[1] * image_plane_size_in_meters[1]
        return np.array([x, y, z])

    def get_world_point(self, image_point, drone_height, uav_yaw_pitch_roll, cam_yaw_pitch_roll, pos, return_zone=False):
        uav_yaw_pitch_roll = (-uav_yaw_pitch_roll[0], uav_yaw_pitch_roll[1], uav_yaw_pitch_roll[2])
        cam_yaw_pitch_roll = (cam_yaw_pitch_roll[0], -cam_yaw_pitch_roll[1], cam_yaw_pitch_roll[2])
        rotation_matrix = self.get_rotation_matrix(uav_yaw_pitch_roll, cam_yaw_pitch_roll)
        unit_vector = self.get_unit_vector(image_point)
        rotated_vector = np.matmul(rotation_matrix, unit_vector)
        ground_vector = rotated_vector / rotated_vector[2] * -drone_height
        print(ground_vector)
        east_north_zone = self.convert_gps(*pos)
        print(f"UTM UAV: {east_north_zone}")
        world_point = ground_vector[:2] + np.array(east_north_zone[:2])
        print(world_point)
        if return_zone:
            return world_point, east_north_zone[2:]
        else:
            return world_point

    def get_gps_point(self, image_point, drone_height, uav_yaw_pitch_roll, cam_yaw_pitch_roll, pos):
        world_point, zone = self.get_world_point(image_point, drone_height, uav_yaw_pitch_roll, cam_yaw_pitch_roll, pos, True)
        lat, lon = self.convert_utm(world_point[0], world_point[1], zone)
        return lat, lon

    @staticmethod
    def convert_gps(lat, lon):
        east_north_zone = utm.from_latlon(lat, lon)
        return east_north_zone

    @staticmethod
    def convert_utm(east, north, zone):
        lat, lon = utm.to_latlon(east, north, *zone)
        return lat, lon

if __name__ == "__main__":
    camera_matrix = np.array([[2.83950728e+03, 0.00000000e+00, 2.04272048e+03], [0.00000000e+00, 2.83981677e+03, 1.07532423e+03], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist_coeffs = np.array([2.90634334e-02, -1.19199243e-01, -1.98079581e-04, -1.04532371e-04, 1.39185482e-01])
    cam_angles = [angle * np.pi / 180 for angle in (float(0), float(-43.7), float(0))]
    uav_angles = [angle * np.pi / 180 for angle in (float(14), float(-9), float(-7.3))]
    fov = np.array([71.60207342836507, 41.64415189407483])
    pos = ([float(56.705429), float(8.229587)])
    image_point = (2720,1880)
    image_size = (3840, 2160)
    height = 4.0

    mapping = GPS()
    mapping.set_image_size(image_size[0], image_size[1])
    mapping.set_camera_parameters(camera_matrix, dist_coeffs, fov)
    gps_point = mapping.get_gps_point(image_point, height, uav_angles, cam_angles, pos)
    print(gps_point)