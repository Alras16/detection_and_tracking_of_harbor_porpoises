import csv
import numpy as np

from datetime import datetime, timedelta

class FlightLog:
    def __init__(self):
        self.rotation_camera = []
        self.rotation_uav = []
        self.time_stamp = []
        self.fly_time = []
        self.position = []
        self.is_video = []
        self.height = []
        self.lines = []
        self.data = []

    def get_flight_log_data(self, flight_log):
        remove_null_bytes(flight_log)
        lines_passed, lines_skipped = 0, 0
        with open(flight_log, encoding="iso8859_10") as csv_file:
            reader = csv.DictReader(csv_file, delimiter=",")       
            for row in reader:
                if row['CUSTOM.updateTime']:
                    try:
                        self.time_stamp.append(datetime.strptime(row['CUSTOM.updateTime'], '%Y/%m/%d %H:%M:%S.%f'))
                        self.fly_time.append(float(row['OSD.flyTime [s]']))
                        self.position.append((float(row['OSD.latitude']), float(row['OSD.longitude'])))
                        self.height.append(float(row['OSD.height [m]']))
                        self.rotation_camera.append([angle * np.pi / 180 for angle in (float(row['GIMBAL.yaw']), float(row['GIMBAL.pitch']), float(row['GIMBAL.roll']))])
                        self.rotation_uav.append([angle * np.pi / 180 for angle in (float(row['OSD.yaw']), float(row['OSD.pitch']), float(row['OSD.roll']))])
                        self.is_video.append(True if row['CUSTOM.isVideo'] else False)
                    except ValueError:
                        lines_skipped += 1
                        continue
                    self.lines.append(lines_passed)
                    lines_passed += 1
        #print(f'Number of lines skipped: {lines_skipped}')
        #print(f'Number of lines passed: {lines_passed}')

    def display_data(self):
        self.data.append(['Time Stamp', 'Fly Time [s]', 'Position', 'Height [m]', 'Rotation [deg]', 'Video Recording'])
        for x in self.lines:
            self.data.append([self.time_stamp[x], self.fly_time[x], self.position[x], self.height[x], self.rotation_uav[x], self.rotation_camera[x], self.is_video[x]])
        print(self.data)

    def get_log_data(self):
        return self.time_stamp, self.fly_time, self.position, self.height, self.rotation_uav, self.rotation_camera, self.is_video
    
def remove_null_bytes(flight_log):
    with open(flight_log, 'rb') as fi:
        data = fi.read()
    with open(flight_log, 'wb') as fo:
        fo.write(data.replace(b'\x00', b''))

if __name__ == "__main__":
    flight_log = 'flight_logs/DJIFlightRecord_2021-05-29_[17-19-31].csv'
    log = FlightLog()
    log.get_flight_log_data(flight_log)
    log.display_data()