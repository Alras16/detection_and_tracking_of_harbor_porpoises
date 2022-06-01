import time
import random
import simplekml

class Kml:
    def __init__(self):
        ...

    def create_kml_file(self, tracks, flight_log_filename):
        kml = simplekml.Kml()
        for track, id in tracks:
            # Create a new line string with object 
            ls = kml.newlinestring(name="Porpoise " + str(id))
            for lat, lon in track:
                ls.coords.addcoordinates([(lon, lat)]) # Longnitude first, latitude after
            # Choose random RGB color and apply it to the line along with the width of it
            random.seed(time.clock())
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            ls.style.linestyle.color = simplekml.Color.rgb(r, g, b)
            ls.style.linestyle.width = 5
        filename = flight_log_filename.split('/')[-1]
        filename = filename.split('.')[0]
        kml.save('mapping/kml_files/' + filename + '.kml')
