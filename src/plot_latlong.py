import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors 
from mpl_toolkits.basemap import Basemap
import pprint

### Load our data
defect_locations = np.genfromtxt("latlong.csv",
                                 delimiter=",",
                                 dtype=[("type","|S256"),("lat",np.float32),("lon",np.float32)],
                                 usecols=(0,1,2))



#### Check mins and maxes
#print type(defect_locations)
#print type(defect_locations[0])
#print defect_locations[0]
#latitudes = map(lambda x: x[0], defect_locations)
#longitudes = map(lambda x: x[1], defect_locations)
#print latitudes[0]
#print longitudes[0]
#max_longitude = max(longitudes)
#min_longitude = min(longitudes)
#max_latitude  = max(latitudes)
#min_latitude  = min(latitudes)
#print "Max longitude: " + str(max_longitude)
#print "Min longitude: " + str(min_longitude)
#print "Max latitude: "  + str(max_latitude)
#print "Min latitude: "  + str(min_latitude)

fig = plt.figure()

### Orient our data onto a map
the_map = Basemap(width = 12000000,
                  height=9000000,
                  projection="lcc",
                  lat_1 = 25, # lower-left corner longitude
                  lat_2 = 55,  # lower-left corner latitude
                  lat_0 = 40,  # upper-right corner longitude
                  lon_0 = -80,  # upper-right corner latitude
                  resolution = "c"
                  )

### Add map features -- e.g. coastlines
the_map.drawcoastlines()
the_map.drawcountries()
the_map.drawstates()
the_map.fillcontinents(color = "coral")
the_map.drawmapboundary(fill_color = "steelblue")

### Add a point on our map
lon = -75
lat = 40
x,y = the_map(lon,lat)
the_map.plot(x,y,'yo',markersize=8)

### Project our data
#x, y = the_map(defect_locations["lon"], defect_locations["lat"])

#longitudes = [ -1*x for x in longitudes ]
#x,y = the_map(longitudes,latitudes)

defect_types = list(set(defect_locations["type"]))
type_to_idx = {k:v for k,v in zip(defect_types,range(len(defect_types)))}
idx_to_color = {k:v for k,v in zip(type_to_idx.values(), colors.cnames.keys()[:len(type_to_idx.values())])}
c = []
for x in defect_locations:
    c.append( idx_to_color[ type_to_idx[ x["type"] ] ] )

x,y = the_map((defect_locations["lon"]*-1),defect_locations["lat"])
for i in xrange(len(x)):
    the_map.plot(x[i], 
                 y[i], 
                 "o",
                 color=c[i],
                 markersize=4)

#the_map.plot(longitudes,
#             latitudes,
#             "o",            # Marker shape
#             color="g", # Marker color
#             markersize=8,   # Marker size
#             )

### Display
plt.tight_layout()
plt.show()
