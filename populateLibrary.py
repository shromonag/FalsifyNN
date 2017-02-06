# Filling up the Library with the roads and cars

from falsifyNN import library, components
from PIL import Image
from scipy import misc

# Create a Library
Lib = library.library()

# Populate the library with roads
Lib.addRoad("./pics/roads/desert.jpg", "Desert Road", 800, 540)
Lib.addRoad("./pics/roads/countryside.jpg", "Countryside Road", 810, 540)
Lib.addRoad("./pics/roads/city.jpg", "City Road", 810, 675)
Lib.addRoad("./pics/roads/cropped_desert.jpg", "Cropped Desert Road", 75, 120)

# Polpulate the library with cars
Lib.addCar("./pics/cars/bmw_rear.png", "BMW")
Lib.addCar("./pics/cars/tesla_rear.png", "Tesla")
Lib.addCar("./pics/cars/suzuki_rear.png", "Suzuki")

print(Lib)
road_type = 1
car_type = 1

roadData = Image.open(Lib.getElement("roads", road_type).road_imageFile)
carData = Image.open(Lib.getElement("cars", car_type).car_imageFile)