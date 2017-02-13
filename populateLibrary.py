# Filling up the Library with the roads and cars


from PIL import Image, ImageDraw
from scipy import misc
from collections import namedtuple
from falsifyNN.components import ImageFile
from falsifyNN import library
import numpy as np
from falsifyNN.utils import coord, scale_image, fit_image, generateImage, shift_xz, modifyImageLook

# Create a Library
Lib = library()

# Populate the library with roads
Lib.addRoad(ImageFile(Image.open("./pics/roads/desert.jpg"), "Desert Road"), coord(800, 540), coord(100, 950), coord(1500,950), [coord(800,950)])
Lib.addRoad(ImageFile(Image.open("./pics/roads/countryside.jpg"), "Countryside Road"), coord(810, 540),coord(100, 1000), coord(1500, 1000), [coord(775,1000)])
Lib.addRoad(ImageFile(Image.open("./pics/roads/city.jpg"), "City Road"), coord(810, 675), coord(100, 925), coord(1500, 925), [coord(508, 925), coord(1025, 925)])
Lib.addRoad(ImageFile(Image.open("./pics/roads/cropped_desert.jpg"), "Cropped Desert Road"), coord(75, 120), coord(100,500), coord(1500, 500), [coord(66,500)])

# Populate the library with cars
Lib.addCar(ImageFile(Image.open("./pics/cars/bmw_rear.png"), "BMW"))
Lib.addCar(ImageFile(Image.open("./pics/cars/tesla_rear.png"), "Tesla"))
Lib.addCar(ImageFile(Image.open("./pics/cars/suzuki_rear.png"), "Suzuki"))




print(Lib)
road_type = 0
car_type = 0

old_road = Lib.getElement("roads", road_type)
print(old_road.description)
print(old_road.data)
print(old_road.data.size)
print(old_road.vp)
print(old_road.min_x)
print(old_road.max_x)
print(old_road.lanes)
new_road = scale_image(old_road, 0.5)
print(new_road.description)
print(new_road.data)
print(new_road.data.size)
print(new_road.vp)
print(new_road.min_x)
print(new_road.max_x)
print(new_road.lanes)
new_road1 = fit_image(old_road, (800,533))
print(new_road1.description)
print(new_road1.data)
print(new_road1.data.size)
print(new_road1.vp)
print(new_road1.min_x)
print(new_road1.max_x)
print(new_road1.lanes)

car = Lib.getElement("cars", car_type)


(loc, new_carimage) = shift_xz(old_road, car, 0.5, 0.5)

new_image = generateImage(old_road.data, new_carimage, loc)
ModifiedImage = modifyImageLook(new_image, 1, 1, 0.25, 1)

ModifiedImage.show()
