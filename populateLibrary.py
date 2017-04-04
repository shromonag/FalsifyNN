# Filling up the Library with the roads and cars


from PIL import Image, ImageDraw
from scipy import misc
from collections import namedtuple
from modify_primitives.components import ImageFile
from modify_primitives import library
import numpy as np
from ml_primitives import uniform_sampling
from modify_primitives.utils import coord, scale_image, fit_image, generateImage, shift_xz, modifyImageLook

def populateLibrary():
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

    return Lib



def generatePicture(params,pic_path, road_type = 0, car_type = 0):
    Lib = populateLibrary()
    old_road = Lib.getElement("roads", road_type)
    car = Lib.getElement("cars", car_type)
    (loc, new_carimage) = shift_xz(old_road, car, params[0], params[1])
    new_image = generateImage(old_road.data, new_carimage, loc)
    ModifiedImage = modifyImageLook(new_image, 1, 1, 1, 1)
    #ModifiedImage.show()
    ModifiedImage.save(pic_path)
