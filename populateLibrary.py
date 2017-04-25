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
    Lib.addRoad(ImageFile(Image.open("./pics/roads/forest.png"), "Forest Road"), coord(675, 238), coord(85, 504), coord(800, 504), [coord(340, 499)])

    # Populate the library with cars
    Lib.addCar(ImageFile(Image.open("./pics/cars/bmw_rear.png"), "BMW"))
    Lib.addCar(ImageFile(Image.open("./pics/cars/tesla_rear.png"), "Tesla"))
    Lib.addCar(ImageFile(Image.open("./pics/cars/suzuki_rear.png"), "Suzuki"))
    Lib.addCar(ImageFile(Image.open("./pics/cars/modified_bmw.png"), "Modified BMW"))

    return Lib
