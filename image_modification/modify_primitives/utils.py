# This file contains utility functions to modify the image

import components as comp
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
from components import ImageFile
from collections import namedtuple
import ghalton
import funcy as fn

coord = namedtuple('coord', ['x', 'y'])

prime_no = [2, 3, 5, 7, 11, 13, 17, 19, 23, 39, 31, 37, 41, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 91, 97]

uniform_sampling = lambda no_random, no_samples: np.random.random_sample((no_samples, no_random))

scaleCoord = lambda initialCoord, scale: coord(int(initialCoord.x*scale[0]), int(initialCoord.y*scale[1]))
get_perm_list = lambda no_random: [np.random.choice(prime_no[i], size = prime_no[i], replace=False) for i in range(no_random)]
i_lattice = lambda i, k, irr_nos: fn.flatten([np.true_divide(i, k), lattice_points(i, irr_nos)])
lattice_points = lambda i, irr_nos: [(i*j)%1 for j in irr_nos]

def scale_image(originalObject, scale):
    scaledData = originalObject.data.resize([int(x) for x in np.multiply(originalObject.data.size,np.full((1,2),scale)[0])])
    if originalObject.componentType == 'Road':
        updatedVP = scaleCoord(originalObject.vp, np.full((1,2),scale)[0])
        updatedMIN_X = scaleCoord(originalObject.min_x, np.full((1,2),scale)[0])
        updatedMAX_X = scaleCoord(originalObject.max_x, np.full((1,2),scale)[0])
        updatedLANES = []
        for i in range(len(originalObject.lanes)):
            updatedLANES.append(scaleCoord(originalObject.lanes[i], np.full((1,2),scale)[0]))
        return comp.road(ImageFile(scaledData, originalObject.description), updatedVP, updatedMIN_X, updatedMAX_X, updatedLANES)

    elif originalObject.componentType =='Car':
        return comp.car(ImageFile(scaledData, originalObject.description))

def fit_image(originalObject, fitMeasurement):
    scaledData = originalObject.data.resize(fitMeasurement)
    if originalObject.componentType == 'Road':
        scale = np.true_divide(fitMeasurement, originalObject.data.size).tolist()
        updatedVP = scaleCoord(originalObject.vp, scale)
        updatedMIN_X = scaleCoord(originalObject.min_x, scale)
        updatedMAX_X = scaleCoord(originalObject.max_x, scale)
        updatedLANES = []
        for i in range(len(originalObject.lanes)):
            updatedLANES.append(scaleCoord(originalObject.lanes[i], scale))
        return comp.road(ImageFile(scaledData, originalObject.description), updatedVP, updatedMIN_X, updatedMAX_X, updatedLANES)

    elif originalObject.componentType == 'Car':
        return comp.car(ImageFile(scaledData, originalObject.description))

def generateImage(baseObject, topObject, loc):
    maskValues = topObject.getdata(3)
    mask = Image.new('L', topObject.size, color = 0)
    mask.putdata(maskValues)
    baseObject.paste(topObject, loc, mask)
    return baseObject

def shift_xz(baseObject, topObject, x, z):
    topObjectsize = topObject.data.size
    baseObjectmin_x = baseObject.min_x
    baseObjectmax_x = baseObject.max_x
    x_min = baseObjectmin_x.x
    x_max = baseObjectmax_x.x - topObjectsize[0]

    # For moving along the x-azix
    x_left = x_min + int((x_max - x_min)*x)
    x_right = x_left + topObjectsize[0]
    lower = min(baseObjectmin_x.y, baseObjectmax_x.y)
    upper = lower - topObjectsize[1]

    # For moving along the z-axis
    # Computing diagonally opposite points
    # Computing (new_upper, new_left)
    new_upper = upper - (upper - baseObject.vp.y) * z

    slope_ul = np.true_divide(baseObject.vp.y - upper, baseObject.vp.x - x_left)
    if slope_ul != 0:
        new_left = x_left + (new_upper - upper)/slope_ul

    slope_ur = np.true_divide(baseObject.vp.y - upper, baseObject.vp.x - x_right)
    if slope_ur != 0:
        new_right = x_right + (new_upper - upper)/slope_ur

    slope_lr = np.true_divide(baseObject.vp.y - lower, baseObject.vp.x - x_right)
    new_lower = lower + slope_lr *(new_right - x_right)

    loc = (int(new_left), int(new_upper))
    compressedImage = topObject.data.resize((int(new_right - new_left), int(new_lower - new_upper)))
    return (loc, compressedImage)

def modifyImageLook(imageData, color, contrast, brightness, sharpness):
    colorMod = ImageEnhance.Color(imageData)
    imageData = colorMod.enhance(color)

    contrastMod = ImageEnhance.Contrast(imageData)
    imageData = contrastMod.enhance(contrast)

    brightnessMod = ImageEnhance.Brightness(imageData)
    imageData = brightnessMod.enhance(brightness)

    sharpnessMod = ImageEnhance.Sharpness(imageData)
    imageData = sharpnessMod.enhance(sharpness)

    return imageData

def halton_sampling(no_random, no_samples):
    perm = get_perm_list(no_random)
    seq = ghalton.GeneralizedHalton(perm)
    return seq.get(no_samples)

def lattice_sampling(no_random, no_samples, k):
    if k is None:
        k = no_samples
    irrational_nums = [(np.sqrt(prime_no[i]) + 1)/2 for i in range(1, no_random)]
    return [i_lattice(i+1, k, irrational_nums) for i in range(no_samples)]