# from populateLibrary import generatePicture
# from ml_primitives import uniform_sampling
# # from neural_nets import squeezedet as nn
#
# PIC_PATH = '/home/tommaso/FalsifyNN/pics/out/test.png'
#
# # Initialize neural network
# # conf = nn.init()
#
# # Initialize samples
# DIM = 2
# N_SAMPLES = 30
#
# samples = uniform_sampling(DIM, N_SAMPLES)
#
# # Test loop
# for sample in samples:
#     generatePicture(sample,PIC_PATH)
#     # print( nn.classify(PIC_PATH,conf) )

from modify_primitives.utils import *
from modify_primitives.heatmap import *
from populateLibrary import *
from PIL import Image, ImageDraw
import numpy as np

Lib = populateLibrary()
samples = uniform_sampling(2, 10)
im = Image.open("./pics/roads/forest.png")
for sample in samples:
    print(sample)
    loc = generatePicture(Lib, [sample[0],sample[1],1,1,1,1], "./pics/out/out.png", 4,3)
    #print(loc)
    col = rgb(0,100,np.random.random_integers(0,100))
    im = draw_circle(im, loc[0], loc[1], 10, col)
show_image(im)