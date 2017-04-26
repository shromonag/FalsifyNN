from neural_nets import squeezedet as nn
from modify_primitives.utils import *
from modify_primitives.heatmap import *
from populateLibrary import *
from PIL import Image, ImageDraw
import numpy as np

DIM = 2
N_SAMPLES = 10

OUT_PIC_PATH = '/home/tommaso/FalsifyNN/pics/out/test'

Lib = populateLibrary()
samples = uniform_sampling(DIM, N_SAMPLES)
im = Image.open("./pics/roads/forest.png")

conf = nn.init()

i = 1;
for sample in samples:
    print(sample)
    loc = generatePicture(Lib, [sample[0], sample[1], 1, 1, 1, 1], OUT_PIC_PATH + str(i) + ".png", 4, 3)
    confidence = nn.classify(OUT_PIC_PATH + str(i) + ".png", conf)
    print(confidence)
    if not confidence:
        score = 0
    else:
        if confidence[0][0] == 0:
            score = int(confidence[0][1]*100)
        else:
            score = 0
    print(score)
    col = rgb(0,100,score)
    im = draw_circle(im, loc[0], loc[1], 10, col)

    i = i + 1
show_image(im)
