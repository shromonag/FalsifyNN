from neural_nets import squeezedet as nn
from modify_primitives.utils import *
from modify_primitives.heatmap import *
from ml_primitives.sampling_primitives import *
from populateLibrary import *
from shutil import copyfile
import csv

DIM = 5
N_SAMPLES = 1000
FOOL_PICS = 100

car_size = [194,140]

FOOL_PIC_PATH = './pics/out/fool/'
NOTFOOL_PIC_PATH = './pics/out/notfool/'

GEN_FILE_NAME = 'hill_chrysler_'

Lib = populateLibrary()

conf = nn.init()
samples = halton_sampling(DIM, N_SAMPLES)

# set a limit to z
Z_LIMIT = 0.5

i_fool = 0      # number of fooling pics
i_not_fool = 0  # number of not fooling pics

for sample in samples:

    sample[1] = sample[1]*Z_LIMIT
    out_pic_name = GEN_FILE_NAME + "tmp.png"


    car_centroid = generatePicture(Lib, [sample[0], sample[1], sample[2]*0.25 + 0.5, 1, 1, 1], out_pic_name, 61, 11)

    confidence = nn.classify(out_pic_name, conf)
    print confidence

    left = car_centroid[0] - (car_size[0]/2)    #left
    top = car_centroid[1] + (car_size[1]/2)    #top
    right = car_centroid[0] + (car_size[0]/2)    #right
    bottom = car_centroid[1] - (car_size[1]/2)    #bottom

    pic_idx = 0
    OUT_PIC_PATH = ""
    if (not confidence) or (len(confidence) > 1):
        i_fool = i_fool + 1
        pic_idx = i_fool
        OUT_PIC_PATH = FOOL_PIC_PATH
    else:
        i_not_fool = i_not_fool + 1
        pic_idx = i_not_fool
        OUT_PIC_PATH = NOTFOOL_PIC_PATH

    copyfile(out_pic_name, OUT_PIC_PATH + "pics/" + GEN_FILE_NAME + str(pic_idx) + ".png")

    f = open(OUT_PIC_PATH + "labels/" + GEN_FILE_NAME + str(pic_idx) + ".txt",'w+')
    w = csv.writer(f)
    w.writerow([left,top,right,bottom])
    f.close()

    if i_fool == FOOL_PICS:
        break

    print str(i_fool) + " / " + str(i_not_fool)
