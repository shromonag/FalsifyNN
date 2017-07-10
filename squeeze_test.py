from neural_nets import squeezedet as nn
from modify_primitives.utils import *
from modify_primitives.heatmap import *
from ml_primitives.sampling_primitives import *
from populateLibrary import *
from shutil import copyfile
import csv

DIM = 2
N_SAMPLES = 1000

PIC_PATH = './pics/out/squeezetest/'

roads = ['bridge', 'tunnel','island','country','hill']
cars = ['fiat','honda','toyota','peugeot','chrysler']

Lib = populateLibrary()
conf = nn.init()
samples = halton_sampling(DIM, N_SAMPLES)


#for road_i in range(57,62):
#    for car_i in range(7,12):

car_i = 8
road_i = 59

car = cars[car_i-7]
road = roads[road_i-57]

print road + " / " + car

gen_file_name = road + '_' + car + '_'

Z_LIMIT = 0.85

pic_idx = 0

for sample in samples:
    out_pic_name = gen_file_name + "tmp.png"

    real_box = generatePicture(Lib, [sample[0], sample[1]*Z_LIMIT, 1, 1, 1, 1], out_pic_name, road_i, car_i)
    (boxes,probs,cats) = nn.classify(out_pic_name, conf)

    copyfile(out_pic_name, PIC_PATH + "pics/" + gen_file_name + str(pic_idx) + ".png")

    f = open(PIC_PATH + "labels/" + gen_file_name + str(pic_idx) + ".csv",'w+')
    w = csv.writer(f)

    w.writerow([0,1] + real_box)

    for box,prob,cat in zip(boxes,probs,cats):
        row = [cat,prob]
        for b in box:
            row += [b]
        w.writerow(row)

    f.close()

    pic_idx = pic_idx + 1
