from neural_nets import squeezedet as nn
from modify_primitives.utils import *
from modify_primitives.heatmap import *
from ml_primitives.sampling_primitives import *
from populateLibrary import *
import csv

DIM = 2
N_SAMPLES = 50

PIC_NAME_BASE = '0000000'
OUT_PIC_PATH = './pics/out/'
TEST_FILE_NAME = './test'

Lib = populateLibrary()

conf = nn.init()
samples = halton_sampling(DIM, N_SAMPLES)


for i in range(134,135):

    out_pic_name = OUT_PIC_PATH + "tmp.png"
    pic_name = PIC_NAME_BASE + str(i) + ".png"
    im = Image.open("./pics/roads/forest/" + pic_name)
    f = open(TEST_FILE_NAME + str(i) + '.csv','w+')
    w = csv.writer(f)

    print("FIGURE" + str(i))

    for sample in samples:
        loc = generatePicture(Lib, [sample[0], sample[1], 1, 1, 1, 1], out_pic_name, i-130, 3)
        confidence = nn.classify(out_pic_name, conf)
        print(confidence)
        if not confidence:
            score = 0
        else:
            if confidence[0][0] == 0:
                score = int(confidence[0][1]*100)
            else:
                score = 0
        #print(score)
        col = rgb(0,100,score)
        im = draw_circle(im, loc[0], loc[1], 5, col)	
	w.writerow(sample+[score])

    im.save(OUT_PIC_PATH + pic_name)
    f.close()
