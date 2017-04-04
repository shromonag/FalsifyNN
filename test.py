from populateLibrary import generatePicture
from ml_primitives import uniform_sampling
from neural_nets import squeezedet as nn


PIC_PATH = '/home/tommaso/FalsifyNN/pics/out/test.png'

# Initialize neural network
conf = nn.init()

# Initialize samples
DIM = 2
N_SAMPLES = 30

samples = uniform_sampling(DIM, N_SAMPLES)

# Test loop
for sample in samples:
    generatePicture(sample,PIC_PATH)
    print( nn.classify(PIC_PATH,conf) )
