from modify_primitives.utils import *
from ml_primitives.sampling_primitives import *
from ml_primitives.search_algo import *

def call_NN(Lib, scene, objects, out_path, other_params, NN_classify):
    image = generateGenImage(Lib, out_path, scene, objects, other_params)
    return NN_classify(out_path)

def (in_path, NN_classify):
    return NN_classify(in_path)
