from modify_primitives.utils import *
from modify_primitives.clustering_bo import *
from populateLibrary import *
from neural_nets import squeezedet as nn



# params[0]: car x pos
# params[1]: car y pos
# scene[0]: road
# scene[1]: car
def wrapper(params):
    Lib = populateLibrary()

    params = params[0]

    out_pic_name = "tmp.png"
    real_box = generatePicture(Lib, [params[0], params[1], 1, 1, 1, 1], out_pic_name, 58, 10)
    (boxes,probs,cats) = nn.classify(out_pic_name, conf)

    # extract max prob box
    max_prob = 0

    for box, prob, cat in zip(boxes, probs, cats):
        if (prob > max_prob) and (cat == 0):
            max_prob = prob
            max_prob_box = box
            max_prob_cat = cat

    return max_prob



conf = nn.init()

opt = bo_class(input_dim=2)
opt.init_BO(f=wrapper)

for _ in range(10):
    opt.run_BO(max_iter=1)
    print opt.bo.suggested_sample
