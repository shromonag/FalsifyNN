from neural_nets import squeezedet as nn
from modify_primitives.utils import *
from populateLibrary import *



# params[0]: car x pos
# params[1]: car y pos
# scene[0]: road
# scene[1]: car
def foo(params,scene):
    Lib = populateLibrary()
    conf = nn.init()

    out_pic_name = "tmp.png"
    real_box = generatePicture(Lib, [params[0], params[1], 1, 1, 1, 1], out_pic_name, scene[0], scene[1])
    (boxes,probs,cats) = nn.classify(out_pic_name, conf)

    # extract max prob box
    max_prob = 0

    for box, prob, cat in zip(boxes, probs, cats):
        if (prob > max_prob) and (cat == 0):
            max_prob = prob
            max_prob_box = box
            max_prob_cat = cat

    return max_prob




from modify_primitives import clustering_bo as BO

bo = BO.bo_class()
BO.init_BO(foo)