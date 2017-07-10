import os
import pexpect

YOLO_PATH = '/home/tommaso/darknet/gpu'

# Call Yolo script and loads the neural nets weights
def init():
    owd = os.getcwd()
    os.chdir(YOLO_PATH)
    process = pexpect.spawn ('./darknet detect cfg/yolo.cfg yolo.weights')
    process.expect('Enter Image Path:')
    os.chdir(owd)
    return process

# Classifies the image and returns Yolo's output
def classify(image_path, process):
    owd = os.getcwd()
    os.chdir(YOLO_PATH)
    process.sendline (image_path)
    process.expect('Enter Image Path:')
    stream = process.before.decode("utf-8")
    os.chdir(owd)
    return parseOut(stream)

# Parse yolo's output
def parseOut(stream):
    # Cut first part of uselsee stream
    stream = stream.split('\n')
    stream = stream[2:-2]

    # Extract labels + confidence values
    res = []
    # for prediction in stream:
    for _ in range(len(stream)/11):

        # get label
        token = stream.pop(0)
        space = token.find(': ')
        label = token[0:space]
        prob = float(token[(space+2):len(token)-2])/100

        # skip line
        token = stream.pop(0)
        # skip i
        token = stream.pop(0)

        # skip line
        token = stream.pop(0)
        # get left box
        token = stream.pop(0)
        left = float(token[5:])

        # skip line
        token = stream.pop(0)
        # get right box
        token = stream.pop(0)
        right = float(token[6:])

        # skip line
        token = stream.pop(0)
        # get top box
        token = stream.pop(0)
        top = float(token[4:])

        # skip line
        token = stream.pop(0)
        # get bot box
        token = stream.pop(0)
        bot = float(token[4:])

        x_len = right-left
        y_len = bot-top
        x_c = left+(x_len/2)
        y_c = top+(y_len/2)

        if label=='car':
            res.append((0,prob,x_c,y_c,x_len,y_len))
        if label=='bicycle':
            res.append((1,prob,x_c,y_c,x_len,y_len))

    return res

process = init()
print(classify('data/dog.jpg', process))
