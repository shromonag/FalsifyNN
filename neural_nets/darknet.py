import os
import pexpect

YOLO_PATH = '/home/tommaso/darknet/gpu'

# Call Yolo script and loads the neural nets weights
def init():
    os.chdir(YOLO_PATH)
    process = pexpect.spawn ('./darknet detect cfg/yolo.cfg yolo.weights')
    process.expect('Enter Image Path:')
    return process

# Classifies the image and returns Yolo's output
def classify(image_path, process):
    os.chdir(YOLO_PATH)
    process.sendline (image_path)
    process.expect('Enter Image Path:')
    stream = process.before.decode("utf-8")
    return parseOut(stream)

# Parse yolo's output
def parseOut(stream):
    # Cut first part of uselsee stream
    stream = stream.split('\n')
    stream = stream[2:-2]

    # Extract labels + confidence values
    res = []
    for prediction in stream:
        space = prediction.find(': ')
        label = prediction[0:space]
        if label == u'car':
            confidence = prediction[space+2:-2]
            res.append((0,float(confidence)/100))
    return res

#process = startYolo()
#print(yolo('data/dog.jpg', process))
#print(yolo('data/dog.jpg', process))
