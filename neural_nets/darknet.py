import os
import pexpect

YOLO_PATH = '/home/tommaso/darknet/gpu'

# Call Yolo script and loads the neural nets weights
def startYolo():
    process = pexpect.spawn ('./darknet detect cfg/yolo.cfg yolo.weights')
    process.expect('Enter Image Path:')
    return process

# Classifies the image and returns Yolo's output
def yolo(image_path, process):
    process.sendline (image_path)
    process.expect('Enter Image Path:')
    print(process.before)


os.chdir(YOLO_PATH)
process = startYolo()
yolo('data/dog.jpg', process)
yolo('data/dog.jpg', process)
