import pexpect

YOLO_PATH = '/home/tommaso/darknet/gpu'

# Call Yolo script and loads the neural nets weights
def startYolo(yolo_path):
    process = pexpect.spawn ('./darknet detect cfg/yolo.cfg yolo.weights')
    process.expect('Enter Image Path:')
    return process

# Classifies the image and returns Yolo's output
def yolo(image_path, process):
    process.sendline (image_path)
    process.expect('Enter Image Path:')
    print process.before


process = startYolo(YOLO_PATH)
yolo('data/dog.jpg', process)
yolo('data/dog.jpg', process)
