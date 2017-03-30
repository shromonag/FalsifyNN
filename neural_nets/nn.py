import squeezedet as nn
#import darknet as nn
#import kittibox as nn

conf = nn.init()
print( nn.classify('/home/tommaso/FalsifyNN/pics/out/cory.png',conf) )
