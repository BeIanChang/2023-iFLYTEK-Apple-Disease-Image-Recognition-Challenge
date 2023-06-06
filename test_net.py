from torchsummary import summary
from module import to_device, Network

INPUT_SIZE = INPUT_SHAPE = (3, 256, 256)
model = to_device(Network('myResNet', 9), 'cuda') 
print(summary(model.cuda(), (INPUT_SHAPE)))