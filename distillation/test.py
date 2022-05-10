from resnet import snet,snet2,resnet18

import torch

model = snet(10,False)
torch.save(model.state_dict(),'./snet.pth')
model = snet2(10,False)
torch.save(model.state_dict(),'./snet2.pth')
model = resnet18(num_classes=10)
torch.save(model.state_dict(),'./resnet18.pth')
