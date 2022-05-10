import torch
import torchvision
import torch.onnx
from resnet import *

# An instance of your model
model = resnet34(num_classes = 10, pretrained=False)

# An example input you would normally provide to your model's forward() method
x = torch.rand(1, 3, 32, 32)

# Export the model
torch_out = torch.onnx._export(model, x, "resnet34.onnx", export_params=True)
