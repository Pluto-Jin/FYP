import torch
import torchvision
import torch.onnx
from kd import *

# An instance of your model
model = create_model('res18',num_classes = 10)
model = load_model(model = model, model_filepath = 'res18_kd.pt', device = torch.device("cuda:0"))

# An example input you would normally provide to your model's forward() method
x = torch.rand(1, 3, 32, 32)

# Export the model
torch_out = torch.onnx._export(model, x, "res18_kd.onnx", export_params=True)
