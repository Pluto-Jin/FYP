import onnxruntime

import onnx
import argparse
import os
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import functional as F
import time
import numpy

import torch

ort_session = onnxruntime.InferenceSession("res18_kd.onnx")

class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#
im_mean = [0.485, 0.456, 0.406]
im_std  = [0.229, 0.224, 0.225]

#'''

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

print()
im = Image.open('dog.jpg')
im_raw = im
im = F.to_tensor(im)
print(im)
im = F.normalize(im, mean=im_mean, std=im_std)
im = torch.unsqueeze(im,0)
print("image shape",im.shape)
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(im)}
start = time.time()
out = ort_session.run(None, ort_inputs)
end = time.time()
print('Inference time on onnx:', end - start)
print(out)

print(class_name[numpy.argmax(out)])


