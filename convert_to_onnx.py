import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from Model_interface import Net

filepath = 'model.onnx'
model = Net.load_from_checkpoint(checkpoint_path='model.pt')
input_sample = torch.zeros(280*280*4)
model.to_onnx(filepath, input_sample, export_params=True)
