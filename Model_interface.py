import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

mean = 0.1307
standard_devation = 0.3081

class Net(pl.LightningModule):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = nn.Dropout2d(0.25)
    self.dropout2 = nn.Dropout2d(0.5)
    self.fc1 = nn.Linear(9216, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = x.reshape(280, 280, 4)
    x = torch.narrow(x, dim=2, start=3, length=1)
    x = x.reshape(1, 1, 280, 280)
    x = F.avg_pool2d(x, 10, stride=10)
    x = x / 255
    x = (x - 0.1307) / 0.3081

    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    output = F.softmax(x, dim=1)
    return output