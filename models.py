import torch
import torch.nn as nn
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class CNN_I3(nn.Module):
    def __init__(self):
        super(CNN_I3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=config['model']['conv1']['kernel_size'],
                               stride=config['model']['conv1']['stride'], padding=config['model']['conv1']['padding'])
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=config['model']['conv2']['kernel_size'],
                               stride=config['model']['conv2']['stride'], padding=config['model']['conv2']['padding'])
        self.fc = nn.Linear(32 * 5 * 5, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CNN_I4(nn.Module):
    def __init__(self):
        super(CNN_I4, self).__init__() #We initialize the nn.Module class
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=config['model']['conv1']['kernel_size'],
                                stride=config['model']['conv1']['stride'], padding=config['model']['conv1']['padding']) #Basic features
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=config['model']['conv2']['kernel_size'],
                                stride=config['model']['conv2']['stride'], padding=config['model']['conv2']['padding']) #More complex features
        self.fc1 = nn.Linear(64 * 5 * 5, 128) #Number of input features=32*5*5=800. Output features=1.
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x): #'x' is the input
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x)) #We have a 4D tensor: (batch size, channels, height, width).
        x = x.view(x.size(0), -1) #The fc requires a 2D tensor: (batch size, features). batch_size=x.size(0)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
