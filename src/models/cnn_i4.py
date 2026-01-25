import torch.nn as nn

class CNN_I4(nn.Module):
    def __init__(self, config):
        super(CNN_I4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=config['model']['conv1']['kernel_size'],
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
