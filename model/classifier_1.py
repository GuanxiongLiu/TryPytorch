import torch
import torch.nn as nn
import torch.nn.functional as F



class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.def_layers()
    
    def def_layers(self):
        # conv
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        # pool
        self.pool = nn.MaxPool2d(2)
        # fc
        self.fc1 = nn.Linear(4*4*32, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x