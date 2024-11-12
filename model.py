import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, opt, in_ch=1, out_ch=64, in_len=360):
        super(SimpleCNN, self).__init__()

        # Convolutional Layer 1 (Reduced model complexity)
        self.conv1 = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(out_ch)

        # Fully Connected Layer
        self.fc1 = nn.Linear(out_ch * (in_len // 2), opt.classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        # Flatten before fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc1(x)

        return x

