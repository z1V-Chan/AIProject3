from torch import nn
import torch.nn.functional as F


class YourNet(nn.Module):
    ###################### Begin #########################
    # You can create your own network here or copy our reference model (LeNet5)
    # We will conduct a unified test on this network to calculate your score

    def __init__(self):
        super(YourNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.conv2 = nn.Conv2d(3, 4, 3)
        self.fc1 = nn.Linear(4 * 5 * 5, 51)  # 5x5 image dimension
        self.fc2 = nn.Linear(51, 35)
        self.fc3 = nn.Linear(35, 10)
        # self.fc1 = nn.Linear(4 * 5 * 5, 120)  # 5x5 image dimension
        # self.fc2 = nn.Linear(120, 90)
        # self.fc3 = nn.Linear(90, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    ######################  End  #########################
