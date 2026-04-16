
from torch.utils.data import DataLoader
from make_data_set import ImageRestorationDataset
import torch.nn as nn
import torch.optim as optim


train_dataset = ImageRestorationDataset("images")
train_loader = DataLoader(train_dataset, 1, True)


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(64, 64, 3, 1,1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, 3, 1,1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + identity
        return out



class RestorationCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.start_conv = nn.Conv2d(3,64,3,1,1)
        self.relu = nn.ReLU()
        self.resid_block1 = ResidualBlock()
        self.resid_block2 = ResidualBlock()
        self.end_conv = nn.Conv2d(64,3,3,1,1)

    def forward(self, x):
        identity = x
        edit = self.start_conv(x)
        edit = self.relu(edit)
        edit = self.resid_block1(edit)
        edit = self.resid_block2(edit)
        edit = self.end_conv(edit)
        correction = identity + edit
        return correction


robot = RestorationCNN()
loss = nn.MSELoss()
optimizer = optim.Adam(robot.parameters())

for epoha in range(100):
    full_error = 0
    cnt = 0
    for deg,orig in train_loader:
        error = loss(robot(deg), orig)
        optimizer.zero_grad()
        error.backward()
        optimizer.step()

        full_error += error.item()
        cnt += 1
        print(error)
    full_error = full_error / cnt
    print(f'epoha {epoha} loss = {full_error}')

