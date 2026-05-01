import torch.cuda
from torch.utils.data import DataLoader
from make_data_set import *
import torch.nn as nn
import torch.optim as optim
import torch
import time




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
        self.resid_block3 = ResidualBlock()
        self.resid_block4 = ResidualBlock()
        self.resid_block5 = ResidualBlock()
        self.resid_block6 = ResidualBlock()
        self.resid_block7 = ResidualBlock()
        self.end_conv = nn.Conv2d(64,3,3,1,1)

    def forward(self, x):
        identity = x
        edit = self.start_conv(x)
        edit = self.relu(edit)
        edit = self.resid_block1(edit)
        edit = self.resid_block2(edit)
        edit = self.resid_block3(edit)
        edit = self.resid_block4(edit)
        edit = self.resid_block5(edit)
        edit = self.resid_block6(edit)
        edit = self.resid_block7(edit)
        edit = self.end_conv(edit)
        output = torch.clamp(identity + 1.5 * edit, 0, 1)
        return output



def model_training(model, kol_vo_epoh):
    last_loss = None
    for epoha in range(kol_vo_epoh):
        full_error = 0
        cnt = 0
        for deg, orig in train_loader:
            deg = deg.to(device)
            orig = orig.to(device)




            error = loss(model(deg), orig)
            optimizer.zero_grad()
            error.backward()
            optimizer.step()
            full_error += error.item()
            cnt += 1
        full_error = full_error / cnt



        if last_loss is None:
            print(f'epoha {epoha} loss = {full_error}')
        else:
            change_percent = ((last_loss - full_error) / last_loss) * 100

            if change_percent >= 0:
                print(f'epoha {epoha} loss = {full_error} (↓ {abs(change_percent):.2f}%)')
            else:
                print(f'epoha {epoha} loss = {full_error} (↑ {abs(change_percent):.2f}%)')

        last_loss = full_error
    return None


if __name__ == '__main__':
    train_dataset = ImageRestorationDataset("clean_img", 'degr_img')
    train_loader = DataLoader(train_dataset, 4, True, num_workers=2)


    robot = RestorationCNN()
    robot = robot.to(device)


    loss = nn.L1Loss()
    optimizer = optim.Adam(robot.parameters(), lr = 0.0006)


    model_training(robot, 30)
    torch.save(robot.state_dict(), "robot_weights_exp3.pth")

