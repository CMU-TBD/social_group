import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):

    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, 3, padding=1)
        self.pool1 = nn.MaxPool3d((1,2,2), stride=(1,2,2))   

        self.conv2 = nn.Conv3d(64, 128, 3, padding=1)
        #self.pool2 = nn.MaxPool3d(2, stride=2)   
        self.pool2 = nn.MaxPool3d((1,2,2), stride=(1,2,2))   

        self.conv3_1 = nn.Conv3d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv3d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool3d(2, stride=2)   

        self.conv4_1 = nn.Conv3d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv3d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool3d(2, stride=2)   

        self.conv5_1 = nn.Conv3d(512, 256, 3, padding=1)
        self.conv5_2 = nn.Conv3d(256, 128, 3, padding=1)
        self.pool5 = nn.MaxPool3d(2, stride=2)   

        self.t_conv1 = nn.ConvTranspose2d(128, 256, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.t_conv5 = nn.ConvTranspose2d(64, 1, 2, stride=2)
        return

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = self.pool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = self.pool5(x)

        x = x[:, :, -1, :, :]

        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = F.relu(self.t_conv4(x))
        x = F.sigmoid(self.t_conv5(x))

        return x
