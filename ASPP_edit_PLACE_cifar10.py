import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP_places(nn.Module):
    def __init__(self):
        super(ASPP_places, self).__init__()
     
        self.conv_1x1_1 = nn.Conv2d(96, 128, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(128)
        
        self.conv_3x3_dil6 = nn.Sequential(
            nn.Conv2d(96,128, kernel_size=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(128)
        )
        self.conv_3x3_dil12 = nn.Sequential(
            nn.Conv2d(96,128, kernel_size=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(128)
        )
        self.conv_3x3_dil18 = nn.Sequential(
            nn.Conv2d(96,128, kernel_size=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=6, dilation=6),
            nn.BatchNorm2d(128)
        )

        
#         self.avg_pool = nn.AdaptiveAvgPool2d(7)

#         self.conv_1x1_2 = nn.Conv2d(256, 128, kernel_size=1)
#         self.bn_conv_1x1_2 = nn.BatchNorm2d(128)

        self.conv_1x1_3 = nn.Conv2d(4*128, 256, kernel_size=1) # (512 = 4*128)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

        self.conv_1x1_4 = nn.Conv2d(256, 256, kernel_size=1)
#         self.residualconv = nn.Conv2d(256, 128, kernel_size=1)
    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2] # 28
        feature_map_w = feature_map.size()[3] # 28
#         residual = self.residualconv(feature_map)
        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: 128x28x28)
#         out_1x1 += residual
        out_3x3_1 = F.relu(self.conv_3x3_dil6(feature_map)) # (shape: 128x28x28)
#         out_3x3_1 += residual
        out_3x3_2 = F.relu(self.conv_3x3_dil12(feature_map)) # (shape: 128x28x28)
#         out_3x3_2 += residual
        out_3x3_3 = F.relu(self.conv_3x3_dil18(feature_map)) # (shape: 128x28x28)
#         out_3x3_3 += residual

#         out_img = self.avg_pool(feature_map) # (shape: (batch_size, 256, 7, 7))
#         print(out_img.shape)
#         out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 128, 7, 7))
#         out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, 28, 28))
#         out_img += residual
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3], 1) # (shape: (batch_size, 640, 28, 28))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, 28, 28))
        out = self.conv_1x1_4(out) # (shape: (batch_size, 256, 28, 28))
#         print(out.shape)
        return out
