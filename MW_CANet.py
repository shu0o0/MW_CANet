import math
import dwt
import torch
import torch.nn as nn


# Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, channel, reduction=2):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        #print(x.shape)
        y = self.avg_pool(x).view(b, c)
        #print(y.shape)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# Dense Block (DB) with Conv and Deconv layers
class DenseBlock(nn.Module):
    def __init__(self, in_channels):
        super(DenseBlock, self).__init__()
        # Alternating between Conv and Deconv layers
        # Conv layer
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        # Deconv layer
        self.dconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        out1 = self.conv(x)
        # out1 = self.bn(out1)
        out1 = self.relu(out1)
        out2 = self.dconv(out1)
        # out2 = self.bn(out2)
        out2 = self.relu(out2)
        input3 = out1 + out2
        out3 = self.conv(input3)
        # out3 = self.bn(out3)
        out3 = self.relu(out3)
        input4 = out1 + out2 + out3
        out4 = self.dconv(input4)
        # out4 = self.bn(out4)
        out4 = self.relu(out4)
        out = out1 + out2 + out3 + out4
        return out


class Down(nn.Module):
    def __init__(self, in_channels):
        super(Down, self).__init__()

        # Dense Block and CAB at each level
        self.dense1 = DenseBlock(in_channels)  # Dense Block
        self.dense2 = DenseBlock(3 * in_channels)  # Dense Block
        self.cab2 = CAB(3 * in_channels)  # Channel Attention Block
        # self.dwt = pw.DWTForward(J=1, wave='haar')  # 1 levels of wavelet decomposition
        self.dwt = dwt.DWT()

    def forward(self, x):
        # print(x.shape)
        # DWT
        y = self.dwt(x)
        # print(y.shape)
        channels = x.shape[1]
        ll = y[:, :channels, :, :]  # LL subband
        lh = y[:, channels:2 * channels, :, :]  # LH subband
        hl = y[:, 2 * channels:3 * channels, :, :]  # HL subband
        hh = y[:, 3 * channels:, :, :]  # HH subband
        # Dense Block
        yl_cat = torch.cat([ll], dim=1)  # .unsqueeze(dim=0)
        # print(yl_cat.shape)
        yl_db = self.dense1(yl_cat)
        # Dense Block + CAB
        yh_cat = torch.cat([lh, hl, hh], dim=1)
        # print(yh_cat.shape)
        yh_cab = self.cab2(yh_cat)
        yh_db = self.dense2(yh_cab)
        return yl_db, yh_db


class Up(nn.Module):
    def __init__(self, in_channels):
        super(Up, self).__init__()
        # Dense Block and CAB at each level
        # self.dense = DenseBlock(in_channels)  # Dense Block
        # self.idwt = pw.DWTInverse(wave='haar')
        self.idwt = dwt.IWT()

    def forward(self, x, y):
        channels = x.shape[1]
        # split????
        # print(x.shape)
        lh = x[:, :channels // 3, :, :]  # LH subband
        hl = x[:, channels // 3:2 * channels // 3, :, :]  # HL subband
        hh = x[:, 2 * channels // 3:, :, :]  # HH subband
        freq_subbands = torch.cat([y, lh, hl, hh], dim=1)
        # IDWT
        out = self.idwt(freq_subbands)
        return out


# Overall Network Structure integrating CAB and DenseBlock
class MW_CANet(nn.Module):
    def __init__(self, in_channels=1, channels=10):
        super(MW_CANet, self).__init__()
        # Initial Conv layer
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=1, padding=0)
        # Dense Block and CAB at each level
        self.dense = DenseBlock(channels)  # Dense Block
        self.skip1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.skip2 = nn.Conv2d(3 * channels, 3 * channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(channels, in_channels, kernel_size=1, padding=0)  # Final Conv layer
        self.tanH = nn.Hardtanh(-math.pi, math.pi)
        self.down = Down(channels)
        self.up = Up(channels)
        # self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        down_out0 = self.conv_in(x)
        down_out1_1, down_out1_2 = self.down(down_out0)  # low, high
        down_out2_1, down_out2_2 = self.down(down_out1_1)
        down_out3_1, down_out3_2 = self.down(down_out2_1)
        up_out3 = self.up(down_out3_2, down_out3_1)
        up_out2 = self.up(down_out2_2, self.dense(down_out2_1 + up_out3))
        low_skip = self.skip1(down_out1_1)
        high_skip = self.skip2(down_out1_2)
        up_out1 = self.up(high_skip, self.dense(low_skip + up_out2))
        up_out0 = self.conv_out(down_out0 + up_out1)
        out = self.tanH(x + up_out0)
        return out


# Example usage
if __name__ == "__main__":
    model = MW_CANet(in_channels=1, channels=10).to('cuda')
    # input_tensor = torch.randn(1, 1, 1072, 1920).to('cuda')  # Example input
    input_tensor = torch.randn(1, 1, 400, 400).to('cuda')  # Example input
    output = model(input_tensor)
    # print(model)
    print(output.shape)  # Output should match input shape (1, 1, 400, 400)
# total = sum([param.nelement() for param in model.parameters()])
# print(int(total)+8282)
# print("Number of parameters: %.2f" % total)
