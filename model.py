import torch
from torch import nn, optim
import torch.fft
from complexPyTorch.complexLayers import ComplexConvTranspose2d, ComplexConv2d
from complexPyTorch.complexFunctions import complex_relu
from utils.pytorch_prototyping.pytorch_prototyping import Unet
import math
from propagation_ASM import propagation_ASM
from MW_CANet import MW_CANet


class CDown(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.COV1 = nn.Sequential(ComplexConv2d(in_channels, out_channels, 3, stride=2, padding=1))

    def forward(self, x):
        out1 = complex_relu((self.COV1(x)))
        return out1


class CUp(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.COV1 = nn.Sequential(ComplexConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1))

    def forward(self, x):
        out1 = complex_relu((self.COV1(x)))
        return out1


class CUp2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.COV1 = nn.Sequential(ComplexConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1))

    def forward(self, x):
        out1 = self.COV1(x)
        return out1


class CCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.netdown1 = CDown(1, 4)
        self.netdown2 = CDown(4, 8)
        self.netdown3 = CDown(8, 16)

        self.netup3 = CUp(16, 8)
        self.netup2 = CUp(8, 4)
        self.netup1 = CUp2(4, 1)

    def forward(self, x):
        out1 = self.netdown1(x)
        out2 = self.netdown2(out1)
        out3 = self.netdown3(out2)

        out18 = self.netup3(out3)
        out19 = self.netup2(out18 + out2)
        out20 = self.netup1(out19 + out1)

        holophase = torch.atan2(out20.imag, out20.real)
        return holophase


class ccnncgh(nn.Module):
    def __init__(self):
        super().__init__()
        # self.ccnn1 = CCNN1()
        self.ccnn2 = CCNN2()
        self.MW_CANet = MW_CANet(in_channels=1, channels=10)

    def forward(self, amp, phase, z, pad, pitch, wavelength, H):
        target_complex = torch.complex(amp * torch.cos(phase), amp * torch.sin(phase))

        # predict_phase = self.ccnn1(target_complex)
        predict_phase = self.MW_CANet(amp)
        predict_complex = torch.complex(amp * torch.cos(predict_phase), amp * torch.sin(predict_phase))

        slmfield = propagation_ASM(u_in=predict_complex, z=z, linear_conv=pad, feature_size=[pitch, pitch],
                                   wavelength=wavelength,
                                   precomped_H=H)

        holophase = self.ccnn2(slmfield)

        return holophase, predict_phase



model = ccnncgh().to('cuda')
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameters: %.2f" % total)
