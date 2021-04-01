# -*- coding: utf-8 -*-
"""Implements SRGAN models: https://arxiv.org/abs/1609.04802

TODO:

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models


def swish(x):
    return x * torch.sigmoid(x)


class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)


class VGGFeatureBlock(nn.Module):
    def __init__(self):
        super(VGGFeatureBlock, self).__init__()
        vgg19 = torchvision.models.vgg19(pretrained=True)
        #vgg19.cuda()
        vgg19f = vgg19.features
        for param in vgg19.parameters():
            param.requires_grad = False
        self.vgg19f1 = vgg19f[:5]
        self.vgg19f2 = vgg19f[5:10]
        self.vgg19f3 = vgg19f[10:19]
        self.vgg19f4 = vgg19f[19:28]
        self.vgg19f5 = vgg19f[28:]
        self.dlayer = nn.AvgPool2d(kernel_size=4, stride=2, padding=1)

    def forward(self, imgin):
        fig1 = self.dlayer(imgin)
        fig2 = self.dlayer(fig1)
        fig3 = self.dlayer(fig2)
        fig4 = self.dlayer(fig3)
        fig5 = self.dlayer(fig4)

        out1 = self.vgg19f1(imgin)
        out2 = self.vgg19f2(out1)
        out3 = self.vgg19f3(out2)
        out4 = self.vgg19f4(out3)
        out5 = self.vgg19f5(out4)

        out1 = torch.cat([out1, fig1], dim=1)
        out2 = torch.cat([out2, fig2], dim=1)
        out3 = torch.cat([out3, fig3], dim=1)
        out4 = torch.cat([out4, fig4], dim=1)
        out5 = torch.cat([out5, fig5], dim=1)
        return [out1, out2, out3, out4, out5]


def test_VGGFeatureBlock():
    from PIL import Image
    import os
    from torchvision.transforms import ToTensor, ToPILImage
    imgpath = '/home/lir0b/data/polar/Simulation_HjghIntensity/101_1010_10'
    imgname = 'S0_theta_10_fi_0.jpg'
    img = Image.open(os.path.join(imgpath, imgname))
    print(img.size)
    imgcuda = ToTensor()(img).unsqueeze(0).cuda()
    imgin = ToTensor()(img).unsqueeze(0)
    vgg = VGGFeatureBlock()
    print(vgg)
    vggout = vgg(imgin)
    for v in vggout:
        print(v.shape)


class PolarBlock(nn.Module):
    def __init__(self, innd=3, outnd=64, ksz=3):
        super(PolarBlock, self).__init__()
        self.net1 = ConvUnit(innd, outnd, ksz)
        self.net2 = ConvUnit(outnd, outnd, ksz)
        self.net3 = ConvUnit(innd, outnd, ksz)

    def forward(self, x1, x2):
        y1 = self.net1(x1)
        y2 = self.net3(x2)
        y1 = self.net2(y1+y2)
        return [y1, y2]


class ResBlock(nn.Module):
    def __init__(self, innd=64, outnd=64, ksz=3, dilation=1, padding=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(innd, outnd, (ksz, ksz), dilation=dilation)
        self.bn = nn.BatchNorm2d(outnd)
        self.conv2 = nn.Conv2d(outnd, outnd, (ksz, ksz), dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.pad = nn.ReflectionPad2d(padding)

    def forward(self, x):
        y = self.relu(self.bn(self.conv1(self.pad(x))))
        return self.relu(self.bn(self.conv2(self.pad(y))))+x


class ConvUnit(nn.Module):
    def __init__(self, innd=3, outnd=64, ksz=3, stride=1, dilation=1, padding=1, maxpool=False, batchsz=1):
        super(ConvUnit, self).__init__()
        self.bn = nn.BatchNorm2d(outnd)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(innd, outnd, (ksz, ksz), stride=stride, dilation=dilation)
        self.pad = nn.ReflectionPad2d(padding)
        #self.maxpool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        #y = self.maxpool(self.relu(self.bn(self.conv(self.pad(x)))))
        y = self.relu(self.bn(self.conv(self.pad(x))))
        return y


class ConvUnit2(nn.Module):
    def __init__(self, innd=3, outnd=64, ksz=3, stride=1, dilation=1, padding=1, maxpool=False):
        super(ConvUnit2, self).__init__()
        self.bn = nn.BatchNorm2d(outnd)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(innd, outnd, (ksz, ksz), stride=stride, dilation=dilation)
        self.conv2 = nn.Conv2d(outnd, outnd, (ksz, ksz), stride=stride, dilation=dilation)
        self.pad = nn.ReflectionPad2d(padding)

    def forward(self, x):
        y = self.relu(self.bn(self.conv1(self.pad(x))))
        y2 = self.relu(self.bn(self.conv2(self.pad(y))))
        return y2


class ConvUnit4(nn.Module):
    def __init__(self, innd=3, outnd=64, ksz=3, stride=1, dilation=1, padding=1, maxpool=False):
        super(ConvUnit4, self).__init__()
        self.bn = nn.BatchNorm2d(outnd)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(innd, outnd, (ksz, ksz), stride=stride, dilation=dilation)
        self.conv2 = nn.Conv2d(outnd, outnd, (ksz, ksz), stride=stride, dilation=dilation)
        self.conv3 = nn.Conv2d(outnd, outnd, (ksz, ksz), stride=stride, dilation=dilation)
        self.conv4 = nn.Conv2d(outnd, outnd, (ksz, ksz), stride=stride, dilation=dilation)
        self.pad = nn.ReflectionPad2d(padding)

    def forward(self, x):
        y1 = self.relu(self.bn(self.conv1(self.pad(x))))
        y2 = self.relu(self.bn(self.conv2(self.pad(y1))))
        y3 = self.relu(self.bn(self.conv3(self.pad(y2))))
        y4 = self.relu(self.bn(self.conv4(self.pad(y3))))
        return y4


class ConvTUnit(nn.Module):
    def __init__(self, innd=3, outnd=64, ksz=3, stride=1, dilation=1, padding=1):
        super(ConvTUnit, self).__init__()
        self.bn = nn.BatchNorm2d(outnd)
        self.relu = nn.ReLU(inplace=True)
        self.convt = nn.ConvTranspose2d(innd, outnd, (ksz, ksz), stride=stride, dilation=dilation, padding=padding)
        #self.pad = nn.ReflectionPad2d(padding)

    def forward(self, x):
        y = self.relu(self.bn(self.convt(x)))
        return y


class VGGUpsamplingBlock(nn.Module):
    def __init__(self):
        super(VGGUpsamplingBlock, self).__init__()
        #self.convt5 = ConvTUnit(innd=512, outnd=512, ksz=3, stride=1, dilation=1, padding=1)
        #self.convt4 = ConvTUnit(innd=512 * 2, outnd=512, ksz=4, stride=2, dilation=1, padding=1)
        #self.convt3 = ConvTUnit(innd=512 * 2, outnd=256, ksz=4, stride=2, dilation=1, padding=1)
        #self.convt2 = ConvTUnit(innd=256 * 2, outnd=128, ksz=4, stride=2, dilation=1, padding=1)
        #self.convt1 = ConvTUnit(innd=128 * 2, outnd=64, ksz=4, stride=2, dilation=1, padding=1)
        #self.convt0 = ConvTUnit(innd=64 * 2, outnd=3, ksz=4, stride=2, dilation=1, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv5 = ConvUnit4(innd=512+3, outnd=512, ksz=3, stride=1, dilation=1, padding=1)
        self.conv4 = ConvUnit4(innd=512+512+3, outnd=512, ksz=3, stride=1, dilation=1, padding=1)
        self.conv3 = ConvUnit4(innd=512+512+3, outnd=256, ksz=3, stride=1, dilation=1, padding=1)
        self.conv2 = ConvUnit4(innd=256+256+3, outnd=128, ksz=3, stride=1, dilation=1, padding=1)
        self.conv1 = ConvUnit2(innd=128+128+3, outnd=64, ksz=3, stride=1, dilation=1, padding=1)
        self.conv0 = ConvUnit2(innd=64+64+3, outnd=3, ksz=3, stride=1, dilation=1, padding=1)

    def forward(self, vggfeat):
        [data1, data2, data3, data4, data5] = vggfeat
        data_out5 = self.conv5(data5)

        data_up5 = self.upsample(torch.cat([data_out5, data5], dim=1))
        data_out4 = self.conv4(data_up5)

        data_up4 = self.upsample(torch.cat([data_out4, data4], dim=1))
        data_out3 = self.conv3(data_up4)

        data_up3 = self.upsample(torch.cat([data_out3, data3], dim=1))
        data_out2 = self.conv2(data_up3)

        data_up2 = self.upsample(torch.cat([data_out2, data2], dim=1))
        data_out1 = self.conv1(data_up2)

        data_up1 = self.upsample(torch.cat([data_out1, data1], dim=1))
        data_out0 = self.conv0(data_up1)
        return data_out0


class RefineNet(nn.Module):
    def __init__(self, input_nc=24, output_nc=3, ngf=64, ksz=3, stride=1, dilation=1):
        super(RefineNet, self).__init__()
        self.net_in = ConvUnit(innd=input_nc, outnd=ngf, ksz=3, stride=1, dilation=1, padding=1)
        self.conv1 = ConvUnit(innd=ngf, outnd=ngf, ksz=3, stride=1, dilation=1, padding=1)
        self.conv2 = ConvUnit(innd=ngf, outnd=int(ngf/2), ksz=3, stride=1, dilation=1, padding=1)
        self.conv3 = ConvUnit(innd=int(ngf/2), outnd=int(ngf/2), ksz=3, stride=1, dilation=1, padding=1)
        self.net_out = ConvUnit(innd=int(ngf/2), outnd=output_nc, ksz=3, stride=1, dilation=1, padding=1)

    def forward(self, I_pred=[], I_obs=[]):
        #I_pred_phi1, I_pred_phi2, I_pred_phi3, I_pred_phi4, I_obs_phi1, I_obs_phi2, I_obs_phi3, I_obs_phi4
        I_list = I_pred + I_obs
        #img_list = [I_pred_phi1, I_pred_phi2, I_pred_phi3, I_pred_phi4, I_obs_phi1, I_obs_phi2, I_obs_phi3, I_obs_phi4]
        img_input = torch.cat(I_list, dim=1)
        fin = self.net_in(img_input)
        f1 = self.conv1(fin)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        fout = self.net_out(f3)
        return fout


class PolarFusion(nn.Module):
    def __init__(self, input_nc=24, output_nc=3, ngf=64):
        super(PolarFusion, self).__init__()
        self.network = RefineNet(input_nc=input_nc, output_nc=output_nc, ngf=ngf)

    def forward(self, I_pred_phi1, I_pred_phi2, I_pred_phi3, I_pred_phi4,
                I_obs_phi1, I_obs_phi2, I_obs_phi3, I_obs_phi4):
        I_pred = [I_pred_phi1, I_pred_phi2, I_pred_phi3, I_pred_phi4]
        I_obs = [I_obs_phi1, I_obs_phi2, I_obs_phi3, I_obs_phi4]
        I_refine = self.network(I_pred, I_obs)
        return I_refine

    def loss(self, I_refine, I_gt):
        return self.loss_l2(I_refine, I_gt)

    def loss_l2(self, I_refine, I_gt):
        return self.l2(I_refine, I_gt)


class PolarNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, ksz=3, dilation=1, padding=1):
        super(PolarNet, self).__init__()
        innd = input_nc
        featnd = ngf
        outnd = output_nc
        self.net_in = nn.Conv2d(innd, featnd, (ksz, ksz), dilation=dilation, padding=padding)
        self.vggfeat = VGGFeatureBlock()
        self.transnet = VGGUpsamplingBlock()

        self.conv1 = nn.Conv2d(featnd, featnd, (ksz, ksz), dilation=dilation, padding=padding)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(featnd)

    def forward(self, I):
        #vggfeat = VGGFeatureBlock()
        vggout = self.vggfeat(I)
        #Ir_pred = self.refnet(vggout)
        It_pred = self.transnet(vggout)
        Ir_pred = I - It_pred
        return [Ir_pred, It_pred]


class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x


class upsampleBlock(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels, out_channels):
        super(upsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        return swish(self.shuffler(self.conv(x)))


class Generator(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(3, 64, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i+1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        for i in range(int(self.upsample_factor/2)):
            self.add_module('upsample' + str(i+1), upsampleBlock(64, 256))

        self.conv3 = nn.Conv2d(64, 3, 9, stride=1, padding=4)

    def forward(self, x):
        x = swish(self.conv1(x))

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i+1))(y)

        x = self.bn2(self.conv2(y)) + x

        for i in range(self.upsample_factor/2):
            x = self.__getattr__('upsample' + str(i+1))(x)

        return self.conv3(x)


# class Discriminator(nn.Module):
#     def __init__(self, input_nc=6, ndf=64):
#         super(Discriminator, self).__init__()
#         self.net = [
#             nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=False)]
#         self.net = nn.Sequential(*self.net)
#
#     def forward(self, x):
#         return self.net(x)


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
#
#         self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
#         self.bn4 = nn.BatchNorm2d(128)
#         self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
#         self.bn5 = nn.BatchNorm2d(256)
#         self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
#         self.bn6 = nn.BatchNorm2d(256)
#         self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
#         self.bn7 = nn.BatchNorm2d(512)
#         self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
#         self.bn8 = nn.BatchNorm2d(512)
#
#         # Replaced original paper FC layers with FCN
#         self.conv9 = nn.Conv2d(512, 1, 1, stride=1, padding=1)
#
#     def forward(self, x):
#         x = swish(self.conv1(x))
#
#         x = swish(self.bn2(self.conv2(x)))
#         x = swish(self.bn3(self.conv3(x)))
#         x = swish(self.bn4(self.conv4(x)))
#         x = swish(self.bn5(self.conv5(x)))
#         x = swish(self.bn6(self.conv6(x)))
#         x = swish(self.bn7(self.conv7(x)))
#         x = swish(self.bn8(self.conv8(x)))
#
#         x = self.conv9(x)
#         return torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


if __name__ == '__main__':
    if False:
        test_VGGFeatureBlock()

    if False:
        net = Discriminator()
        print(net)
        data = torch.randn(1, 3, 512, 512)
        y = net(data)
        print(y.shape)

    if False:
        data = torch.randn(1, 3, 1024, 1024)
        vggfeat = VGGFeatureBlock()
        vggupfeat = VGGUpsamplingBlock()
        t = vggupfeat(vggfeat(data))
        print(t.shape)


        #net = Generator(5, 4)
        #print(net)
        #test_VGGFeatureBlock()
        #m = nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1, output_padding=1, groups=2)
        #input = torch.randn(20, 16, 50, 100)
        #output = m(input)
        #print(output.shape)
        #conv = ConvUnit(innd=3, outnd=64, ksz=5, stride=1, dilation=1, padding=1)
        #convt = ConvTUnit(innd=3, outnd=64, ksz=3, stride=1, dilation=1, padding=1)
        #conv1 = ConvUnit(innd=3, outnd=64, ksz=3, stride=1, dilation=1, padding=1)
        #conv2 = ConvUnit(innd=64, outnd=64, ksz=5, stride=1, dilation=1, padding=2)
        #conv3 = ConvUnit(innd=64, outnd=64, ksz=7, stride=1, dilation=1, padding=3)
        #maxpool = nn.MaxPool2d((2, 2))
        #data1 = conv1(data)
        #print(data1.shape)
        #data1_1 = maxpool(data1)
        #print(data1_1.shape)
        #data2 = conv2(data1_1)
        #print(data2.shape)
        #data2_1 = maxpool(data2)
        #print(data2_1.shape)
    if True:
        data = torch.randn(1, 3, 512, 512)

        vgg19 = torchvision.models.vgg19(pretrained=True)
        # vgg19.cuda()
        vgg19f = vgg19.features
        for i in vgg19f:
            print(i)

        vggfeat = VGGFeatureBlock()

        vggout = vggfeat(data)
        [data1, data2, data3, data4, data5] = vggout
        model = VGGUpsamplingBlock()
        feat = model(vggout)
        print(feat.shape)

        for f in vggout:
            print(f.shape)



        #ttt = VGGUpsamplingBlock()
        #print(ttt())

        #print(data1.shape)
        #print(data2.shape)
        #print(data3.shape)
        #print(data4.shape)
        #print(data5.shape)

        #convt5 = ConvTUnit(innd=512, outnd=512, ksz=3, stride=1, dilation=1, padding=1)
        #convt4 = ConvTUnit(innd=512*2, outnd=512, ksz=4, stride=2, dilation=1, padding=1)
        #convt3 = ConvTUnit(innd=512*2, outnd=256, ksz=4, stride=2, dilation=1, padding=1)
        #convt2 = ConvTUnit(innd=256*2, outnd=128, ksz=4, stride=2, dilation=1, padding=1)
        #convt1 = ConvTUnit(innd=128*2, outnd=64, ksz=4, stride=2, dilation=1, padding=1)
        #convt0 = ConvTUnit(innd=64*2, outnd=3, ksz=4, stride=2, dilation=1, padding=1)




        #print('\n')
        #data_out5 = convt5(data5)
        #print(data_out5.shape)

        #data_out4 = convt4(torch.cat([data_out5, data5], dim=1))
        #print(data_out4.shape)

        #data_out3 = convt3(torch.cat([data_out4, data4], dim=1))
        #print(data_out3.shape)

        #data_out2 = convt2(torch.cat([data_out3, data3], dim=1))
        #print(data_out2.shape)

        #data_out1 = convt1(torch.cat([data_out2, data2], dim=1))
        #print(data_out1.shape)

        #data_out0 = convt0(torch.cat([data_out1, data1], dim=1))
        #print(data_out0.shape)



