from collections import OrderedDict

import torchvision.models
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage
import torch
from PIL import Image
import os
import torch.optim as optim
import random
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
from models import Generator, residualBlock, swish, PolarNet, RefineNet, VGGFeatureBlock


class PSFmodel(nn.Module):
    def __init__(self, innd=3, outnd=32, num_layer=3):
        super(PSFmodel, self).__init__()
        # define submodule
        self.n_residual_blocks = 3
        self.relu = nn.ReLU(inplace=True)
        self.conv_in = nn.Conv2d(innd, outnd, (3, 3), dilation=1, padding=1)
        self.resnet = residualBlock(in_channels=outnd, n=outnd)
        #resconv2 = residualBlock(in_channels=outnd, n=outnd)

        self.conv2 = nn.Conv2d(outnd, outnd, 3, dilation=2)
        self.conv4 = nn.Conv2d(outnd, outnd, (3, 3), dilation=4)
        self.conv8 = nn.Conv2d(outnd, outnd, (3, 3), dilation=8)

        self.pad2 = nn.ReflectionPad2d(2)
        self.pad4 = nn.ReflectionPad2d(4)
        self.pad8 = nn.ReflectionPad2d(8)

        self.bn2 = nn.BatchNorm2d(outnd)
        self.conv_out = nn.Conv2d(outnd, innd, (3, 3), dilation=1, padding=1)

        # define physical network
        self.psfnet_main_ref = self.build_PSFnet(num_layer)
        self.psfnet_main_trans = self.build_PSFnet(num_layer)
        #self.psfnet_ref = self.psfnet_main_ref+nn.Identity()
        #self.psfnet_side_ref = nn.Sequential(nn.Identity)
        #self.psfnet_side_trans = nn.Sequential(nn.Identity)

    def forward(self, Ir_pred, It_pred):
        Ir_back = self.psfnet_main_ref(Ir_pred)
        It_back = self.psfnet_main_trans(It_pred)
        return [Ir_back, It_back]

    def build_PSFnet(self, num_layer=3):
        # generate order dict for sequential container
        od = OrderedDict([])
        odlist = [('conv_in', self.conv_in), ('relu', self.relu)]
        for n in range(num_layer):
            odlist.append(('resblk_'+str(n), self.resnet))

        odlist.append(('pad2_1', self.pad2))
        odlist.append(('conv2_1', self.conv2))
        odlist.append(('bn_1', self.bn2))
        odlist.append(('relu_1', self.relu))

        #odlist.append(('pad4_2', self.pad4))
        #odlist.append(('conv4_2', self.conv4))
        #odlist.append(('bn_2', self.bn2))
        #odlist.append(('relu_2', self.relu))

        #odlist.append(('pad8_3', self.pad8))
        #odlist.append(('conv8_3', self.conv8))
        #odlist.append(('bn_3', self.bn2))
        #odlist.append(('relu_3', self.relu))

        odlist.append(('conv_out', self.conv_out))
        psfnet_main = nn.Sequential(OrderedDict(odlist))
        return psfnet_main


class PolarFusion(nn.Module):
    def __init__(self):
        super(PolarFusion, self).__init__()
        self.mainnet = RefineNet()

    def forward(self, I_pred_phi1, I_pred_phi2, I_pred_phi3, I_pred_phi4, I_obs_phi1, I_obs_phi2, I_obs_phi3, I_obs_phi4):
        I_refine = self.mainnet(I_pred_phi1, I_pred_phi2, I_pred_phi3, I_pred_phi4, I_obs_phi1, I_obs_phi2, I_obs_phi3, I_obs_phi4)
        return I_refine

    def loss(self, I_refine, I_gt):
        return self.loss_l2(I_refine, I_gt)

    def loss_l2(self, I_refine, I_gt):
        return self.l2(I_refine, I_gt)


class Sepmodelv3(nn.Module):
    def __init__(self):
        super(Sepmodelv3, self).__init__()
        self.mainnet = PolarNet()
        self.refinenet = PolarFusion()
        self.l2 = nn.MSELoss()
        self.vggfeat = VGGFeatureBlock()

    def forward(self, I_obs1, I_obs2, I_obs3, I_obs4):
        [Ir_pred1, It_pred1] = self.mainnet(I_obs1)
        [Ir_pred2, It_pred2] = self.mainnet(I_obs2)
        [Ir_pred3, It_pred3] = self.mainnet(I_obs3)
        [Ir_pred4, It_pred4] = self.mainnet(I_obs4)
        #Ir_refine = self.refinenet(Ir_pred1, Ir_pred2, Ir_pred3, Ir_pred4, I_obs1, I_obs2, I_obs3, I_obs4)
        It_refine = self.refinenet(It_pred1, It_pred2, It_pred3, It_pred4, I_obs1, I_obs2, I_obs3, I_obs4)
        #return [Ir_refine, It_refine, Ir_pred1, It_pred1, Ir_pred2, It_pred2, Ir_pred3, It_pred3, Ir_pred4, It_pred4]
        return [It_refine, It_pred1, It_pred2, It_pred3, It_pred4]

    # def loss(self, Ir_pred, Ir_gt, It_pred, It_gt):
    #     # solve |Ir-Ir_p|_2+|It-It_p|_2
    #     lref = self.loss_ref(Ir_pred, Ir_gt)
    #     ltrans = self.loss_trans(It_pred, It_gt)
    #     ltotal = lref+ltrans
    #     total_loss = [lref, ltrans, ltotal]
    #     return total_loss

    def loss(self, It_refine, It_gt, It_pred=[], It_gt_polar=[]):
        loss_p = self.loss_trans_polar(It_pred, It_gt_polar)
        loss_r = self.loss_trans_refine(It_refine, It_gt)
        loss_vgg = self.loss_vgg(It_refine, It_gt)
        loss_t = loss_p+loss_r+0.0001*loss_vgg
        return [loss_p, loss_r, loss_vgg, loss_t]

    def loss_ref(self, Ir_pred, Ir_gt):
        return self.l2(Ir_pred, Ir_gt)

    def loss_trans_polar(self, It_pred=[], It_gt=[]):
        It_pred_stack = torch.stack(It_pred, dim=1)
        It_gt_stack = torch.stack(It_gt, dim=1)
        return self.l2(It_pred_stack, It_gt_stack)

    def loss_trans_refine(self, It_refine, It_gt):
        return self.l2(It_refine, It_gt)

    def loss_vgg(self, It_refine, It_gt):
        refine_vgg = self.vggfeat(It_refine)
        gt_vgg = self.vggfeat(It_gt)
        vggerror = torch.sum(torch.tensor([self.l2(refine_vgg[i], gt_vgg[i]) for i in range(len(refine_vgg))]))
        return vggerror

    def loss_trans(self, It_pred, It_gt):
        return self.l2(It_pred, It_gt)


class Sepmodelv2(nn.Module):
    def __init__(self):
        super(Sepmodelv2, self).__init__()
        self.mainnet = PolarNet()
        self.l2 = nn.MSELoss()

    def forward(self, I):
        return self.mainnet(I)

    def loss(self, I_pred, I_gt):
        [Ir_pred, It_pred] = I_pred
        [Ir_gt, It_gt] = I_gt
        #[I0, I45, I90, I135] = I_obs
        lref = self.loss_ref(Ir_pred, Ir_gt)
        ltrans = self.loss_trans(It_pred, It_gt)
        ltotal = lref+ltrans
        total_loss = [lref, ltrans, ltotal]
        return total_loss

    def loss_ref(self, Ir_pred, Ir_gt):
        return self.l2(Ir_pred, Ir_gt)

    def loss_trans(self, It_pred, It_gt):
        return self.l2(It_pred, It_gt)


class RefDiscriminator(nn.Module):
    def __init__(self):
        super(RefDiscriminator, self).__init__()
        self.net = Discriminator()

    def forward(self, I, I_pred):
        x = torch.cat([I, I_pred], dim=1)
        return self.net(x)

    def dloss(self, I_I_pred_out, I_I_gt_out, eps=1e-5):
        return torch.mean(torch.log(1-I_I_pred_out+eps) - torch.log(I_I_gt_out+eps))

    def gloss(self, I_I_pred_out):
        return torch.mean(-1*torch.log(I_I_pred_out))


class TransDiscriminator(nn.Module):
    def __init__(self):
        super(TransDiscriminator, self).__init__()
        self.net = Discriminator()

    def forward(self, I, I_pred):
        x = torch.cat([I, I_pred], dim=1)
        y = self.net(x)
        return y

    def dloss(self, I_I_pred_out, I_I_gt_out, eps=1e-5):
        #print(I_I_pred_out)
        tt = torch.mean(torch.log(1-I_I_pred_out+eps) - torch.log(I_I_gt_out+eps))
        #print(tt.item())
        return tt

    def gloss(self, I_I_pred_out, eps=1e-5):
        tt = torch.mean(-1*torch.log(I_I_pred_out+eps))
        #print(tt)
        return tt


class PolarEngine:
    def __init__(self, Ir, It, theta_in, d):
        self.num_bounce = 10
        self.weight_mirror = 0.3

        self.init_ref_para = 0.5
        self.init_ref_perp = 0.5
        self.init_trans_para = 0.5
        self.init_trans_perp = 0.5

        self.theta_in = theta_in
        self.n = 1.5
        self.d = d
        self.theta_out = self.calc_theta_out(self.theta_in)

        self.Ir = Ir
        self.It = It
        self.R_para = 0
        self.R_perp = 0
        self.T_para = 0
        self.T_perp = 0

    def calc_RT(self, theta_in, theta_out):
        R_para = np.power(np.tan(theta_in - theta_out), 2) / np.power(np.tan(theta_in + theta_out), 2)
        R_perp = np.power(np.sin(theta_in - theta_out), 2) / np.power(np.sin(theta_in + theta_out), 2)
        T_para = 1 - R_para
        T_perp = 1 - R_perp
        return [R_para, R_perp, T_para, T_perp]

    def calc_theta_out(self, theta_in):
        theta_out = np.arcsin(np.sin(theta_in)/self.n)
        return theta_out

    def shift(self, I):
        tx = np.int(2*self.d*np.cos(self.theta_in))

        [b, c, h, w] = I.shape
        filternp = np.zeros([3, 1, 41, 41])
        filternp[:, :, 20, 20 - tx] = 1
        filter = torch.Tensor(filternp).cuda()
        I_trans = torch.nn.functional.conv2d(I, filter, groups=3, padding=20)
        return I_trans

    def simulate_perp_para(self, I, weight_perp, weight_para):
        I_para = weight_para*I
        I_perp = weight_perp*I
        return [I_para, I_perp]

    def mirror_ref(self, I, weight_mirror):
        I_mirror = weight_mirror*I
        I_polar = (1-weight_mirror)*I
        return [I_mirror, I_polar]

    def ref_trans(self, I_para, I_perp, R_para, R_perp, T_para, T_perp):
        I_ref_para = I_para*R_para
        I_ref_perp = I_perp*R_perp
        I_trans_para = I_para*T_para
        I_trans_perp = I_perp*T_perp
        return [I_ref_para, I_ref_perp, I_trans_para, I_trans_perp]

    def translate_conv(self, I, d, theta=0):
        tx = d*torch.cos(theta)
        ty = d*torch.sin(theta)
        [b, c, h, w] = I.shape
        filternp = np.zeros([1, c, 40, 40])
        filternp[0, :, 20-ty, 20-tx] = 1
        filter = torch.Tensor(filternp)
        I_trans = torch.nn.functional.conv2d(I, filter, padding=20)

    def polarized_trace(self):
        [Ir_para, Ir_perp] = self.simulate_perp_para(self.Ir, self.init_ref_para, self.init_ref_perp)
        [It_para, It_perp] = self.simulate_perp_para(self.It, self.init_trans_para, self.init_trans_perp)

        # mirror reflection
        [Ir_para_mirror, Ir_para_polar] = self.mirror_ref(Ir_para, self.weight_mirror)
        [Ir_perp_mirror, Ir_perp_polar] = self.mirror_ref(Ir_perp, self.weight_mirror)

        Ir_para_out = Ir_para_polar
        Ir_perp_out = Ir_perp_polar

        I_para = Ir_para_polar
        I_perp = Ir_perp_polar

        [R_para, R_perp, T_para, T_perp] = self.calc_RT(self.theta_in, self.theta_out)
        # from air to glass1 (a->g1)
        [Ir_para, Ir_perp, It_para, It_perp] = self.ref_trans(I_para, I_perp, R_para, R_perp, T_para, T_perp)

        Ir_para_out = Ir_para_out + Ir_para
        Ir_perp_out = Ir_perp_out + Ir_perp

        I_para = It_para
        I_perp = It_perp
        for i in range(self.num_bounce):
            # g1 -> g2
            # from glass2 to air (g2->a)
            [R_para, R_perp, T_para, T_perp] = self.calc_RT(self.theta_out, self.theta_in)
            [Ir_para, Ir_perp, It_para, It_perp] = self.ref_trans(I_para, I_perp, R_para, R_perp, T_para, T_perp)
            # g2->g1
            # from g1 to a
            [R_para, R_perp, T_para, T_perp] = self.calc_RT(self.theta_out, self.theta_in)
            [Ir_para, Ir_perp, It_para, It_perp] = self.ref_trans(Ir_para, Ir_perp, R_para, R_perp, T_para, T_perp)

            ## add shift
            Ir_para_shift = self.shift(Ir_para)
            Ir_perp_shift = self.shift(Ir_perp)
            It_para_shift = self.shift(It_para)
            It_perp_shift = self.shift(It_perp)

            I_para = Ir_para_shift
            I_perp = Ir_perp_shift

            Ir_para_out = Ir_para_out + It_para_shift
            Ir_perp_out = Ir_perp_out + It_perp_shift
        return [Ir_para_out, Ir_perp_out]


class Sepmodel(nn.Module):
    def __init__(self, is_single=True):
        super(Sepmodel, self).__init__()
        innd = 3
        outnd = 64
        self.n_residual_blocks = 5
        self.is_single = is_single
        if True:
            self._init_layer(innd, outnd)
        if is_single:
            self.conv_input = nn.Conv2d(innd, outnd, (3, 3), dilation=1, padding=1)
        else:
            self.conv_input = nn.Conv2d(innd*4, outnd, (3, 3), dilation=1, padding=1)
        if True:
            for i in range(self.n_residual_blocks):
                self.add_module('residual_block' + str(i + 1), residualBlock())
            #self.conv2 = nn.Conv2d(outnd, outnd, 3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(outnd)
        self.conv_out = nn.Conv2d(outnd, 2 * 3, (3, 3), dilation=1, padding=1)

        self.psfnet = PSFmodel(innd, outnd)
        #self.psfnet_trans = PSFmodel(innd, outnd)

    def forward(self, imgin=[]):
        # input0, input45, input90, input135
        imginput = torch.cat(tuple(imgin), dim=1)
        #xin = self.conv_input()
        #xin = self.relu(xin)
        #Itot = torch.mean(input4, dim=1)
        #x = swish(self.conv_input(input4))

        # forward layer: I_obs->Ir_pred, It_pred
        x = self.relu(self.conv_input(imginput))
        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i+1))(y)
        x = self.bn2(self.conv2(self.pad2(x))) + x

        x = self.relu(self.bn2(self.conv2(self.pad2(x))))
        x = self.relu(self.bn2(self.conv4(self.pad4(x))))
        x = self.relu(self.bn2(self.conv8(self.pad8(x))))
        x = self.relu(self.bn2(self.conv1(self.pad1(x))))

        x = self.conv_out(x)
        [Ir, It] = torch.split(x, (3, 3), dim=1)

        # backward layer: Ir, It->I_obs_pred
        #self.psfnet = self.build_PSFnet(3, 3)
        [Ir_obs, It_obs] = self.psfnet(Ir, It)
        #It_obs = self.psfnet_trans(It)

        return [Ir, It, Ir_obs+It_obs]

    def _init_layer(self, innd=64, outnd=64):
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

        # define vgg net features
        vgg19 = torchvision.models.vgg19(pretrained=True)
        vgg19f = vgg19.features
        for param in vgg19.parameters():
            param.requires_grad = False
        self.vgg19f1 = vgg19f[:5]
        self.vgg19f2 = vgg19f[5:10]
        self.vgg19f3 = vgg19f[10:19]
        self.vgg19f4 = vgg19f[19:28]
        self.vgg19f5 = vgg19f[28:]

        # define reflection padding layer here
        self.pad1 = nn.ReflectionPad2d(1)
        self.pad2 = nn.ReflectionPad2d(2)
        self.pad4 = nn.ReflectionPad2d(4)
        self.pad8 = nn.ReflectionPad2d(8)
        self.pad16 = nn.ReflectionPad2d(16)
        self.pad32 = nn.ReflectionPad2d(32)
        self.pad64 = nn.ReflectionPad2d(64)

        # define dilation conv layer here
        self.conv1 = nn.Conv2d(outnd, outnd, (3, 3))
        self.conv2 = nn.Conv2d(outnd, outnd, (3, 3), dilation=2)
        self.conv4 = nn.Conv2d(outnd, outnd, (3, 3), dilation=4)
        self.conv8 = nn.Conv2d(outnd, outnd, (3, 3), dilation=8)
        self.conv16 = nn.Conv2d(outnd, outnd, (3, 3), dilation=16)
        self.conv32 = nn.Conv2d(outnd, outnd, (3, 3), dilation=32)
        self.conv64 = nn.Conv2d(outnd, outnd, (3, 3), dilation=64)

        # define resdual net block here
        self.resblk_1_1 = ResBlock(in_channel=64, out_channel=64, dilation=1, padding=1)
        self.resblk_2_2 = ResBlock(in_channel=64, out_channel=64, dilation=2, padding=2)
        self.resblk_4_4 = ResBlock(in_channel=64, out_channel=64, dilation=4, padding=4)
        self.resblk_8_8 = ResBlock(in_channel=64, out_channel=64, dilation=8, padding=8)
        self.resblk_16_16 = ResBlock(in_channel=64, out_channel=64, dilation=16, padding=16)

        # define conv input and output here
        self.relu = torch.nn.ReLU()

        # input layer single image 3 channel
        self.conv_input3 = nn.Conv2d(3, outnd, (3, 3), dilation=1, padding=1)

        # input layer 4 single image 3 channel
        self.conv_input3_4 = nn.Conv2d(3 * 4, outnd, (3, 3), dilation=1, padding=1)

        # 2*3 output: (Ir It) or (Iun, Ipol)
        self.conv_out2_3 = nn.Conv2d(outnd, 2 * 3, (3, 3), dilation=1, padding=1)

        # 3*3 output: (D, S, P)
        self.conv_out3_3 = nn.Conv2d(outnd, 2 * 3, (3, 3), dilation=1, padding=1)

    def build_sepnet(self):
        pass

    def build_resblk(self, innd=64, outnd=64, dilation=1, padding=1, stride=1):
        convdp = nn.Conv2d(innd, outnd, (3, 3), dilation=dilation, padding=padding, stride=stride, bias=False)
        left = nn.Sequential(self.convdp, nn.BatchNorm2d(outnd), nn.ReLU(True), self.convdp, nn.BatchNorm2d(outnd))
        right = nn.Identity()
        return left+right

    def build_resnet(self, innd=64, outnd=64):
        resblk_1_1 = ResBlock(in_channel=innd, out_channel=outnd, dilation=1, padding=1)
        resblk_2_2 = ResBlock(in_channel=outnd, out_channel=outnd, dilation=2, padding=2)
        resblk_4_4 = ResBlock(in_channel=outnd, out_channel=outnd, dilation=4, padding=4)
        resblk_8_8 = ResBlock(in_channel=outnd, out_channel=outnd, dilation=8, padding=8)
        resblk_16_16 = ResBlock(in_channel=outnd, out_channel=outnd, dilation=16, padding=16)
        resnet = nn.Sequential(resblk_1_1, resblk_2_2, resblk_4_4, resblk_8_8, resblk_16_16, resblk_1_1)
        return resnet

    def build_convref_net(self, innd=64, outnd=64):
        relu = torch.nn.ReLU()
        batchnorm2d = nn.BatchNorm2d(outnd)
        # define reflection padding layer here
        pad1 = nn.ReflectionPad2d(1)
        pad2 = nn.ReflectionPad2d(2)
        pad4 = nn.ReflectionPad2d(4)
        pad8 = nn.ReflectionPad2d(8)
        pad16 = nn.ReflectionPad2d(16)
        pad32 = nn.ReflectionPad2d(32)
        pad64 = nn.ReflectionPad2d(64)

        # define dilation conv layer here
        conv1 = nn.Conv2d(innd, outnd, (3, 3), dilation=1)
        conv2 = nn.Conv2d(outnd, outnd, (3, 3), dilation=2)
        conv4 = nn.Conv2d(outnd, outnd, (3, 3), dilation=4)
        conv8 = nn.Conv2d(outnd, outnd, (3, 3), dilation=8)
        conv16 = nn.Conv2d(outnd, outnd, (3, 3), dilation=16)
        conv32 = nn.Conv2d(outnd, outnd, (3, 3), dilation=32)
        conv64 = nn.Conv2d(outnd, outnd, (3, 3), dilation=64)

        # define convref layer here
        convref1 = nn.Sequential(pad1, conv1, batchnorm2d, relu)
        convref2 = nn.Sequential(pad2, conv2, batchnorm2d, relu)
        convref4 = nn.Sequential(pad4, conv4, batchnorm2d, relu)
        convref8 = nn.Sequential(pad8, conv8, batchnorm2d, relu)
        convref16 = nn.Sequential(pad16, conv16, batchnorm2d, relu)
        convref32 = nn.Sequential(pad32, conv32, batchnorm2d, relu)
        convref64 = nn.Sequential(pad64, conv64, batchnorm2d, relu)
        convref_net = nn.Sequential(convref1, convref2, convref4, convref8, convref16, convref32, convref64)
        return convref_net

    def calc_vggfeat(self, inputs):
        h = inputs.shape[2]
        w = inputs.shape[3]
        out1 = self.vgg19f1(inputs)
        out2 = self.vgg19f2(out1)
        out3 = self.vgg19f3(out2)
        out4 = self.vgg19f4(out3)
        out5 = self.vgg19f5(out4)
        return [out1, out2, out3, out4, out5]

    def loss(self, IrIt=[], IrIr_pred=[], I_obs=[], I_obs_pred=[], weights=[1, 1, 1, 1]):
        '''
        Lrt = L1(gt, pred)+Lvgg(gt, pred)+Le(grad)+Ld
        Ldsp = L1(dxD+dyD)+Lrec(I=DS+P)+L2(S)+L1(P)
        Lphy = Lrec(I=Ir+It)+Lrec(I0=DrSr+Pr0+DtSt+Pt0)+Lrec(I45=DrSr+Pr45+DtSt+Pt45)...
        Lback = Ir_pred-Ir_obs
        '''

        #I0, I45, I90, I135 = I_pol
        Ir_real, It_real = IrIt
        Ir_pred, It_pred = IrIr_pred
        I_obs_1 = I_obs
        I_obs_pred_1 = I_obs_pred

        #loss_sep = weights[0] * self.loss_sep(Ir_real, It_real, Ir_pred, It_pred)
        #loss_sep = weights[0] * torch.mean(torch.abs(Ir_real-Ir_pred))+torch.mean(torch.abs(It_real-It_pred))
        loss_sep = weights[0] * self.loss_sep(Ir_real, It_real, Ir_pred, It_pred)
        loss_vgg_sep = weights[1] * self.loss_vgg(Ir_real, It_real, Ir_pred, It_pred)
        loss_exclu = weights[2] * self.loss_exclusion(Ir_pred, It_pred)
        loss_backward = weights[3] * self.loss_backward(I_obs_1, I_obs_pred_1)
        #loss_forward = self.loss_forward(Ir_real, Ir_pred) + self.loss_forward(It_real, It_pred)

        #allloss = [loss_sep, loss_vgg_sep, loss_exclu]
        allloss = [loss_sep, loss_vgg_sep, loss_exclu, loss_backward]
        #allloss = [loss_sep]
        return allloss

    def loss_forward(self, I_gt, I_pred):
        loss_val = torch.mean(torch.abs(I_gt-I_pred))
        return loss_val

    def loss_backward(self, I_obs, I_obs_pred):
        loss_val = torch.mean(torch.abs(I_obs - I_obs_pred))
        return loss_val

    def loss_sep(self, Ir, It, Ir_pred, It_pred):
        loss_r = torch.mean(torch.abs(Ir_pred-Ir))
        loss_t = torch.mean(torch.abs(It_pred-It))
        loss_forward_rec_l1 = loss_r + loss_t
        loss_backward_rec_l1 = 0
        return loss_forward_rec_l1

    def loss_vgg(self, Ir, It, Ir_pred, It_pred):
        #l2 = nn.MSELoss()
        #l1 = nn.L1Loss()
        vgg_r = self.calc_vggfeat(Ir)
        vgg_t = self.calc_vggfeat(It)
        vgg_pred_r = self.calc_vggfeat(Ir_pred)
        vgg_pred_t = self.calc_vggfeat(It_pred)
        vgg_real_r = self.calc_vggfeat(Ir)
        vgg_real_t = self.calc_vggfeat(It)
        vggloss_r = torch.mean(torch.tensor([self.l2(vgg_pred_r[id], vgg_real_r[id]) for id in range(len(vgg_pred_r))]))
        vggloss_t = torch.mean(torch.tensor([self.l2(vgg_pred_t[id], vgg_real_t[id]) for id in range(len(vgg_pred_t))]))
        return vggloss_r+vggloss_t

    def loss_exclusion(self, Ir, It):
        return self.calc_exclusion_loss(Ir, It, level=1)

    def calc_l1_loss(self, inputs, outputs):
        return nn.L1Loss(inputs, outputs)

    def calc_perceptual_loss(self, inputs, outputs, weights=[]):
        input_feat = self.calc_vggfeat(inputs)
        output_feat = self.calc_vggfeat(outputs)
        loss = []
        if len(weights) == 0:
            weights = torch.ones(len(input_feat)).cuda()
        for i in range(len(input_feat)):
            loss.append(weights[i] * nn.MSELoss(input_feat[i], output_feat[i]))
        return torch.sum(torch.tensor(loss))

    def calc_exclusion_loss(self, inputs, outputs, level=1):
        kx = torch.tensor([[1., 0., -1.]], dtype=torch.float)
        ky = torch.tensor([[1.], [0.], [-1.]], dtype=torch.float)
        channels = inputs.shape[1]
        gkx = kx.repeat(channels, channels, 1, 1).float().cuda()
        gky = ky.repeat(channels, channels, 1, 1).float().cuda()
        loss = torch.tensor(0)
        avgp = nn.AvgPool2d((3, 3), padding=1)
        sigm = nn.Sigmoid()

        input_gradx = F.conv2d(inputs, gkx, padding=(0, 1))
        input_grady = F.conv2d(inputs, gky, padding=(1, 0))
        # print(input_gradx.shape, input_grady.shape)
        input_grad = input_gradx + input_grady

        output_gradx = F.conv2d(outputs, gkx, padding=(0, 1))
        output_grady = F.conv2d(outputs, gky, padding=(1, 0))

        alphax = 2.0 * torch.mean(input_gradx) / torch.mean(output_gradx)
        alphay = 2.0 * torch.mean(input_grady) / torch.mean(output_grady)

        input_gradx_s = (sigm(input_gradx) * 2) - 1
        input_grady_s = (sigm(input_grady) * 2) - 1
        output_gradx_s = (sigm(output_gradx * alphax) * 2) - 1
        output_grady_s = (sigm(output_grady * alphay) * 2) - 1
        gradx = input_gradx_s*input_gradx_s*output_gradx_s*output_gradx_s
        grady = input_grady_s*input_grady_s*output_grady_s*output_grady_s

        #output_grad = output_gradx + output_grady
        loss = torch.mean(gradx+grady)
        #print('%.6f, %.6f, %.6f'%(loss.item(), torch.mean(gradx).item(), torch.mean(grady).item()))
        return loss


class ResBlock(nn.Module):
    def __init__(self, in_channel=64, out_channel=64, dilation=1, padding=1, stride=1, shortcut=None):
        super(ResBlock, self).__init__()
        self.convdp = nn.Conv2d(in_channel, out_channel, (3, 3), dilation=dilation, padding=padding, stride=stride, bias=False)
        self.left = nn.Sequential(self.convdp, nn.BatchNorm2d(out_channel), nn.ReLU(True),
                                  self.convdp, nn.BatchNorm2d(out_channel))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class Intmodel(nn.Module):
    def __init__(self):
        super(Intmodel, self).__init__()
        innd = 3
        outnd = 64
        self.conv_input1 = nn.Conv2d(innd, outnd, (3, 3), dilation=1, padding=1)
        self.conv_input3 = nn.Conv2d(innd*4, outnd, (3, 3), dilation=1, padding=1)
        self.conv = nn.Conv2d(outnd, outnd, (3, 3), padding=1)
        self.conv1 = nn.Conv2d(outnd, outnd, (3, 3), dilation=1, padding=1)
        self.conv2 = nn.Conv2d(outnd, outnd, (3, 3), dilation=2, padding=2)
        self.conv3 = nn.Conv2d(outnd, outnd, (3, 3), dilation=3, padding=3)
        self.relu = torch.nn.ReLU(inplace=True)

        self.blocks = nn.Identity()
        self.resblk1 = ResBlock(outnd, outnd, dilation=1, padding=1)
        self.resblk2 = ResBlock(outnd, outnd, dilation=2, padding=2)
        self.resblk4 = ResBlock(outnd, outnd, dilation=4, padding=4)
        self.resblk8 = ResBlock(outnd, outnd, dilation=8, padding=8)
        self.resblk16 = ResBlock(outnd, outnd, dilation=16, padding=16)

        self.convs2 = nn.Conv2d(outnd, outnd, (3, 3), stride=2, padding=1)
        self.dconvs2 = nn.ConvTranspose2d(outnd, outnd, (3, 3), stride=2, padding=1)

        self.conv_out3 = nn.Conv2d(outnd, 3, (3, 3), padding=1)
        self.conv_out12 = nn.Conv2d(outnd, 12, (3, 3), padding=1)
        self.conv_out6 = nn.Conv2d(outnd, 6, (3, 3), padding=1)
        self.Dnet = nn.Sequential(self.conv_input1, self.conv, self.conv, self.conv,
                                    self.resblk2, self.resblk4, self.resblk8, self.resblk16, self.resblk1,
                                    self.conv, self.conv, self.conv, self.relu, self.conv_out3)
        self.DSnet = nn.Sequential(self.conv_input1, self.conv, self.conv, self.conv,
                                   self.resblk2, self.resblk4, self.resblk8, self.resblk16, self.resblk1,
                                   self.conv, self.conv, self.conv, self.relu, self.conv_out6)
        self.Snet = nn.Sequential(self.conv_input3, self.conv, self.conv, self.conv,
                                  self.resblk2, self.resblk4, self.resblk8, self.resblk16, self.resblk1,
                                  self.conv, self.conv, self.conv, self.relu, self.conv_out3)
        self.Pnet = nn.Sequential(self.conv_input3, self.conv, self.conv, self.conv,
                                  self.resblk2, self.resblk4, self.resblk8, self.resblk16, self.resblk1,
                                  self.conv, self.conv, self.conv, self.relu, self.conv_out12)

    def forward(self, I0, I1, I2, I3):
        inputDS = torch.cat((I0, I1, I2, I3), dim=1)
        Itot = 0.25*(I0+I1+I2+I3)
        x = self.blocks(Itot)
        resDS = self.DSnet(x)
        #print(D.shape)
        [resD, resS] = torch.split(resDS, (3, 3), dim=1)
        D = resD + Itot
        S = resS + Itot
        #inputS = torch.cat((I0-D*S, I1-D*S, I2-D*S, I3-D*S), dim=1)
        #x1 = self.blocks(inputS)
        #S = self.Snet(x1)
        #S = resS + x1
        inputP = torch.cat((I0-D*S, I1-D*S, I2-D*S, I3-D*S), dim=1)
        x3 = self.blocks(inputP)
        resP = self.Pnet(x3)
        P = resP + x3
        [P0, P1, P2, P3] = torch.split(P, (3, 3, 3, 3), dim=1)
        #[P0, P1, P2, P3] = I0-D*S, I1-D*S, I2-D*S, I3-D*S
        return [D, S, P0, P1, P2, P3]

    def resblock(self, inputs, in_channel=64, out_channel=64, dilation=1):
        x = self.blocks(inputs)
        dlayer = nn.Conv2d(in_channel, out_channel, (3, 3), dilation=dilation, padding=dilation)
        y1 = dlayer(x)
        y = self.relu(y1)
        return y+x

    def loss(self, inputs=[], outputs=[], weights=[1, 1, 1, 1, 1]):
        [I0, I1, I2, I3] = inputs
        Itot = 0.25*(I0+I1+I2+I3)
        [D, S, P0, P1, P2, P3] = outputs
        Lrec = self.loss_rec(I0, D, S, P0)\
               +self.loss_rec(I1, D, S, P1)\
               +self.loss_rec(I2, D, S, P2)\
               +self.loss_rec(I3, D, S, P3)
        Ld = self.loss_D(D) + self.loss_D_I(D, Itot)
        Ls = self.loss_S(S)
        #Ls = torch.tensor(0)
        Lsparse = self.loss_P_sparse(P0)+self.loss_P_sparse(P1)+self.loss_P_sparse(P2)+self.loss_P_sparse(P3)
        Ldyn = self.loss_P_dyn(P0, P1, P2, P3)
        #return Lrec+Ld+Ls+Lsparse+Ldyn
        return [Lrec, Ld, Ls, Lsparse, Ldyn]

    def loss_rec(self, I, D, S, P):
        l1 = nn.L1Loss()
        I_rec = D * S
        return l1(I_rec, I)

    def loss_D(self, D):
        # l1 = nn.L1Loss()
        kx = torch.tensor([1, 0, -1]).float()
        ky = torch.tensor([[1], [0], [-1]]).float()
        gkx = kx.repeat(3, 3, 1, 1).float().cuda()
        gky = ky.repeat(3, 3, 1, 1).float().cuda()
        gradx = torch.abs(F.conv2d(D, gkx, padding=(0, 1)))
        grady = torch.abs(F.conv2d(D, gky, padding=(1, 0)))
        grad = gradx + grady
        return torch.mean(torch.abs(grad))

    def loss_D_I(self, D, I):
        l1 = nn.L1Loss()
        d_I = l1(D, I)
        return d_I

    def loss_S(self, S):
        l2 = nn.MSELoss()
        kx = torch.tensor([1, 0, -1]).float()
        ky = torch.tensor([[1], [0], [-1]]).float()
        gkx = kx.repeat(3, 3, 1, 1).float().cuda()
        gky = ky.repeat(3, 3, 1, 1).float().cuda()
        gradx = torch.abs(F.conv2d(S, gkx, padding=(0, 1)))
        grady = torch.abs(F.conv2d(S, gky, padding=(1, 0)))
        [S0, S1, S2] = torch.split(S, (1, 1, 1), dim=1)
        graddim = torch.abs(S0-S1)+torch.abs(S1-S2)+torch.abs(S0-S2)
        grad = torch.sum(gradx, dim=1) + torch.sum(grady, dim=1)
        gradd = torch.sum(graddim, dim=1)
        return torch.mean(grad*grad)+torch.mean(gradd*gradd)

    def loss_P_sparse(self, P):
        return torch.mean(torch.abs(P))

    def loss_P_dyn(self, P0, P1, P2, P3):
        return torch.mean(torch.abs(P0*P1*P2*P3))


class Discriminator(nn.Module):
    def __init__(self, nc=6, ndf=64, ksz=3, stride=1, padding=1):
        super(Discriminator, self).__init__()
        tt = [
            nn.Conv2d(nc, ndf, kernel_size=ksz, stride=stride, padding=padding),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=ksz, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=ksz, stride=stride, padding=padding, bias=False),
            nn.Sigmoid()]
        self.net = nn.Sequential(*tt)

        # self.main = nn.Sequential(
        #     # input is (nc) x 64 x 64
        #     nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf) x 32 x 32
        #     nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*2) x 16 x 16
        #     nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*4) x 8 x 8
        #     nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*8) x 4 x 4
        #     nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        return self.net(x)
        #outputs = self.main(inputs)
        #return outputs.view(-1, 1).squeeze(1)

    #def dloss(real_output, pred_output):
    #    return -0.5 * torch.mean(torch.log(real_output + 1e-10) + torch.log(1 - pred_output + 1e-10))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == '__main__':
    pass