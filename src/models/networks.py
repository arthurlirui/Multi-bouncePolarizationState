import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from pprint import pprint
import torch.nn.functional as F
from torch.cuda import memory_cached
#tt = torch.cuda.memory_summary(device=None, abbreviated=False)

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', gpu_ids=[]):    
    norm_layer = get_norm_layer(norm_type=norm)     
    if netG == 'global':    
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)       
    elif netG == 'local':        
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    elif netG == 'DiffuseNet':
        netG = DiffuseGenerator(input_nc=input_nc, output_nc=output_nc, ngf=ngf,
                                n_downsampling=n_downsample_global, n_blocks=n_blocks_global,
                                norm_layer=norm_layer, padding_type='reflect')
    elif netG == 'DNet':
        netG = DNet(input_nc=input_nc, output_nc=output_nc, ngf=ngf,
                    n_downsampling=n_downsample_global, n_blocks=n_blocks_global, norm_layer=norm_layer,
                    padding_type='reflect')
    elif netG == 'SNet':
        netG = SNet(input_nc=input_nc, output_nc=output_nc, ngf=ngf,
                    n_downsampling=n_downsample_global, n_blocks=n_blocks_global, norm_layer=norm_layer,
                    padding_type='reflect')
    elif netG == 'HNet':
        netG = HNet(input_nc=input_nc, output_nc=output_nc, ngf=ngf,
                    n_downsampling=n_downsample_global, n_blocks=n_blocks_global, norm_layer=norm_layer,
                    padding_type='reflect')
    elif netG == 'HNet3x3':
        netG = HNet3x3(input_nc=input_nc, output_nc=output_nc, ngf=ngf,
                       n_downsampling=n_downsample_global, n_blocks=n_blocks_global, norm_layer=norm_layer,
                       padding_type='reflect')
    elif netG == 'ResNet3x1':
        netG = ResNet3x1(input_nc=input_nc, output_nc=output_nc, ngf=ngf,
                         n_downsampling=n_downsample_global, n_blocks=n_blocks_global, norm_layer=norm_layer,
                         padding_type='reflect')
    elif netG == 'ResNet1x1':
        netG = ResNet1x1(input_nc=input_nc, output_nc=output_nc, ngf=ngf,
                         n_downsampling=n_downsample_global, n_blocks=n_blocks_global, norm_layer=norm_layer,
                         padding_type='reflect')
    elif netG == 'ResNet1x2':
        netG = ResNet1x2(input_nc=input_nc, output_nc=output_nc, ngf=ngf,
                         n_downsampling=n_downsample_global, n_blocks=n_blocks_global, norm_layer=norm_layer,
                         padding_type='reflect')
    elif netG == 'ResNet1x3':
        netG = ResNet1x3(input_nc=input_nc, output_nc=output_nc, ngf=ngf,
                         n_downsampling=n_downsample_global, n_blocks=n_blocks_global, norm_layer=norm_layer,
                         padding_type='reflect')
    elif netG == 'PolarFusion':
        from models.sep_networks import PolarFusion
        netG = PolarFusion()
    elif netG == 'PolarNet':
        from models.sep_networks import PolarNet
        netG = PolarNet()
    else:
        raise('generator not implemented!')
    #print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class SpatialGradient(nn.Module):
    def __init__(self, minbatch=1, channels=3):
        super(SpatialGradient, self).__init__()
        self.L1Loss = nn.L1Loss()
        self.weights = [1, 1, 1]

        gradx3 = torch.zeros(minbatch, channels, 3, 3)
        gradx3[0, :, 1, 0] = 1
        gradx3[0, :, 1, 2] = -1
        self.gradx3 = gradx3.cuda()

        grady3 = torch.zeros(minbatch, channels, 3, 3)
        grady3[0, :, 0, 1] = 1
        grady3[0, :, 2, 1] = -1
        self.grady3 = grady3.cuda()

        self.gradk = [self.define_k1(minbatch, channels), self.define_k2(minbatch, channels),
                      self.define_k3(minbatch, channels), self.define_k4(minbatch, channels)]
        self.pad = torch.nn.ReflectionPad2d(1)

    def define_k1(self, minbatch=1, channels=3):
        gradk = torch.zeros(minbatch, channels, 3, 3)
        gradk[0, :, 0, 1] = 1
        gradk[0, :, 2, 1] = -1
        return gradk.cuda()

    def define_k2(self, minbatch=1, channels=3):
        gradk = torch.zeros(minbatch, channels, 3, 3)
        gradk[0, :, 1, 0] = 1
        gradk[0, :, 1, 2] = -1
        return gradk.cuda()

    def define_k3(self, minbatch=1, channels=3):
        gradk = torch.zeros(minbatch, channels, 3, 3)
        gradk[0, :, 0, 0] = 1
        gradk[0, :, 2, 2] = -1
        return gradk.cuda()

    def define_k4(self, minbatch=1, channels=3):
        gradk = torch.zeros(minbatch, channels, 3, 3)
        gradk[0, :, 0, 2] = 1
        gradk[0, :, 2, 0] = -1
        return gradk.cuda()

    def define_k5(self, minbatch=1, channels=1):
        gradk = torch.ones(minbatch, channels, 3, 3)
        gradk[0, :, 1, 1] = -8
        return gradk.cuda()

    def forward(self, img):
        gradlist = []
        for k in self.gradk:
            gradlist.append(F.conv2d(self.pad(img), k))
        return gradlist


class DiffuseLoss(nn.Module):
    def __init__(self, minbatch=1, channels=3):
        super(DiffuseLoss, self).__init__()
        self.L1Loss = nn.L1Loss()
        self.weights = [1, 1, 1]

        gradx3 = torch.zeros(minbatch, channels, 3, 3)
        gradx3[0, :, 1, 0] = 1
        gradx3[0, :, 1, 2] = -1
        self.gradx3 = gradx3.cuda()

        grady3 = torch.zeros(minbatch, channels, 3, 3)
        grady3[0, :, 0, 1] = 1
        grady3[0, :, 2, 1] = -1
        self.grady3 = grady3.cuda()

        self.gradk = [self.define_k1(minbatch, channels), self.define_k2(minbatch, channels),
                      self.define_k3(minbatch, channels), self.define_k4(minbatch, channels)]

    def define_k1(self, minbatch=1, channels=3):
        gradk = torch.zeros(minbatch, channels, 3, 3)
        gradk[0, :, 0, 1] = 1
        gradk[0, :, 2, 1] = -1
        return gradk.cuda()

    def define_k2(self, minbatch=1, channels=3):
        gradk = torch.zeros(minbatch, channels, 3, 3)
        gradk[0, :, 1, 0] = 1
        gradk[0, :, 1, 2] = -1
        return gradk.cuda()

    def define_k3(self, minbatch=1, channels=3):
        gradk = torch.zeros(minbatch, channels, 3, 3)
        gradk[0, :, 0, 0] = 1
        gradk[0, :, 2, 2] = -1
        return gradk.cuda()

    def define_k4(self, minbatch=1, channels=3):
        gradk = torch.zeros(minbatch, channels, 3, 3)
        gradk[0, :, 0, 2] = 1
        gradk[0, :, 2, 0] = -1
        return gradk.cuda()

    def define_k5(self, minbatch=1, channels=1):
        gradk = torch.ones(minbatch, channels, 3, 3)
        gradk[0, :, 1, 1] = -8
        return gradk.cuda()

    def forward(self, S0, S1, S2):
        loss = 0
        for k in self.gradk:
            gradS0 = F.conv2d(S0, k)
            loss += torch.mean(torch.abs(gradS0))
        return loss


class ShadingMatrixLoss(nn.Module):
    def __init__(self):
        super(ShadingMatrixLoss, self).__init__()

    def forward(self, Sout=[], Sin=[], H=[]):
        [S0, S1, S2] = Sout
        [Si0, Si1, Si2] = Sin
        [H00, H01, H02, H10, H11, H12, H20, H21, H22] = H
        S0_est = H00 * Si0 + H01 * Si1 + H02 * Si2
        S1_est = H10 * Si0 + H11 * Si1 + H12 * Si2
        S2_est = H20 * Si0 + H21 * Si1 + H22 * Si2
        loss = 0
        loss += nn.MSELoss(S0_est, S0)
        loss += nn.MSELoss(S1_est, S1)
        loss += nn.MSELoss(S2_est, S2)
        return loss


class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]      

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

class DNet(nn.Module):
    def __init__(self, input_nc, output_nc,
                 ngf=64, n_downsampling=1, n_blocks=4,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect', sv_num=1):
        super(DNet, self).__init__()
        # Sd 3*1 images
        # Hd 3*3 images
        encoder_model = self.define_encoder(input_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        if sv_num == 1:
            self.encoderS0 = nn.Sequential(*encoder_model)
        else:
            self.encoderS0 = nn.Sequential(*encoder_model)
            self.encoderS1 = nn.Sequential(*encoder_model)
            self.encoderS2 = nn.Sequential(*encoder_model)

        Sd_decoder_model = self.define_Sd_decoder(output_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.Sd_decoderS0 = nn.Sequential(*Sd_decoder_model)
        self.Sd_decoderS1 = nn.Sequential(*Sd_decoder_model)
        self.Sd_decoderS2 = nn.Sequential(*Sd_decoder_model)
        self.name = 'DNet'

    def define_encoder(self, input_nc,
                       ngf=64, n_downsampling=1, n_blocks=4,
                       norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        activation = nn.ReLU(True)
        pad1 = nn.ReflectionPad2d(1)
        conv1 = nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0)
        conv2 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        norm1 = norm_layer(ngf)

        model = []
        model += [pad1, conv1, norm1, activation]
        model += [pad1, conv2, norm1, activation]

        # downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            conv_ds = nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1)
            model += [conv_ds, norm_layer(ngf * mult * 2), activation]

        # resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            dilation_ratio = 2 ** (i + 1)
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                                  norm_layer=norm_layer, dilation_ratio=dilation_ratio)]

        model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                              norm_layer=norm_layer, dilation_ratio=1)]
        return model

    def define_Sd_decoder(self, output_nc,
                          ngf=64, n_downsampling=1, n_blocks=4,
                          norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        activation = nn.ReLU(True)
        pad1 = nn.ReflectionPad2d(1)
        #conv1 = nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0)
        #conv2 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        #norm1 = norm_layer(ngf)
        model = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]

        conv3 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        conv_out = nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0)
        model += [pad1, conv3, norm_layer(ngf), activation]
        model += [pad1, conv_out]
        # self.model = nn.Sequential(*model)
        return model

    def forward(self, S0, S1, S2):
        Sd0 = S0 + self.Sd_decoderS0(self.encoderS0(S0))

        #tt = memory_summary(device=None, abbreviated=False)
        #print(tt)

        #Sd1 = S1 + self.Sd_decoderS1(self.encoderS0(S1))
        #Sd2 = S2 + self.Sd_decoderS2(self.encoderS0(S2))
        Sd1 = Sd0
        Sd2 = Sd0
        return [Sd0, Sd1, Sd2]


class SNet(nn.Module):
    def __init__(self, input_nc, output_nc,
                 ngf=64, n_downsampling=1, n_blocks=3,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(SNet, self).__init__()
        encoder_model = self.define_encoder(input_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.encoderSd0 = nn.Sequential(*encoder_model)
        self.encoderSd1 = nn.Sequential(*encoder_model)
        self.encoderSd2 = nn.Sequential(*encoder_model)

        Hd_decoder_model = self.define_Hd_decoder(output_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.Hd_decoderH00 = nn.Sequential(*Hd_decoder_model)
        self.Hd_decoderH01 = nn.Sequential(*Hd_decoder_model)
        self.Hd_decoderH02 = nn.Sequential(*Hd_decoder_model)
        self.Hd_decoderH10 = nn.Sequential(*Hd_decoder_model)
        self.Hd_decoderH11 = nn.Sequential(*Hd_decoder_model)
        self.Hd_decoderH12 = nn.Sequential(*Hd_decoder_model)
        self.Hd_decoderH20 = nn.Sequential(*Hd_decoder_model)
        self.Hd_decoderH21 = nn.Sequential(*Hd_decoder_model)
        self.Hd_decoderH22 = nn.Sequential(*Hd_decoder_model)
        self.name = 'SNet'

    def define_encoder(self, input_nc,
                       ngf=64, n_downsampling=1, n_blocks=4,
                       norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        activation = nn.ReLU(True)
        pad1 = nn.ReflectionPad2d(1)
        conv1 = nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0)
        conv2 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        norm1 = norm_layer(ngf)

        model = []
        model += [pad1, conv1, norm1, activation]
        model += [pad1, conv2, norm1, activation]

        # downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            conv_ds = nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1)
            model += [conv_ds, norm_layer(ngf * mult * 2), activation]

        # resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            dilation_ratio = 2 ** (i + 1)
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                                  norm_layer=norm_layer, dilation_ratio=dilation_ratio)]

        model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                              norm_layer=norm_layer, dilation_ratio=1)]
        return model

    def define_Hd_decoder(self, output_nc, ngf=64,
                          n_downsampling=1, n_blocks=4, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        activation = nn.ReLU(True)
        pad1 = nn.ReflectionPad2d(1)
        # conv1 = nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0)
        # conv2 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        # norm1 = norm_layer(ngf)
        model = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]

        conv3 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        conv_out = nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0)
        model += [pad1, conv3, norm_layer(ngf), activation]
        model += [pad1, conv_out]
        # self.model = nn.Sequential(*model)
        return model

    def forward(self, Sd0, Sd1, Sd2):
        eSd0 = self.encoderSd0(Sd0)
        eSd1 = self.encoderSd0(Sd1)
        eSd2 = self.encoderSd0(Sd2)
        H00 = Sd0 + self.Hd_decoderH00(eSd0)
        H01 = Sd0 + self.Hd_decoderH01(eSd0)
        H02 = Sd0 + self.Hd_decoderH02(eSd0)

        H10 = Sd1 + self.Hd_decoderH10(eSd1)
        H11 = Sd1 + self.Hd_decoderH11(eSd1)
        H12 = Sd1 + self.Hd_decoderH12(eSd1)

        H20 = Sd1 + self.Hd_decoderH20(eSd2)
        H21 = Sd1 + self.Hd_decoderH21(eSd2)
        H22 = Sd1 + self.Hd_decoderH22(eSd2)
        return [H00, H01, H02, H10, H11, H12, H20, H21, H22]


class PNet(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class ResNet(nn.Module):
    def __init__(self, input_nc, output_nc,
                 ngf=64, n_downsampling=1, n_blocks=3,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(ResNet, self).__init__()
        encoder_model = self.define_encoder(input_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.encoder = nn.Sequential(*encoder_model)

        decoder_model = self.define_decoder(output_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.decoder = nn.Sequential(*decoder_model)
        self.name = 'ResNet'

    def define_encoder(self, input_nc,
                       ngf=64, n_downsampling=1, n_blocks=4,
                       norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        activation = nn.ReLU(True)
        pad1 = nn.ReflectionPad2d(1)
        conv1 = nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0)
        conv2 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        norm1 = norm_layer(ngf)

        model = []
        model += [pad1, conv1, norm1, activation]
        model += [pad1, conv2, norm1, activation]

        # downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            conv_ds = nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1)
            model += [conv_ds, norm_layer(ngf * mult * 2), activation]

        # resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            dilation_ratio = 2 ** (i + 1)
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                                  norm_layer=norm_layer, dilation_ratio=dilation_ratio)]

        model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                              norm_layer=norm_layer, dilation_ratio=1)]
        return model

    def define_decoder(self, output_nc, ngf=64,
                          n_downsampling=1, n_blocks=4, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        activation = nn.ReLU(True)
        pad1 = nn.ReflectionPad2d(1)
        model = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]

        conv3 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        conv_out = nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0)
        model += [pad1, conv3, norm_layer(ngf), activation]
        model += [pad1, conv_out]
        return model

    def forward(self, inputs):
        outputs = inputs + self.decoder(self.encoder(inputs))
        return outputs


class ResNet1x3(nn.Module):
    def __init__(self, input_nc, output_nc,
                 ngf=64, n_downsampling=1, n_blocks=3,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(ResNet1x3, self).__init__()
        encoder_model0 = self.define_encoder(input_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.encoder0 = nn.Sequential(*encoder_model0)

        encoder_model1 = self.define_encoder(input_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.encoder1 = nn.Sequential(*encoder_model1)

        encoder_model2 = self.define_encoder(input_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.encoder2 = nn.Sequential(*encoder_model2)

        decoder_model0 = self.define_decoder(output_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.decoder0 = nn.Sequential(*decoder_model0)

        decoder_model1 = self.define_decoder(output_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.decoder1 = nn.Sequential(*decoder_model1)

        decoder_model2 = self.define_decoder(output_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.decoder2 = nn.Sequential(*decoder_model2)
        self.name = 'ResNet1x3'

    def define_encoder(self, input_nc,
                       ngf=64, n_downsampling=1, n_blocks=4,
                       norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        activation = nn.ReLU(True)
        pad1 = nn.ReflectionPad2d(1)
        conv1 = nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0)
        conv2 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        norm1 = norm_layer(ngf)

        model = []
        model += [pad1, conv1, norm1, activation]
        model += [pad1, conv2, norm1, activation]

        # downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            conv_ds = nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1)
            model += [conv_ds, norm_layer(ngf * mult * 2), activation]

        # resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            dilation_ratio = 2 ** (i + 1)
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                                  norm_layer=norm_layer, dilation_ratio=dilation_ratio)]

        model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                              norm_layer=norm_layer, dilation_ratio=1)]
        return model

    def define_decoder(self, output_nc, ngf=64,
                       n_downsampling=1, n_blocks=4, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        activation = nn.ReLU(True)
        pad1 = nn.ReflectionPad2d(1)
        model = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]

        conv3 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        conv_out = nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0)
        model += [pad1, conv3, norm_layer(ngf), activation]
        model += [pad1, conv_out]
        return model

    def forward(self, s):
        sout0 = s + self.decoder0(self.encoder0(s))
        sout1 = s + self.decoder1(self.encoder1(s))
        sout2 = s + self.decoder2(self.encoder2(s))
        return [sout0, sout1, sout2]


class ResNet3x1(nn.Module):
    def __init__(self, input_nc, output_nc,
                 ngf=64, n_downsampling=1, n_blocks=3,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(ResNet3x1, self).__init__()
        encoder_model0 = self.define_encoder(input_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.encoder0 = nn.Sequential(*encoder_model0)

        encoder_model1 = self.define_encoder(input_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.encoder1 = nn.Sequential(*encoder_model1)

        encoder_model2 = self.define_encoder(input_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.encoder2 = nn.Sequential(*encoder_model2)

        decoder_model0 = self.define_decoder(output_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.decoder0 = nn.Sequential(*decoder_model0)

        decoder_model1 = self.define_decoder(output_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.decoder1 = nn.Sequential(*decoder_model1)

        decoder_model2 = self.define_decoder(output_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.decoder2 = nn.Sequential(*decoder_model2)
        self.name = 'ResNet3x1'

    def define_encoder(self, input_nc,
                       ngf=64, n_downsampling=1, n_blocks=4,
                       norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        activation = nn.ReLU(True)
        pad1 = nn.ReflectionPad2d(1)
        conv1 = nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0)
        conv2 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        norm1 = norm_layer(ngf)

        model = []
        model += [pad1, conv1, norm1, activation]
        model += [pad1, conv2, norm1, activation]

        # downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            conv_ds = nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1)
            model += [conv_ds, norm_layer(ngf * mult * 2), activation]

        # resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            dilation_ratio = 2 ** (i + 1)
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                                  norm_layer=norm_layer, dilation_ratio=dilation_ratio)]

        model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                              norm_layer=norm_layer, dilation_ratio=1)]
        return model

    def define_decoder(self, output_nc, ngf=64,
                       n_downsampling=1, n_blocks=4, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        activation = nn.ReLU(True)
        pad1 = nn.ReflectionPad2d(1)
        model = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]

        conv3 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        conv_out = nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0)
        model += [pad1, conv3, norm_layer(ngf), activation]
        model += [pad1, conv_out]
        return model

    def forward(self, inputs=[]):
        [s0, s1, s2] = inputs
        sout0 = s0 + self.decoder0(self.encoder0(s0))
        sout1 = s1 + self.decoder1(self.encoder1(s1))
        sout2 = s2 + self.decoder2(self.encoder2(s2))
        return [sout0, sout1, sout2]


class ResNet1x1(nn.Module):
    def __init__(self, input_nc, output_nc,
                 ngf=64, n_downsampling=1, n_blocks=3,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(ResNet1x1, self).__init__()
        encoder_model0 = self.define_encoder(input_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.encoder0 = nn.Sequential(*encoder_model0)

        decoder_model0 = self.define_decoder(output_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.decoder0 = nn.Sequential(*decoder_model0)

        self.name = 'ResNet1x1'

    def define_encoder(self, input_nc,
                       ngf=64, n_downsampling=1, n_blocks=4,
                       norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        activation = nn.ReLU(True)
        pad1 = nn.ReflectionPad2d(1)
        conv1 = nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0)
        conv2 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        norm1 = norm_layer(ngf)

        model = []
        model += [pad1, conv1, norm1, activation]
        model += [pad1, conv2, norm1, activation]

        # downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            conv_ds = nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1)
            model += [conv_ds, norm_layer(ngf * mult * 2), activation]

        # resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            dilation_ratio = 2 ** (i + 1)
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                                  norm_layer=norm_layer, dilation_ratio=dilation_ratio)]

        model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                              norm_layer=norm_layer, dilation_ratio=1)]
        return model

    def define_decoder(self, output_nc, ngf=64,
                       n_downsampling=1, n_blocks=4, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        activation = nn.ReLU(True)
        pad1 = nn.ReflectionPad2d(1)
        model = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]

        conv3 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        conv_out = nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0)
        model += [pad1, conv3, norm_layer(ngf), activation]
        model += [pad1, conv_out]
        return model

    def forward(self, x):
        sout = x + self.decoder0(self.encoder0(x))
        return sout


class ResNet1x2(nn.Module):
    def __init__(self, input_nc, output_nc,
                 ngf=64, n_downsampling=1, n_blocks=3,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(ResNet1x2, self).__init__()
        encoder_model0 = self.define_encoder(input_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.encoder0 = nn.Sequential(*encoder_model0)

        decoder_model0 = self.define_decoder(output_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.decoder0 = nn.Sequential(*decoder_model0)

        encoder_model1 = self.define_encoder(input_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.encoder1 = nn.Sequential(*encoder_model1)

        decoder_model1 = self.define_decoder(output_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.decoder1 = nn.Sequential(*decoder_model1)

        self.name = 'ResNet1x2'

    def define_encoder(self, input_nc,
                       ngf=64, n_downsampling=1, n_blocks=4,
                       norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        activation = nn.ReLU(True)
        pad1 = nn.ReflectionPad2d(1)
        conv1 = nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0)
        conv2 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        norm1 = norm_layer(ngf)

        model = []
        model += [pad1, conv1, norm1, activation]
        model += [pad1, conv2, norm1, activation]

        # downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            conv_ds = nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1)
            model += [conv_ds, norm_layer(ngf * mult * 2), activation]

        # resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            dilation_ratio = 2 ** (i + 1)
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                                  norm_layer=norm_layer, dilation_ratio=dilation_ratio)]

        model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                              norm_layer=norm_layer, dilation_ratio=1)]
        return model

    def define_decoder(self, output_nc, ngf=64,
                       n_downsampling=1, n_blocks=4, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        activation = nn.ReLU(True)
        pad1 = nn.ReflectionPad2d(1)
        model = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]

        conv3 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        conv_out = nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0)
        model += [pad1, conv3, norm_layer(ngf), activation]
        model += [pad1, conv_out]
        return model

    def forward(self, x):
        s0 = x + self.decoder0(self.encoder0(x))
        s1 = x + self.decoder1(self.encoder1(x))
        return [s0, s1]


class HNet3x3(nn.Module):
    def __init__(self, input_nc, output_nc,
                 ngf=64, n_downsampling=1, n_blocks=3,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(HNet3x3, self).__init__()
        encoder_model = self.define_encoder(input_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.encoder = nn.Sequential(*encoder_model)
        #self.encoderSd1 = nn.Sequential(*encoder_model)
        #self.encoderSd2 = nn.Sequential(*encoder_model)

        H_decoder_model = self.define_H_decoder(output_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.decoder = nn.Sequential(*H_decoder_model)
        #self.Hd_decoderH1 = nn.Sequential(*H_decoder_model)
        #self.Hd_decoderH2 = nn.Sequential(*H_decoder_model)
        self.name = 'HNet3x3'

    def define_encoder(self, input_nc,
                       ngf=64, n_downsampling=1, n_blocks=4,
                       norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        activation = nn.ReLU(True)
        pad1 = nn.ReflectionPad2d(1)
        conv1 = nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0)
        conv2 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        norm1 = norm_layer(ngf)

        model = []
        model += [pad1, conv1, norm1, activation]
        model += [pad1, conv2, norm1, activation]

        # downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            conv_ds = nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1)
            model += [conv_ds, norm_layer(ngf * mult * 2), activation]

        # resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            dilation_ratio = 2 ** (i + 1)
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                                  norm_layer=norm_layer, dilation_ratio=dilation_ratio)]

        model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                              norm_layer=norm_layer, dilation_ratio=1)]
        return model

    def define_H_decoder(self, output_nc, ngf=64,
                          n_downsampling=1, n_blocks=4, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        activation = nn.ReLU(True)
        pad1 = nn.ReflectionPad2d(1)
        # conv1 = nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0)
        # conv2 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        # norm1 = norm_layer(ngf)
        model = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]

        conv3 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        conv_out = nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0)
        model += [pad1, conv3, norm_layer(ngf), activation]
        model += [pad1, conv_out]
        # self.model = nn.Sequential(*model)
        return model

    def forward(self, Sout):
        h0 = self.decoder(self.encoder(Sout))
        return h0


class HNet(nn.Module):
    def __init__(self, input_nc, output_nc,
                 ngf=64, n_downsampling=1, n_blocks=3,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(HNet, self).__init__()
        encoder_model = self.define_encoder(input_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.encoderSd0 = nn.Sequential(*encoder_model)
        #self.encoderSd1 = nn.Sequential(*encoder_model)
        #self.encoderSd2 = nn.Sequential(*encoder_model)

        H_decoder_model = self.define_H_decoder(output_nc, ngf, n_downsampling, n_blocks, norm_layer, padding_type)
        self.Hd_decoderH0 = nn.Sequential(*H_decoder_model)
        self.Hd_decoderH1 = nn.Sequential(*H_decoder_model)
        self.Hd_decoderH2 = nn.Sequential(*H_decoder_model)
        self.name = 'HNet'

    def define_encoder(self, input_nc,
                       ngf=64, n_downsampling=1, n_blocks=4,
                       norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        activation = nn.ReLU(True)
        pad1 = nn.ReflectionPad2d(1)
        conv1 = nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0)
        conv2 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        norm1 = norm_layer(ngf)

        model = []
        model += [pad1, conv1, norm1, activation]
        model += [pad1, conv2, norm1, activation]

        # downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            conv_ds = nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1)
            model += [conv_ds, norm_layer(ngf * mult * 2), activation]

        # resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            dilation_ratio = 2 ** (i + 1)
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                                  norm_layer=norm_layer, dilation_ratio=dilation_ratio)]

        model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation,
                              norm_layer=norm_layer, dilation_ratio=1)]
        return model

    def define_H_decoder(self, output_nc, ngf=64,
                          n_downsampling=1, n_blocks=4, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        activation = nn.ReLU(True)
        pad1 = nn.ReflectionPad2d(1)
        # conv1 = nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0)
        # conv2 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        # norm1 = norm_layer(ngf)
        model = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]

        conv3 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        conv_out = nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0)
        model += [pad1, conv3, norm_layer(ngf), activation]
        model += [pad1, conv_out]
        # self.model = nn.Sequential(*model)
        return model

    def forward(self, Sout):
        feat = self.encoderSd0(Sout)
        H0 = self.Hd_decoderH0(feat)
        H1 = self.Hd_decoderH1(feat)
        H2 = self.Hd_decoderH2(feat)
        return [H0, H1, H2]


class DiffuseGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=1, n_blocks=4, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(DiffuseGenerator, self).__init__()
        activation = nn.ReLU(True)
        pad1 = nn.ReflectionPad2d(1)
        conv1 = nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0)
        conv2 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        #conv_k3f64 = nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0)
        norm1 = norm_layer(ngf)

        model = []
        model += [pad1, conv1, norm1, activation]
        model += [pad1, conv2, norm1, activation]

        # downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            conv_ds = nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1)
            model += [conv_ds, norm_layer(ngf * mult * 2), activation]

        # resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            dilation_ratio = 2**(i+1)
            model += [ResnetBlock(ngf*mult, padding_type=padding_type, activation=activation,
                                  norm_layer=norm_layer, dilation_ratio=dilation_ratio)]

        model += [ResnetBlock(ngf*mult, padding_type=padding_type, activation=activation,
                              norm_layer=norm_layer, dilation_ratio=1)]

        # upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]

        conv3 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=0)
        conv_out = nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0)
        model += [pad1, conv3, norm_layer(ngf), activation]
        model += [pad1, conv_out]
        #model += [conv_l3, conv_l4]
        #pprint(model)
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x+self.model(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False, dilation_ratio=1):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim=dim, padding_type=padding_type, norm_layer=norm_layer,
                                                activation=activation, use_dropout=use_dropout,
                                                dilation_ratio=dilation_ratio)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout, dilation_ratio=1):
        conv_block = []
        p = dilation_ratio
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(p)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(p)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, dilation=dilation_ratio),
                       norm_layer(dim), activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = dilation_ratio
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(p)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(p)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, dilation=dilation_ratio), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        #print(x.shape)
        #print(self.conv_block(x).shape)
        out = x + self.conv_block(x)
        return out


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)


class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()        
        self.output_nc = output_nc        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 norm_layer(ngf), nn.ReLU(True)]             
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model) 

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))        
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i)).nonzero() # n x 4            
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]                    
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)                                        
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat                       
        return outputs_mean


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
