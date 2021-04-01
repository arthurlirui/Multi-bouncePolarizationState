import numpy as np
import torch
import os
import sys
#sys.path.append('./p2phd')
from torch.autograd import Variable
from util.image_pool import ImagePool
#from base_model import BaseModel
#import networks
from models.base_model import BaseModel
from models import networks
#from models.base_model import *
from torchvision.transforms import transforms
from models.networks import *
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class SuperpixelLighting:
    def __init__(self):
        pass

    def load_label(self, path, filename):
        fp = open(os.path.join(path, filename), 'rb+')
        rawlabel = np.fromfile(fp, dtype=np.uint16)
        imglabel = np.reshape(rawlabel, [self.h, self.w])
        self.label = imglabel

    def load_img(self, path, filename):
        img = Image.open(os.path.join(path, filename))
        self.img = np.array(img)
        [h, w, d] = self.img.shape
        self.h = h
        self.w = w
        self.d = d

    def load_minimg(self, path, filename):
        img = Image.open(os.path.join(path, filename))
        self.minimg = np.array(img)

    def proc_label(self):
        [h, w, d] = self.img.shape
        imgout = self.img.copy()
        #imgnd = np.reshape(self.img, [h*w, d])
        #labelnd = np.reshape(self.label, [h*w, 1])
        labelset = np.unique(self.label)
        nlabel = len(labelset)
        # calculate maxcolor in superpixel
        for i in labelset:
            maski = self.label == i
            indy, indx = np.where(maski)
            supi = self.img[indy, indx, :]
            maxi = np.max(supi, axis=0)
            avgi = np.mean(supi, 1)
            imgout[indy, indx, :] = maxi
        return imgout


class PolarPhysicalModel(BaseModel):
    '''
    this model estimate unit environment illumination
    '''
    def __init__(self):
        super(PolarPhysicalModel, self).__init__()
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def initialize(self, opt):
        pass

    def forward(self, Iin=[]):
        [I0, I45, I90, I135] = Iin
        # calculate Imin
        Imin = torch.min(torch.min(torch.min(I0, I45), I90), I135)
        [I0_, I45_, I90_, I135_] = [I0-Imin, I45-Imin, I90-Imin, I135-Imin]

        # calculate Stokes vector
        S0 = 0.5 * (I0 + I45 + I90 + I135)
        S1 = I0 - I90
        S2 = I45 - I135

        # calc DOLP AOLP
        AOLP = 0.5 * torch.atan(S2 / S1)
        DOLP = torch.sqrt(S1 * S1 + S2 * S2) / (S0+0.001)

        # estimate unit global illumination
        # (1) unit illumination: average of light color (high intensity area)
        # (2) convert to Stokes vector as illumination input
        return {'S': [S0, S1, S2], 'DOLP': DOLP, 'AOLP': AOLP,
                'Imin': Imin, 'Ip': [I0_, I45_, I90_, I135_]}


class ShadingModel(BaseModel):
    def __init__(self):
        super(ShadingModel, self).__init__()

    def name(self):
        return 'ShadingModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = 6
        output_nc = 3
        ngf = 32
        netname = 'HNet3x3'
        n_downsample_global = 2
        n_blocks_global = 3
        n_blocks_local = 0

        self.ShadingNet = networks.define_G(input_nc, output_nc, ngf, netname, n_downsample_global,
                                            n_blocks_global, n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        self.spatial_gradient = SpatialGradient()
        self.optimizer_HNet = torch.optim.Adam(self.ShadingNet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.L2Loss = torch.nn.MSELoss()

    def forward(self, Sd0, S0, weight=0.3):
        '''
        :param Sd0: diffuse image
        :param S0: original or depolarized image
        :return:
        '''
        inputs = torch.cat([Sd0, S0], dim=1)
        H00 = self.ShadingNet(inputs)
        data_loss = self.L2Loss(S0, H00 * Sd0)
        gradlist = self.spatial_gradient(H00)
        smooth_loss = [torch.mean(grad.pow(2)) for grad in gradlist]
        smooth_loss = torch.sum(torch.Tensor(smooth_loss))
        total_loss = data_loss + weight * smooth_loss
        return {'H00': H00, 'L': total_loss, 'Ld': data_loss, 'Lsmooth': smooth_loss}


class HModel3x1(BaseModel):
    def __init__(self):
        super(HModel3x1, self).__init__()

    def name(self):
        return 'HModel3x1'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = 3
        output_nc = 3
        ngf = 24
        netname = 'ResNet1x3'
        n_downsample_global = 2
        n_blocks_global = 2
        n_blocks_local = 0

        self.network = networks.define_G(input_nc, output_nc, ngf, netname, n_downsample_global,
                                         n_blocks_global, n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        self.spatial_gradient = SpatialGradient()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.L2Loss = torch.nn.MSELoss()
        #self.L2norm = torch.mean()

    def H0_loss(self, I_phi, h0=[], phi=0, lambda_h=0.1):
        [h00, h10, h20] = h0
        I_phi_est = 0.5*(h00+h10*torch.cos(2*phi)+h20*torch.sin(2*phi))
        l2 = torch.mean(h00 ** 2) + torch.mean(h10 ** 2) + torch.mean(h20 ** 2)
        total_loss = self.L2Loss(I_phi, I_phi_est) + lambda_h * l2
        return total_loss

    def H0_all_loss(self, h0=[], hd0=[], hs0=[]):
        [h00, h10, h20] = h0
        [hd00, hd10, hd20] = hd0
        [hs00, hs10, hs20] = hs0
        total_loss = self.L2Loss(h00, hd00+hs00)+self.L2Loss(h10, hd10+hs10)+self.L2Loss(h20, hd20+hs20)
        return total_loss

    # def Hd0_loss(self, Idp_phi, h0=[], phi=0, lambda_h=0.1):
    #     [h00, h10, h20] = h0
    #     I_phi_est = 0.5*(h00+h10*torch.cos(2*phi)+h20*torch.sin(2*phi))
    #     l2 = torch.mean(h00 ** 2) + torch.mean(h10 ** 2) + torch.mean(h20 ** 2)
    #     total_loss = self.L2Loss(Idp_phi, I_phi_est) + lambda_h * l2
    #     return total_loss
    #
    # def Hs0_loss(self, Is_phi, h0=[], phi=0, lambda_h=0.1):
    #     [h00, h10, h20] = h0
    #     I_phi_est = 0.5*(h00+h10*torch.cos(2*phi)+h20*torch.sin(2*phi))
    #     l2 = torch.mean(h00 ** 2) + torch.mean(h10 ** 2) + torch.mean(h20 ** 2)
    #     total_loss = self.L2Loss(Is_phi, I_phi_est) + lambda_h * l2
    #     return total_loss

    def forward(self, I_phi, phi):
        [h00, h10, h20] = self.network(I_phi)
        I_phi_est = 0.5*(h00+h10*torch.cos(2*phi)+h20*torch.sin(2*phi))
        #l2 = torch.mean(h00**2) + torch.mean(h10**2) + torch.mean(h20**2)
        total_loss = self.H0_loss(I_phi=I_phi, h0=[h00, h10, h20], phi=phi)
        return {'H00': h00, 'H10': h10, 'H20': h20, 'L': total_loss}


class DePolarModel(BaseModel):
    def __init__(self):
        super(DePolarModel, self).__init__()

    def name(self):
        return 'DePolarModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = 3
        output_nc = 3
        ngf = 32
        netname = 'ResNet1x1'
        n_downsample_global = 3
        n_blocks_global = 2
        n_blocks_local = 0

        self.DePolarNet = networks.define_G(input_nc, output_nc, ngf, netname, n_downsample_global,
                                            n_blocks_global, n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        self.spatial_gradient = SpatialGradient()
        self.optimizer_ResNet1x1 = torch.optim.Adam(self.DePolarNet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.L2Loss = torch.nn.MSELoss()
        self.L1Loss = torch.nn.L1Loss()

    def forward(self, I_phi, I_min, dolp, kp):
        '''
        :param Sin: diffuse image
        :return:
        '''
        I_phi_de = self.DePolarNet(I_phi)
        loss1 = self.L2Loss(I_phi, I_phi_de)
        #loss2 = torch.mean(kp*dolp.expand_as(I_min)*torch.pow(I_phi_de-I_min, 2))

        loss2 = torch.mean(kp*dolp.expand_as(I_min)*(I_phi_de-I_min)**2)

        total_loss = loss1 + loss2
        return {'Ide': I_phi_de, 'L': total_loss, 'L1': loss1, 'L2': loss2}
        # inputs = torch.cat(S, dim=1)
        # outputs = self.DePolarNet(inputs)
        # [S0, S1, S2] = S
        # [Sup0, Sup1, Sup2, Sp0, Sp1, Sp2] = torch.split(outputs, split_size_or_sections=[3, 3, 3, 3, 3, 3], dim=1)
        # s0_loss = self.L2Loss(S0, Sup0 + Sp0)
        # s1_loss = self.L2Loss(S1, Sup1 + Sp1)
        # s2_loss = self.L2Loss(S2, Sup2 + Sp2)
        # data_loss = s0_loss + s1_loss + s2_loss
        # content_loss = self.L2Loss(Sup0, Sp0)
        # up_loss = torch.mean(torch.abs(Sup1)) + torch.mean(torch.abs(Sup2))
        # total_loss = data_loss + content_loss + up_loss
        # return {'Sup': [Sup0, Sup1, Sup2], 'Sp': [Sp0, Sp1, Sp2], 'L': total_loss, 'Ld': data_loss, 'Lc': content_loss, 'Lup': up_loss}


class SpecularModel(BaseModel):
    def __init__(self):
        super(SpecularModel, self).__init__()

    def name(self):
        return 'SpecularModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = 6
        output_nc = 3
        ngf = 32
        netname = 'HNet3x3'
        n_downsample_global = 2
        n_blocks_global = 3
        n_blocks_local = 0

        self.SpecularNet = networks.define_G(input_nc, output_nc, ngf, netname, n_downsample_global,
                                            n_blocks_global, n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        self.spatial_gradient = SpatialGradient()
        self.optimizer_HNet = torch.optim.Adam(self.ShadingNet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.L2Loss = torch.nn.MSELoss()

    def forward(self, Sd0, S0, weight=0.3):
        '''
        :param Sd0: diffuse image
        :param S0: original or depolarized image
        :return:
        '''
        inputs = torch.cat([Sd0, S0], dim=1)
        H00 = self.SpecularNet(inputs)
        data_loss = self.L2Loss(S0, H00 * Sd0)
        gradlist = self.spatial_gradient(H00)
        smooth_loss = [torch.mean(grad.pow(2)) for grad in gradlist]
        smooth_loss = torch.sum(torch.Tensor(smooth_loss))
        total_loss = data_loss + weight * smooth_loss
        return {'H00': H00, 'L': total_loss, 'Ld': data_loss, 'Lsmooth': smooth_loss}


class ShadingMatrixModel3x3(BaseModel):
    def __init__(self):
        super(ShadingMatrixModel, self).__init__()

    def name(self):
        return 'ShadingMatrixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.input_nc
        opt.ngf = 32
        self.HNet = []
        for yy in range(3):
            hnet_row = []
            for xx in range(3):
                hnet = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'HNet9x9',
                                         opt.n_downsample_global, opt.n_blocks_global,
                                         opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
                hnet_row.append(hnet)
            self.HNet.append(hnet_row)

        self.spatial_gradient = SpatialGradient()

    def forward(self, Sout=[], Sin=[]):
        [S0, S1, S2] = Sout
        hnet00 = self.HNet[0][0]
        h00 = hnet00(S0)
        torch.cuda.empty_cache()

        hnet01 = self.HNet[0][1]
        h01 = hnet01(S0)
        torch.cuda.empty_cache()

        hnet02 = self.HNet[0][2]
        h02 = hnet02(S0)
        torch.cuda.empty_cache()

        hnet10 = self.HNet[1][0]
        h10 = hnet10(S1)
        torch.cuda.empty_cache()

        hnet11 = self.HNet[1][1]
        h11 = hnet11(S1)
        torch.cuda.empty_cache()

        hnet12 = self.HNet[1][2]
        h12 = hnet12(S1)
        torch.cuda.empty_cache()

        hnet20 = self.HNet[2][0]
        h20 = hnet20(S2)
        torch.cuda.empty_cache()

        hnet21 = self.HNet[2][1]
        h21 = hnet21(S2)
        torch.cuda.empty_cache()

        hnet22 = self.HNet[2][2]
        h22 = hnet22(S2)
        torch.cuda.empty_cache()

        data_loss = self.L1Loss(S0, h00 * Sin[0] + h01 * Sin[1] + h02 * Sin[2])

        # process diffuse parts

        [H0, H1, H2] = self.HNet(Sout)
        #shading_loss = self.L1Loss(Sout, H0*S0+H1*S1+H2*S2)

        h0_loss = 0
        h1_loss = 0
        h2_loss = 0
        if row == 1:
            gradlist = self.spatial_gradient(H0)
            for grad in gradlist:
                h0_loss += torch.mean(torch.abs(grad))
            h1_loss += 0.0001*torch.mean(torch.abs(H1))
            h2_loss += 0.0001*torch.mean(torch.abs(H2))
        elif row == 2 or row == 3:
            h0_loss += 0.0001*torch.mean(torch.abs(H0))
            h1_loss += torch.mean(torch.abs(H1))
            h2_loss += torch.mean(torch.abs(H2))
        shading_loss = data_loss+h0_loss+h1_loss+h2_loss

        return {'Ls': shading_loss, 'H': [H0, H1, H2], 'Ld': data_loss, 'L': [h0_loss, h1_loss, h2_loss]}

    def save(self, which_epoch):
        #self.save_network(self.SNet, 'SNet', which_epoch, self.gpu_ids)
        self.save_network(self.HNet, 'HNet', which_epoch, self.gpu_ids)
        #if self.gen_features:
        #    self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


class ShadingMatrixModel(BaseModel):
    def __init__(self):
        super(ShadingMatrixModel, self).__init__()

    def name(self):
        return 'ShadingMatrixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        # self.use_features = opt.instance_feat or opt.label_feat
        # self.gen_features = self.use_features and not self.opt.load_features
        # input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        input_nc = opt.input_nc
        opt.ngf = 32
        self.HNet = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'HNet',
                                      opt.n_downsample_global, opt.n_blocks_global,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        self.spatial_gradient = SpatialGradient()

        if self.opt.verbose:
            print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            # self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            self.load_network(self.HNet, self.HNet.name, opt.which_epoch, pretrained_path)
            #self.load_network(self.SNet, self.SNet.name, opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            # if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
            #    raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            # self.fake_pool = ImagePool(opt.pool_size)
            # self.old_lr = opt.lr

            # define loss functions
            #self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.L1Loss = torch.nn.L1Loss()

            # add our loss here
            minbatch = 1
            channels = 3
            self.criterionDiffuse = networks.DiffuseLoss(minbatch, channels)
            #self.criterionShading = networks.ShadingMatrixLoss()

            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # Names so we can breakout loss
            # self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake')

            # initialize optimizers
            self.optimizer_HNet = torch.optim.Adam(self.HNet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer_SNet = torch.optim.Adam(self.SNet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def forward(self, Sout, Sin=[], row=1):
        # process diffuse parts
        [S0, S1, S2] = Sin
        [H0, H1, H2] = self.HNet(Sout)
        #shading_loss = self.L1Loss(Sout, H0*S0+H1*S1+H2*S2)
        data_loss = self.L1Loss(Sout, H0*S0+H1*S1+H2*S2)
        h0_loss = 0
        h1_loss = 0
        h2_loss = 0
        if row == 1:
            gradlist = self.spatial_gradient(H0)
            for grad in gradlist:
                h0_loss += torch.mean(torch.abs(grad))
            h1_loss += 0.0001*torch.mean(torch.abs(H1))
            h2_loss += 0.0001*torch.mean(torch.abs(H2))
        elif row == 2 or row == 3:
            h0_loss += 0.0001*torch.mean(torch.abs(H0))
            h1_loss += torch.mean(torch.abs(H1))
            h2_loss += torch.mean(torch.abs(H2))
        shading_loss = data_loss+h0_loss+h1_loss+h2_loss

        return {'Ls': shading_loss, 'H': [H0, H1, H2], 'Ld': data_loss, 'L': [h0_loss, h1_loss, h2_loss]}

    def save(self, which_epoch):
        #self.save_network(self.SNet, 'SNet', which_epoch, self.gpu_ids)
        self.save_network(self.HNet, 'HNet', which_epoch, self.gpu_ids)
        #if self.gen_features:
        #    self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


class SmoothingModel(BaseModel):
    def __init__(self):
        super(SmoothingModel, self).__init__()

    def name(self):
        return 'SmoothingModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = 3
        output_nc = 3
        ngf = 32
        netname = 'ResNet1x2'
        n_downsample_global = 3
        n_blocks_global = 2
        n_blocks_local = 0

        self.network = networks.define_G(input_nc, output_nc, ngf, netname, n_downsample_global,
                                         n_blocks_global, n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        self.spatial_gradient = SpatialGradient()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.L2Loss = torch.nn.MSELoss()
        self.L1Loss = torch.nn.L1Loss()

    def loss_a(self, Id, Is, I_phi=[], Idp=[], Ip=[]):
        total_loss = 0
        for i in range(len(I_phi)):
            total_loss += self.L2Loss(I_phi[i], Id+Is)
            total_loss += self.L2Loss(Idp[i], Id)
            total_loss += self.L2Loss(Ip[i], Is)
        return total_loss

    def loss_d(self, Id, rho, kd=1):
        gradlist = self.spatial_gradient(Id)
        total_loss = 0
        rho[rho>1]=1
        rho[rho<0]=0
        for grad in gradlist:
            #loss = torch.mean(torch.pow(torch.abs(grad), kd*rho))
            loss = torch.mean(torch.pow(torch.abs(grad), 1))
            total_loss += loss
        return total_loss

    def loss_s(self, Is, rho):
        eps = 0.0001
        rho_weight = rho
        mean_rho = torch.mean(rho_weight)
        rho_weight[rho_weight > mean_rho] = 10
        rho_weight[rho_weight <= mean_rho] = 1
        loss = torch.mean(torch.abs(Is)/(rho_weight+eps))
        #loss = torch.mean(torch.abs(Is))
        return loss

    def forward(self, I_phi, Idp, Ip, rho, kd=1, weights=[1, 1, 10]):
        [Id, Is] = self.network(I_phi)
        La = self.loss_a(Id=Id, Is=Is, I_phi=I_phi, Idp=Idp, Ip=Ip)
        Ld = self.loss_d(Id=Id, rho=rho, kd=1)
        Ls = self.loss_s(Is=Is, rho=rho)
        total_loss = weights[0]*La+weights[1]*Ld+weights[2]*Ls
        #total_loss = weights[0] * La
        #Is[Is < 0] = 0
        #Is[Is == 1] = 1
        return {'Id': Id, 'Is': Is, 'L': total_loss, 'La': La, 'Ld': Ld, 'Ls': Ls}


class DiffuseModel(BaseModel):
    def __init__(self):
        super(DiffuseModel, self).__init__()

    def name(self):
        return 'DiffuseModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        # self.use_features = opt.instance_feat or opt.label_feat
        # self.gen_features = self.use_features and not self.opt.load_features
        # input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        input_nc = opt.input_nc
        self.DNet = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'DNet',
                                      opt.n_downsample_global, opt.n_blocks_global,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        if self.opt.verbose:
            print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            # self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            self.load_network(self.DNet, self.DNet.name, opt.which_epoch, pretrained_path)
            #self.load_network(self.SNet, self.SNet.name, opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            # if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
            #    raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            # self.fake_pool = ImagePool(opt.pool_size)
            # self.old_lr = opt.lr

            # define loss functions
            #self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.L1Loss = torch.nn.L1Loss()

            # add our loss here
            minbatch = 1
            channels = 3
            self.criterionDiffuse = networks.DiffuseLoss(minbatch, channels)
            #self.criterionShading = networks.ShadingMatrixLoss()

            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # Names so we can breakout loss
            # self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake')

            # initialize optimizers
            self.optimizer_DNet = torch.optim.Adam(self.DNet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer_SNet = torch.optim.Adam(self.SNet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def forward(self, Sout=[], Sin=[]):
        # process diffuse parts
        [S0, S1, S2] = Sout
        Sd = self.DNet(S0, S1, S2)
        [Sd0, Sd1, Sd2] = Sd
        #Hd = self.SNet(Sd0, Sd1, Sd2)
        data_loss = self.L1Loss(S0, Sd0)
        diffuse_loss = self.criterionDiffuse(Sd0, Sd1, Sd2)
        #shading_loss = self.criterionShading(Sout, Sin, Hd)
        return {'La': data_loss, 'Ld': diffuse_loss, 'Sd': Sd}

    def save(self, which_epoch):
        #self.save_network(self.SNet, 'SNet', which_epoch, self.gpu_ids)
        self.save_network(self.DNet, 'DNet', which_epoch, self.gpu_ids)
        #if self.gen_features:
        #    self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


class PLDModel(BaseModel):
    def __init__(self):
        super(PLDModel, self).__init__()
        # init model
        #self.initialize(opt)

    def name(self):
        return 'PLDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True)
        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):
            return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, d_real, d_fake), flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        #self.use_features = opt.instance_feat or opt.label_feat
        #self.gen_features = self.use_features and not self.opt.load_features
        #input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        input_nc = opt.input_nc
        #netG_input_nc = input_nc
        # define network
        #self.phy_model = PolarPhysicalModel()
        self.DNet = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'DNet',
                                      opt.n_downsample_global, opt.n_blocks_global,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        self.SNet = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'SNet',
                                      opt.n_downsample_global, opt.n_blocks_global,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            #self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            self.load_network(self.DNet, self.DNet.name, opt.which_epoch, pretrained_path)
            self.load_network(self.SNet, self.SNet.name, opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            #if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
            #    raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            #self.fake_pool = ImagePool(opt.pool_size)
            #self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()

            # add our loss here
            minbatch = 1
            channels = 3
            self.criterionDiffuse = networks.DiffuseLoss(minbatch, channels)
            self.criterionShading = networks.ShadingMatrixLoss()

            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
        
            # Names so we can breakout loss
            #self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake')

            # initialize optimizers
            self.optimizer_DNet = torch.optim.Adam(self.DNet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_SNet = torch.optim.Adam(self.SNet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def forward(self, Sout=[], Sin=[]):
        # process diffuse parts
        [S0, S1, S2] = Sout
        Sd = self.DNet(S0, S1, S2)
        [Sd0, Sd1, Sd2] = Sd
        Hd = self.SNet(Sd0, Sd1, Sd2)
        diffuse_loss = self.criterionDiffuse(Sd0, Sd1, Sd2)
        shading_loss = self.criterionShading(Sout, Sin, Hd)
        return {'D': diffuse_loss, 'S': shading_loss}

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.SNet, 'SNet', which_epoch, self.gpu_ids)
        self.save_network(self.DNet, 'DNet', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


class InferenceModel(PLDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)


if __name__ == '__main__':
    if False:
        from PIL import Image
        import torchvision.transforms as transforms
        imgpath = '/home/lir0b/data/allpolardata/polardataset/1/1_0d.png'
        I = Image.open(imgpath)
        trans = transforms.Compose([transforms.ToTensor()])
        tt = trans(I)
        print(tt)
    if True:
        import os
        imgpath = '/home/lir0b/data/allpolardata/polardataset/1/'
        I0 = Image.open(os.path.join(imgpath, '1_0d.png'))
        I45 = Image.open(os.path.join(imgpath, '1_45d.png'))
        I90 = Image.open(os.path.join(imgpath, '1_90d.png'))
        I135 = Image.open(os.path.join(imgpath, '1_135d.png'))
        ppm = PolarPhysicalModel()
        Iin = [I0, I45, I90, I135]
        [S0, S1, S2, AOLP, DOLP] = ppm(Iin)


