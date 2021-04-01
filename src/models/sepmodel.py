import numpy as np
import torch
import os
import sys
from torch.autograd import Variable
#from util.image_pool import ImagePool
#from base_model import BaseModel
#import networks
from pid.models.base_model import BaseModel
from pid.models.networks import *
#from models.base_model import *
from torchvision.transforms import transforms
#from models.networks import *
import matplotlib.pyplot as plt
from PIL import Image
from pid.models.sep_networks import PolarFusion, PolarNet, VGGFeatureBlock
import torchvision.utils as vutils


class SepModel(BaseModel):
    def __init__(self):
        super(SepModel, self).__init__()
        self.epoch = 0

    def name(self):
        return 'SepModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        # input_nc = 6
        # output_nc = 3
        # ngf = 32
        # netname = 'HNet3x3'
        # n_downsample_global = 2
        # n_blocks_global = 3
        # n_blocks_local = 0
        #
        # self.ShadingNet = networks.define_G(input_nc, output_nc, ngf, netname, n_downsample_global,
        #                                     n_blocks_global, n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        #
        # self.spatial_gradient = SpatialGradient()

        #self.L2Loss = torch.nn.MSELoss()

        self.mainnet = PolarNet(input_nc=3, output_nc=3, ngf=64)
        self.refinenet = PolarFusion(input_nc=24, output_nc=3, ngf=64)

        #self.l2 = nn.MSELoss()
        #self.vggfeat = VGGFeatureBlock()
        #self.optimizer = torch.optim.Adam(params=[self.mainnet.parameters(), self.refinenet.parameters()], lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer = torch.optim.Adam([{'params': self.mainnet.parameters()},
                                           {'params': self.refinenet.parameters()}],
                                          lr=opt.lr, betas=(opt.beta1, 0.999))

        self.l2 = nn.MSELoss()
        self.vggfeat = VGGFeatureBlock()
        #self.optimizer_ref = torch.optim.Adam(params=self.refinenet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def loss_phycal_model(self, I=[], Igt=[]):
        pass

    def loss(self, It_refine, It_gt, It_pred=[], It_gt_polar=[]):
        loss_p = self.loss_trans_polar(It_pred, It_gt_polar)
        loss_r = self.loss_trans_refine(It_refine, It_gt)
        loss_vgg = self.loss_vgg(It_refine, It_gt)
        loss_t = loss_p+loss_r+0.0001*loss_vgg
        loss_dict = {}
        loss_dict['polar'] = loss_p
        loss_dict['ref'] = loss_r
        loss_dict['vgg'] = loss_vgg
        loss_dict['total'] = loss_t
        return loss_dict

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

    def forward(self, I_obs=[], I_gt=[]):
        [I_obs1, I_obs2, I_obs3, I_obs4] = I_obs
        [I_gt1, I_gt2, I_gt3, I_gt4] = I_gt
        I_gt_main = 0.25*(I_gt1+I_gt2+I_gt3+I_gt4)
        [Ir_pred1, It_pred1] = self.mainnet(I_obs1)
        [Ir_pred2, It_pred2] = self.mainnet(I_obs2)
        [Ir_pred3, It_pred3] = self.mainnet(I_obs3)
        [Ir_pred4, It_pred4] = self.mainnet(I_obs4)
        # Ir_refine = self.refinenet(Ir_pred1, Ir_pred2, Ir_pred3, Ir_pred4, I_obs1, I_obs2, I_obs3, I_obs4)
        It_refine = self.refinenet(It_pred1, It_pred2, It_pred3, It_pred4, I_obs1, I_obs2, I_obs3, I_obs4)
        # return [Ir_refine, It_refine, Ir_pred1, It_pred1, Ir_pred2, It_pred2, Ir_pred3, It_pred3, Ir_pred4, It_pred4]
        outputs = {}
        outputs['It_refine'] = It_refine
        It_pred = [It_pred1, It_pred2, It_pred3, It_pred4]
        outputs['It_pred'] = It_pred
        Ir_pred = [Ir_pred1, Ir_pred2, Ir_pred3, Ir_pred4]
        outputs['Ir_pred'] = Ir_pred
        outputs['loss'] = self.loss(It_refine, I_gt_main, It_pred=It_pred, It_gt_polar=I_gt)
        return outputs

    def inference(self, I_obs=[]):
        [I_obs1, I_obs2, I_obs3, I_obs4] = I_obs
        [Ir_pred1, It_pred1] = self.mainnet(I_obs1)
        [Ir_pred2, It_pred2] = self.mainnet(I_obs2)
        [Ir_pred3, It_pred3] = self.mainnet(I_obs3)
        [Ir_pred4, It_pred4] = self.mainnet(I_obs4)
        It_refine = self.refinenet(It_pred1, It_pred2, It_pred3, It_pred4, I_obs1, I_obs2, I_obs3, I_obs4)
        outputs = {}
        outputs['It_refine'] = It_refine
        outputs['It_pred'] = [It_pred1, It_pred2, It_pred3, It_pred4]
        outputs['Ir_pred'] = [Ir_pred1, Ir_pred2, Ir_pred3, Ir_pred4]
        return outputs

    def get_model_name(self, model_name: str, which_epoch: int):
        name = f'%s-%d' % (model_name, which_epoch)
        return name

    def save(self, opt, which_epoch: int):
        torch.save({
            'epoch': which_epoch,
            'mainnet': self.mainnet.state_dict(),
            'refinenet': self.refinenet.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, os.path.join(opt.checkpoints_dir,
                        self.get_model_name(model_name=opt.checkpoints_name, which_epoch=which_epoch)))

    def load(self, opt):
        # load networks
        if not self.isTrain or opt.continue_train:
            checkpt_path = os.path.join(opt.checkpoints_dir, opt.checkpoints_name)
            if os.path.exists(checkpt_path):
                checkpoint = torch.load(checkpt_path)
                self.mainnet.load_state_dict(checkpoint['mainnet'])
                self.refinenet.load_state_dict(checkpoint['refinenet'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.epoch = checkpoint['epoch']
            else:
                self.epoch = 0
                #loss = checkpoint['loss']

    def save_results(self, savepath: str, epoch: int, inputs: list, outputs: list):
        savepath_epoch = os.path.join(savepath, str(epoch))
        if not os.path.exists(savepath_epoch):
            os.mkdir(savepath_epoch)
        for i, data in enumerate(inputs):
            I = data['I']
            Igt = data['GT']

            for ii, imgi in enumerate(I):
                filename = f'intput_%d_%d_I_%d.png' % (epoch, i, ii)
                vutils.save_image(imgi.detach(),
                                  os.path.join(savepath_epoch, filename),
                                  normalize=False)

            for ii, imgi in enumerate(Igt):
                filename = f'intput_%d_%d_GT_%d.png' % (epoch, i, ii)
                vutils.save_image(imgi.detach(),
                                  os.path.join(savepath_epoch, filename),
                                  normalize=False)
        for i, data in enumerate(outputs):
            It_refine = data['It_refine']
            It_pred = data['It_pred']
            Ir_pred = data['Ir_pred']

            filename = f'output_%d_%d_refine.png' % (epoch, i)
            vutils.save_image(It_refine.detach(),
                              os.path.join(savepath_epoch, filename),
                              normalize=False)
            for ii, imgi in enumerate(It_pred):
                filename = f'output_%d_%d_It_pred_%d.png' % (epoch, i, ii)
                vutils.save_image(imgi.detach(),
                                  os.path.join(savepath_epoch, filename),
                                  normalize=False)

            for ii, imgi in enumerate(Ir_pred):
                filename = f'output_%d_%d_Ir_pred_%d.png' % (epoch, i, ii)
                vutils.save_image(imgi.detach(),
                                  os.path.join(savepath_epoch, filename),
                                  normalize=False)


class PolarEngine(BaseModel):
    '''
    In our simulation model, we directly simulate intensity change instead of Stokes vector
    '''
    default_setting = {'num_bounce': 10,
                       'weight_mirror': 0.3,
                       'refractive_index': 1.5}

    def __init__(self):
        super(PolarEngine, self).__init__()
        # R1, T1 as air to glass Muller's matrix
        # R2, T2 as glass to air Muller's matrix

        # we assume incoming light is unpolarized
        self.init_ref_para = 0.5
        self.init_ref_perp = 0.5
        self.init_trans_para = 0.5
        self.init_trans_perp = 0.5
        self.L2 = torch.nn.MSELoss()

    def initialize(self, opt, setting=default_setting):
        self.num_bounce = setting['num_bounce']
        self.weight_mirror = setting['weight_mirror']
        self.ref_index = setting['refractive_index']

        self.init_params()

    def calc_params(self, theta_in=0.1, thickness=10, weight_mirror=0.3, ref_index=1.5):
        # define Variable here
        allvars = {}
        allvars['theta_in'] = torch.tensor([theta_in]).cuda()
        allvars['thickness'] = torch.tensor([thickness]).cuda()
        allvars['weight_mirror'] = torch.tensor([weight_mirror]).cuda()
        allvars['ref_index'] = torch.tensor([ref_index]).cuda()
        allvars['theta_out'] = self.theta_in_out(allvars['theta_in'], allvars['ref_index'])
        # R T for glass to air: input angle theta_out, output angle theta_in
        [allvars['R_para'], allvars['R_perp'], allvars['T_para'], allvars['T_perp']] = self.calc_R_T(allvars['theta_out'], allvars['theta_in'])
        allvars['dx'] = self.delta_x(allvars['thickness'], allvars['theta_out'])
        return allvars

    def init_params(self, theta_in=0.1, thickness=9.9, weight_mirror=0.3, ref_index=1.5,
                    lr=0.0002, beta1=0.5):
        # define Variable here
        self.theta_in = torch.tensor([theta_in], requires_grad=True).cuda()
        self.thickness = torch.tensor([thickness], requires_grad=True).cuda()
        self.weight_mirror = torch.tensor([weight_mirror], requires_grad=True).cuda()
        self.ref_index = torch.tensor([ref_index], requires_grad=False).cuda()

        self.theta_out = self.theta_in_out(self.theta_in, self.ref_index)
        # R T for glass to air: input angle theta_out, output angle theta_in
        [self.R_para, self.R_perp, self.T_para, self.T_perp] = self.calc_R_T(self.theta_out, self.theta_in)
        self.dx = self.delta_x(self.thickness, self.theta_out)

        #self.optimizer = torch.optim.Adam([self.theta_in, self.thickness, self.weight_mirror],
        #                                  lr=lr, betas=(beta1, 0.999))
        self.optimizer = torch.optim.Adam([torch.tensor([0.0], requires_grad=True)], lr=lr, betas=(beta1, 0.999))

    def get_variables(self):
        vars = {}
        vars['theta_in'] = self.theta_in
        vars['theta_out'] = self.theta_out
        vars['thickness'] = self.thickness
        vars['weight_mirror'] = self.weight_mirror
        vars['R_para'] = self.R_para
        vars['R_perp'] = self.R_perp
        vars['T_para'] = self.T_para
        vars['T_perp'] = self.T_perp
        vars['dx'] = self.dx
        return vars

    def generate_multi_bounce_image(self, Ir, It,
                                    weight_mirror=0.3,
                                    theta_in=0.1,
                                    thickness=10,
                                    ref_index=1.5,
                                    ref_para=0.5, ref_perp=0.5,
                                    trans_para=0.5, trans_perp=0.5):
        theta_out = self.theta_in_out(theta_in, ref_index)
        # R T for glass to air: input angle theta_out, output angle theta_in
        [R_para, R_perp, T_para, T_perp] = self.calc_R_T(theta_out, theta_in)
        dx = self.delta_x(thickness, theta_out)

        msg = f'Tin:%.3f Tout:%.3f R:[%.3f, %.3f] T:[%.3f, %.3f] Dx:%.3f' % (theta_in, theta_out,
                                                                            R_para, R_perp, T_para, T_perp,
                                                                            dx)
        # unpolarized light
        [Ir_para, Ir_perp] = self.simulate_perp_para(Ir, ref_para, ref_perp)
        [It_para, It_perp] = self.simulate_perp_para(It, trans_para, trans_perp)

        # mirror reflection
        [Ir_para_mirror, Ir_para_polar] = self.mirror_ref(Ir_para, weight_mirror)
        [Ir_perp_mirror, Ir_perp_polar] = self.mirror_ref(Ir_perp, weight_mirror)

        # polarized incoming
        Ir_para_out = Ir_para_polar
        Ir_perp_out = Ir_perp_polar

        I_para = Ir_para_polar
        I_perp = Ir_perp_polar

        # [R_para, R_perp, T_para, T_perp] = self.calc_RT(self.theta_in, self.theta_out)
        # from air to glass1 (a->g1)
        [Ir_para, Ir_perp, It_para, It_perp] = self.ref_trans(I_para, I_perp,
                                                              R_para, R_perp,
                                                              T_para, T_perp)
        Ir_para_out = Ir_para_out + Ir_para
        Ir_perp_out = Ir_perp_out + Ir_perp

        for i in range(self.num_bounce):
            # g1 -> g2
            # from glass2 to air (g2->a)
            # [R_para, R_perp, T_para, T_perp] = self.calc_RT(self.theta_out, self.theta_in)
            [Ir_para, Ir_perp, It_para, It_perp] = self.ref_trans(I_para, I_perp,
                                                                  R_para, R_perp,
                                                                  T_para, T_perp)
            # g2 -> g1
            # from g1 to a
            # [R_para, R_perp, T_para, T_perp] = self.calc_RT(self.theta_out, self.theta_in)
            [Ir_para, Ir_perp, It_para, It_perp] = self.ref_trans(Ir_para, Ir_perp,
                                                                  R_para, R_perp,
                                                                  T_para, T_perp)

            # add shift
            Ir_para_shift = self.shift(Ir_para, dx)
            Ir_perp_shift = self.shift(Ir_perp, dx)
            It_para_shift = self.shift(It_para, dx)
            It_perp_shift = self.shift(It_perp, dx)

            I_para = Ir_para_shift
            I_perp = Ir_perp_shift

            Ir_para_out = Ir_para_out + It_para_shift
            Ir_perp_out = Ir_perp_out + It_perp_shift
        Ir_para_out = Ir_para_out + Ir_para_mirror
        Ir_perp_out = Ir_perp_out + Ir_perp_mirror
        I_para = It_para + Ir_para_out
        I_perp = It_perp + Ir_perp_out

        results = {}
        results['I_para'] = I_para
        results['I_perp'] = I_perp
        results['dx'] = dx
        return results

    def polarized_trace_reflected(self, I, weight_mirror, dx,
                                  R_para, R_perp, T_para, T_perp,
                                  para=0.5, perp=0.5):
        '''
        if side=True, return reflected light, otherwise return transmitted light
        '''

        # unpolarized light
        [I_para, I_perp] = self.simulate_perp_para(I, para, perp)

        # mirror reflection
        [I_para_mirror, I_para_polar] = self.mirror_ref(I_para, weight_mirror)
        [I_perp_mirror, I_perp_polar] = self.mirror_ref(I_perp, weight_mirror)

        # polarized mirror reflection
        I_para_out = I_para_mirror
        I_perp_out = I_perp_mirror

        # initial incident light
        I_para = I_para_polar
        I_perp = I_perp_polar

        # [R_para, R_perp, T_para, T_perp] = self.calc_RT(self.theta_in, self.theta_out)
        # from air to glass1 (a->g1)
        [Ir_para, Ir_perp, It_para, It_perp] = self.ref_trans(I_para, I_perp,
                                                              R_para, R_perp, T_para, T_perp)
        I_para_out = I_para_out + Ir_para
        I_perp_out = I_perp_out + Ir_perp

        for i in range(self.num_bounce):
            # g1 -> g2
            # from glass2 to air (g2->a)
            [Ir_para, Ir_perp, It_para, It_perp] = self.ref_trans(I_para, I_perp,
                                                                  R_para, R_perp,
                                                                  T_para, T_perp)
            # g2 -> g1
            # from g1 to a
            [Ir_para, Ir_perp, It_para, It_perp] = self.ref_trans(Ir_para, Ir_perp,
                                                                  R_para, R_perp,
                                                                  T_para, T_perp)

            # add shift
            Ir_para_shift = self.shift(Ir_para, dx)
            Ir_perp_shift = self.shift(Ir_perp, dx)
            It_para_shift = self.shift(It_para, dx)
            It_perp_shift = self.shift(It_perp, dx)

            I_para = Ir_para_shift
            I_perp = Ir_perp_shift

            I_para_out += It_para_shift
            I_perp_out += It_perp_shift
        return [I_para_out, I_perp_out]

    def polarized_trace_transmitted(self, I, weight_mirror, dx,
                                    R_para, R_perp, T_para, T_perp,
                                    para=0.5, perp=0.5):
        # unpolarized light
        [I_para, I_perp] = self.simulate_perp_para(I, para, perp)

        # mirror reflection
        [I_para_mirror, I_para_polar] = self.mirror_ref(I_para, weight_mirror)
        [I_perp_mirror, I_perp_polar] = self.mirror_ref(I_perp, weight_mirror)

        # polarized mirror reflection
        #I_para_out = I_para_mirror
        #I_perp_out = I_perp_mirror

        # initial incident light
        I_para = I_para_polar
        I_perp = I_perp_polar

        # [R_para, R_perp, T_para, T_perp] = self.calc_RT(self.theta_in, self.theta_out)
        # from air to glass1 (a->g1)
        [Ir_para, Ir_perp, It_para, It_perp] = self.ref_trans(I_para, I_perp,
                                                              R_para, R_perp, T_para, T_perp)
        # from glass2 to air
        [Ir_para, Ir_perp, It_para, It_perp] = self.ref_trans(It_para, It_perp,
                                                              R_para, R_perp, T_para, T_perp)
        I_para_out = It_para
        I_perp_out = It_perp
        I_para = Ir_para
        I_perp = Ir_perp

        for i in range(self.num_bounce):
            # g2 -> g1
            # from glass1 to air (g2->a)
            [Ir_para, Ir_perp, It_para, It_perp] = self.ref_trans(I_para, I_perp,
                                                                  R_para, R_perp,
                                                                  T_para, T_perp)
            # g2 -> g1
            # from g1 to a
            [Ir_para, Ir_perp, It_para, It_perp] = self.ref_trans(Ir_para, Ir_perp,
                                                                  R_para, R_perp,
                                                                  T_para, T_perp)

            # add shift
            Ir_para_shift = self.shift(Ir_para, dx)
            Ir_perp_shift = self.shift(Ir_perp, dx)
            It_para_shift = self.shift(It_para, dx)
            It_perp_shift = self.shift(It_perp, dx)

            I_para = Ir_para_shift
            I_perp = Ir_perp_shift

            I_para_out += It_para_shift
            I_perp_out += It_perp_shift
        return [I_para_out, I_perp_out]

    def polarized_trace_two(self, Ir, It, weight_mirror, dx, R_para, R_perp, T_para, T_perp):
        [Ir_para, Ir_perp] = self.polarized_trace_reflected(Ir,
                                                            weight_mirror=weight_mirror,
                                                            dx=dx,
                                                            R_para=R_para, R_perp=R_perp,
                                                            T_para=T_para, T_perp=T_perp)
        [It_para, It_perp] = self.polarized_trace_transmitted(It,
                                                              weight_mirror=weight_mirror,
                                                              dx=dx,
                                                              R_para=R_para, R_perp=R_perp,
                                                              T_para=T_para, T_perp=T_perp)
        I_para = Ir_para + It_para
        I_perp = Ir_perp + It_perp
        return [I_para, I_perp, Ir_para, Ir_perp, It_para, It_perp]

    def polarized_trace(self, Ir, It,
                        ref_para=0.5, ref_perp=0.5,
                        trans_para=0.5, trans_perp=0.5):
        # unpolarized light
        [Ir_para, Ir_perp] = self.simulate_perp_para(Ir, ref_para, ref_perp)
        [It_para, It_perp] = self.simulate_perp_para(It, trans_para, trans_perp)

        # mirror reflection
        [Ir_para_mirror, Ir_para_polar] = self.mirror_ref(Ir_para, self.weight_mirror)
        [Ir_perp_mirror, Ir_perp_polar] = self.mirror_ref(Ir_perp, self.weight_mirror)

        # polarized incoming
        Ir_para_out = Ir_para_polar
        Ir_perp_out = Ir_perp_polar

        I_para = Ir_para_polar
        I_perp = Ir_perp_polar

        #[R_para, R_perp, T_para, T_perp] = self.calc_RT(self.theta_in, self.theta_out)
        # from air to glass1 (a->g1)
        [Ir_para, Ir_perp, It_para, It_perp] = self.ref_trans(I_para, I_perp,
                                                              self.R_para, self.R_perp,
                                                              self.T_para, self.T_perp)
        Ir_para_out = Ir_para_out + Ir_para
        Ir_perp_out = Ir_perp_out + Ir_perp

        for i in range(self.num_bounce):
            # g1 -> g2
            # from glass2 to air (g2->a)
            #[R_para, R_perp, T_para, T_perp] = self.calc_RT(self.theta_out, self.theta_in)
            [Ir_para, Ir_perp, It_para, It_perp] = self.ref_trans(I_para, I_perp,
                                                                  self.R_para, self.R_perp,
                                                                  self.T_para, self.T_perp)
            # g2 -> g1
            # from g1 to a
            #[R_para, R_perp, T_para, T_perp] = self.calc_RT(self.theta_out, self.theta_in)
            [Ir_para, Ir_perp, It_para, It_perp] = self.ref_trans(Ir_para, Ir_perp,
                                                                  self.R_para, self.R_perp,
                                                                  self.T_para, self.T_perp)

            # add shift
            Ir_para_shift = self.shift(Ir_para, self.dx)
            Ir_perp_shift = self.shift(Ir_perp, self.dx)
            It_para_shift = self.shift(It_para, self.dx)
            It_perp_shift = self.shift(It_perp, self.dx)

            I_para = Ir_para_shift
            I_perp = Ir_perp_shift

            Ir_para_out = Ir_para_out + It_para_shift
            Ir_perp_out = Ir_perp_out + It_perp_shift
        Ir_para_out = Ir_para_out + Ir_para_mirror
        Ir_perp_out = Ir_perp_out + Ir_perp_mirror
        I_para = It_para + Ir_para_out
        I_perp = It_perp + Ir_perp_out
        return [I_para, I_perp]

    def loss(self, I_obs, I_out):
        return self.L2(I_obs, I_out)

    def forward(self, Ir, It, I_obs):
        [I_para, I_perp, Ir_para, Ir_perp, It_para, It_perp] = self.polarized_trace_two(Ir=Ir, It=It,
                                                                                        weight_mirror=self.weight_mirror,
                                                                                        dx=self.dx,
                                                                                        R_para=self.R_para,
                                                                                        R_perp=self.R_perp,
                                                                                        T_para=self.T_para,
                                                                                        T_perp=self.T_perp)
        I_out = I_para + I_perp
        new_loss = self.loss(I_obs=I_obs, I_out=I_out)
        results = {}
        results['I'] = I_out
        results['loss'] = new_loss
        results['params'] = self.get_variables()
        return results

    def theta_in_out(self, theta_in, ref_index):
        #theta_out = np.arcsin(np.sin(theta_in) / ref_index)
        theta_out = torch.asin(torch.sin(theta_in) / ref_index)
        return theta_out

    def calc_R_T(self, theta_in, theta_out):
        R_para = torch.pow(torch.tan(theta_in-theta_out), 2)/torch.pow(torch.tan(theta_in+theta_out), 2)
        R_perp = torch.pow(torch.sin(theta_in-theta_out), 2)/torch.pow(torch.sin(theta_in+theta_out), 2)
        T_para = 1 - R_para
        T_perp = 1 - R_perp
        return [R_para, R_perp, T_para, T_perp]

    def delta_x(self, thickness, theta_out):
        dx = 2*torch.tan(theta_out)*thickness
        return dx

    def shift(self, I, dx, ksz=81):
        filternp = np.zeros([3, 1, ksz, ksz])
        #filternp = np.zeros([3, 41, 41])
        dxint = int(torch.round(dx))
        cpt = int(ksz/2)
        filternp[:, :, cpt, cpt - dxint] = 1
        #filternp[:, 20, 20 - dxint] = 1
        filter = torch.Tensor(filternp).cuda()
        I_trans = torch.nn.functional.conv2d(I, filter, groups=3, padding=cpt)

        #I_trans = torch.nn.functional.conv2d(I, filter, padding=20)
        return I_trans

    # def calc_RT(self, theta_in, theta_out):
    #     R_para = np.power(np.tan(theta_in - theta_out), 2) / np.power(np.tan(theta_in + theta_out), 2)
    #     R_perp = np.power(np.sin(theta_in - theta_out), 2) / np.power(np.sin(theta_in + theta_out), 2)
    #     T_para = 1 - R_para
    #     T_perp = 1 - R_perp
    #     return [R_para, R_perp, T_para, T_perp]
    #
    # def calc_theta_out(self, theta_in):
    #     theta_out = np.arcsin(np.sin(theta_in)/self.n)
    #     return theta_out
    #
    # def shift(self, I):
    #     tx = np.int(2*self.d*np.cos(self.theta_in))
    #     [b, c, h, w] = I.shape
    #     filternp = np.zeros([3, 1, 41, 41])
    #     filternp[:, :, 20, 20 - tx] = 1
    #     filter = torch.Tensor(filternp).cuda()
    #     I_trans = torch.nn.functional.conv2d(I, filter, groups=3, padding=20)
    #     return I_trans

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

    # def translate_conv(self, I, d, theta=0):
    #     tx = d*torch.cos(theta)
    #     ty = d*torch.sin(theta)
    #     [b, c, h, w] = I.shape
    #     filternp = np.zeros([1, c, 40, 40])
    #     filternp[0, :, 20-ty, 20-tx] = 1
    #     filter = torch.Tensor(filternp)
    #     I_trans = torch.nn.functional.conv2d(I, filter, padding=20)

    # def polarized_trace(self):
    #     [Ir_para, Ir_perp] = self.simulate_perp_para(self.Ir, self.init_ref_para, self.init_ref_perp)
    #     [It_para, It_perp] = self.simulate_perp_para(self.It, self.init_trans_para, self.init_trans_perp)
    #
    #     # mirror reflection
    #     [Ir_para_mirror, Ir_para_polar] = self.mirror_ref(Ir_para, self.weight_mirror)
    #     [Ir_perp_mirror, Ir_perp_polar] = self.mirror_ref(Ir_perp, self.weight_mirror)
    #
    #     Ir_para_out = Ir_para_polar
    #     Ir_perp_out = Ir_perp_polar
    #
    #     I_para = Ir_para_polar
    #     I_perp = Ir_perp_polar
    #
    #     [R_para, R_perp, T_para, T_perp] = self.calc_RT(self.theta_in, self.theta_out)
    #     # from air to glass1 (a->g1)
    #     [Ir_para, Ir_perp, It_para, It_perp] = self.ref_trans(I_para, I_perp, R_para, R_perp, T_para, T_perp)
    #
    #     Ir_para_out = Ir_para_out + Ir_para
    #     Ir_perp_out = Ir_perp_out + Ir_perp
    #
    #     I_para = It_para
    #     I_perp = It_perp
    #     for i in range(self.num_bounce):
    #         # g1 -> g2
    #         # from glass2 to air (g2->a)
    #         [R_para, R_perp, T_para, T_perp] = self.calc_RT(self.theta_out, self.theta_in)
    #         [Ir_para, Ir_perp, It_para, It_perp] = self.ref_trans(I_para, I_perp, R_para, R_perp, T_para, T_perp)
    #         # g2->g1
    #         # from g1 to a
    #         [R_para, R_perp, T_para, T_perp] = self.calc_RT(self.theta_out, self.theta_in)
    #         [Ir_para, Ir_perp, It_para, It_perp] = self.ref_trans(Ir_para, Ir_perp, R_para, R_perp, T_para, T_perp)
    #
    #         ## add shift
    #         Ir_para_shift = self.shift(Ir_para)
    #         Ir_perp_shift = self.shift(Ir_perp)
    #         It_para_shift = self.shift(It_para)
    #         It_perp_shift = self.shift(It_perp)
    #
    #         I_para = Ir_para_shift
    #         I_perp = Ir_perp_shift
    #
    #         Ir_para_out = Ir_para_out + It_para_shift
    #         Ir_perp_out = Ir_perp_out + It_perp_shift
    #     return [Ir_para_out, Ir_perp_out]