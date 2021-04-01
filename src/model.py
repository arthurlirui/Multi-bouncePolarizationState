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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        if False:
            vgg19 = torchvision.models.vgg19(pretrained=True)
            vgg19f = vgg19.features
            for param in vgg19.parameters():
                 param.requires_grad = False
            self.vgg19f1 = vgg19f[:5]
            self.vgg19f2 = vgg19f[5:10]
            self.vgg19f3 = vgg19f[10:19]
            self.vgg19f4 = vgg19f[19:28]
            self.vgg19f5 = vgg19f[28:]
        # dont change vgg19f[1-5]
        #self.trained = torch.nn.Sequential()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size,
        # stride=1, padding0, dilation=1, groups=1, bias=True,padding_mode='zeros')
        #vggnd = 1475  # 1472 vgg + 3*5 original
        vggnd = 3
        outnd = 64

        self.conv_input4 = nn.Conv2d(vggnd*4, outnd, (3, 3), dilation=1, padding=1)
        #self.conv_input1 = nn.Conv2d(vggnd, outnd, (3, 3), dilation=1, padding=1)
        self.conv2 = nn.Conv2d(outnd, outnd, (3, 3), dilation=2, padding=2)
        self.relu = torch.nn.ReLU()
        self.conv4 = nn.Conv2d(outnd, outnd, (3, 3), dilation=4, padding=4)
        self.conv8 = nn.Conv2d(outnd, outnd, (3, 3), dilation=8, padding=8)
        self.conv16 = nn.Conv2d(outnd, outnd, (3, 3), dilation=16, padding=16)
        self.conv32 = nn.Conv2d(outnd, outnd, (3, 3), dilation=32, padding=32)
        self.conv64 = nn.Conv2d(outnd, outnd, (3, 3), dilation=64, padding=64)
        self.conv_out = nn.Conv2d(outnd, 2*3, (3, 3), dilation=1, padding=1)
        self.sepnet = nn.Sequential(self.conv_input4, self.conv2, self.relu, self.conv4, self.relu,
                                    self.conv8, self.relu, self.conv16, self.relu, self.conv32, self.relu,
                                    self.conv64, self.relu, self.conv_out)

        self.conv_input5 = nn.Conv2d(vggnd * 5, outnd, (3, 3), dilation=1, padding=1)
        self.conv_out6 = nn.Conv2d(outnd, 6 * 3, (3, 3), dilation=1, padding=1)
        self.intnet = nn.Sequential(self.conv_input5, self.conv2, self.relu, self.conv4, self.relu,
                                    self.conv8, self.relu, self.conv16, self.relu, self.conv32, self.relu,
                                    self.conv64, self.relu, self.conv_out6)

    def forward(self, input0, input45=None, input90=None, input135=None):
        #if input.is_cuda and self.ngpu > 1:
            #output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        #else:
        # feat0 = self.build_vggfeat(input0)
        # feattot = torch.cat((feat0, input0), axis=1)
        # feat45 = self.build_vggfeat(input45)
        # feattot = torch.cat((feattot, feat45, input45), axis=1)
        # feat90 = self.build_vggfeat(input90)
        # feattot = torch.cat((feattot, feat90, input90), axis=1)
        # feat135 = self.build_vggfeat(input135)
        # feattot = torch.cat((feattot, feat135, input135), axis=1)
        # print(feattot.shape)
        input4 = torch.cat((input0, input45, input90, input135), dim=1)
        #input4.to(device)
        IrIt = self.sepnet(input4)
        [Ir, It] = torch.split(IrIt, (3, 3), dim=1)
        intinIr = torch.cat((Ir, input4), dim=1)
        intinIt = torch.cat((It, input4), dim=1)
        intoutIr = self.intnet(intinIr)
        intoutIt = self.intnet(intinIt)
        #[Ir, It] = torch.split(imgout, (3, 3), dim=1)
        #return [Ir, It]
        return [IrIt, intoutIr, intoutIt]

    def calc_vggfeat(self, inputs):
        h = inputs.shape[2]
        w = inputs.shape[3]
        op1 = nn.Upsample((h, w))  # upsampling to image size
        out1 = self.vgg19f1(inputs)
        #feat1 = op1(out1)
        #feat1 = torch.cat((feat1, input), axis=1)
        out2 = self.vgg19f2(out1)
        #feat2 = op1(out2)
        #feat2 = torch.cat((feat2, input), axis=1)
        out3 = self.vgg19f3(out2)
        #feat3 = op1(out3)
        #feat3 = torch.cat((feat3, input), axis=1)
        out4 = self.vgg19f4(out3)
        #feat4 = op1(out4)
        #feat4 = torch.cat((feat4, input), axis=1)
        out5 = self.vgg19f5(out4)
        #feat5 = op1(out5)
        #feat5 = torch.cat((feat5, input), axis=1)
        #allfeat = torch.cat((feat1, feat2, feat3, feat4, feat5), axis=1)
        return [out1, out2, out3, out4, out5]

    def loss(self, I_pol = [], IrIt=[], IrIr_pred=[], Ir_dsp=[], It_dsp=[], weights=[]):
        '''
        Lrt = L1(gt, pred)+Lvgg(gt, pred)+Le(grad)+Ld
        Ldsp = L1(dxD+dyD)+Lrec(I=DS+P)+L2(S)+L1(P)
        Lphy = Lrec(I=Ir+It)+Lrec(I0=DrSr+Pr0+DtSt+Pt0)+Lrec(I45=DrSr+Pr45+DtSt+Pt45)...
        '''
        l2 = nn.MSELoss()
        l1 = nn.L1Loss()
        I0, I45, I90, I135 = I_pol
        Ir_real, It_real = IrIt
        Ir_pred, It_pred = IrIr_pred
        Dr, Sr, Pr0, Pr45, Pr90, Pr135 = Ir_dsp
        Dt, St, Pt0, Pt45, Pt90, Pt135 = It_dsp

        loss_sep = self.loss_sep(Ir_real, It_real, Ir_pred, It_pred)
        #loss_vgg_sep = self.loss_vgg(Ir_real, It_real, Ir_pred, It_pred)

        #loss_edge = self.loss_exclusion(Ir_pred, It_pred)



        loss_rec = self.loss_rec(I_pol, Ir_dsp, It_dsp)

        loss_flat_D = self.loss_flat_D(Dr) + self.loss_flat_D(Dt)
        #loss_smooth_S = self.loss_smooth_S(Sr) + self.loss_smooth_S(St)
        #loss_sparse_P = self.loss_sparse_P(Pr0)+self.loss_sparse_P(Pr45)\
        #                +self.loss_sparse_P(Pr90)+self.loss_sparse_P(Pr135)\
        #                +self.loss_sparse_P(Pt0)+self.loss_sparse_P(Pt45)\
        #                +self.loss_sparse_P(Pt90)+self.loss_sparse_P(Pt135)
        #allloss = [loss_sep, loss_vgg_sep, loss_edge, loss_rec, loss_flat_D, loss_smooth_S, loss_sparse_P]
        allloss = [loss_sep, loss_rec, loss_flat_D]
        return allloss

    def loss_sep(self, Ir, It, Ir_pred, It_pred):
        l1 = nn.L1Loss()
        loss_r = l1(Ir_pred, Ir)
        loss_t = l1(It_pred, It)
        return loss_r + loss_r

    def loss_vgg(self, Ir, It, Ir_pred, It_pred):
        #l2 = nn.MSELoss()
        l1 = nn.L1Loss()
        vgg_r = self.calc_vggfeat(Ir)
        vgg_t = self.calc_vggfeat(It)
        vgg_pred_r = self.calc_vggfeat(Ir_pred)
        vgg_pred_t = self.calc_vggfeat(It_pred)
        vgg_real_r = self.calc_vggfeat(Ir)
        vgg_real_t = self.calc_vggfeat(It)
        vggloss_r = torch.sum(torch.tensor([l1(vgg_pred_r[id], vgg_real_r[id]) for id in range(len(vgg_pred_r))]))
        vggloss_t = torch.sum(torch.tensor([l1(vgg_pred_t[id], vgg_real_t[id]) for id in range(len(vgg_pred_t))]))
        return vggloss_r+vggloss_t

    def loss_rec(self, pol_img=[], dsp_r=[], dsp_t=[]):
        l1 = nn.L1Loss()
        I0, I45, I90, I135 = pol_img
        Dr, Sr, Pr0, Pr45, Pr90, Pr135 = dsp_r
        Dt, St, Pt0, Pt45, Pt90, Pt135 = dsp_t
        I0_rec = Dr * Sr + Pr0 + Dt * St + Pt0
        I45_rec = Dr * Sr + Pr45 + Dt * St + Pt45
        I90_rec = Dr * Sr + Pr90 + Dt * St + Pt90
        I135_rec = Dr * Sr + Pr135 + Dt * St + Pt135
        loss_0 = l1(I0_rec, I0)
        loss_45 = l1(I45_rec, I45)
        loss_90 = l1(I90_rec, I90)
        loss_135 = l1(I135_rec, I135)
        return loss_0+loss_45+loss_90+loss_135

    def loss_flat_D(self, D):
        #l1 = nn.L1Loss()
        kx = torch.tensor([1, 0, -1]).float()
        ky = torch.tensor([[1], [0], [-1]]).float()
        gkx = kx.repeat(3, 3, 1, 1).float().cuda()
        gky = ky.repeat(3, 3, 1, 1).float().cuda()
        gradx = torch.abs(F.conv2d(D, gkx, padding=(0, 1)))
        grady = torch.abs(F.conv2d(D, gky, padding=(1, 0)))
        grad = gradx+grady
        return torch.mean(torch.abs(grad))

    def loss_smooth_S(self, S):
        #l2 = nn.MSELoss()
        kx = torch.tensor([1, 0, -1]).float()
        ky = torch.tensor([[1], [0], [-1]]).float()
        gkx = kx.repeat(3, 3, 1, 1).float().cuda()
        gky = ky.repeat(3, 3, 1, 1).float().cuda()
        gradx = torch.abs(F.conv2d(S, gkx, padding=(0,1)))
        grady = torch.abs(F.conv2d(S, gky, padding=(1,0)))
        grad = gradx+grady
        return torch.mean(grad*grad)

    def loss_sparse_P(self, P):
        #l1 = nn.L1Loss()
        return torch.mean(torch.abs(P))

    def loss_exclusion(self, Ir, It):
        return self.calc_exclusion_loss(Ir, It, level=1)

    def calc_l1_loss(self, inputs, outputs):
        return nn.L1Loss(inputs, outputs)
        #return torch.mean(torch.abs(inputs-outputs))

    def calc_perceptual_loss(self, inputs, outputs, weights=[]):
        input_feat = self.calc_vggfeat(inputs)
        output_feat = self.calc_vggfeat(outputs)
        loss = []
        if len(weights) == 0:
            weights = torch.ones(len(input_feat))
        for i in range(len(input_feat)):
            loss.append(weights[i] * nn.MSELoss(input_feat[i], output_feat[i]))
        return torch.sum(torch.tensor(loss))

    def calc_exclusion_loss(self, inputs, outputs, level=1):
        kx = torch.tensor([[1., 0., -1.]], dtype=torch.float)
        ky = torch.tensor([[1.], [0.], [-1.]], dtype=torch.float)
        channels = inputs.shape[1]
        gkx = kx.repeat(channels, channels, 1, 1).float().cuda()
        gky = ky.repeat(channels, channels, 1, 1).float().cuda()
        loss = []
        avgp = nn.AvgPool2d((3, 3), padding=1)
        sigm = nn.Sigmoid()
        for l in range(level):
            input_gradx = torch.abs(F.conv2d(inputs, gkx, padding=(0, 1)))
            input_grady = torch.abs(F.conv2d(inputs, gky, padding=(1, 0)))
            #print(input_gradx.shape, input_grady.shape)
            input_grad = input_gradx+input_grady

            output_gradx = torch.abs(F.conv2d(outputs, gkx, padding=(0, 1)))
            output_grady = torch.abs(F.conv2d(outputs, gky, padding=(1, 0)))
            output_grad = output_gradx+output_grady
            loss.append(torch.mean(sigm(input_grad*output_grad)))
            #inputs = avgp(inputs)
            #outputs = avgp(outputs)
        return torch.sum(torch.tensor(loss))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        nc = 6
        ndf = 64
        self.ngpu = 1
        #self.loss = nn.BCELoss()

        def dloss(real_output, pred_output):
            return 0.5*torch.mean(-1*(torch.log(real_output+1e-10)) + torch.log(1-pred_output+1e-10))

        self.loss = dloss
        #self.loss = (tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))) * 0.5

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def test_single_train():
    use_CUDA = True
    if torch.cuda.is_available() and not use_CUDA:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    rootpath = '/media/lir0b/关羽/Simulation_RT'
    imgpath0 = os.path.join(rootpath, '49_4877_10', 'S0_theta_10_fi_0.jpg')
    imgpath45 = os.path.join(rootpath, '49_4877_10', 'S0_theta_10_fi_45.jpg')
    imgpath90 = os.path.join(rootpath, '49_4877_10', 'S0_theta_10_fi_90.jpg')
    imgpath135 = os.path.join(rootpath, '49_4877_10', 'S0_theta_10_fi_135.jpg')
    gtpath = '/home/lir0b/Code/perceptual-reflection-removal/synthetic/'
    imggtIr = os.path.join(gtpath, 'reflection_layer', '49.jpg')
    imggtIt = os.path.join(gtpath, 'transmission_layer', '4877.jpg')

    img0 = Image.open(imgpath0)
    img45 = Image.open(imgpath45)
    img90 = Image.open(imgpath90)
    img135 = Image.open(imgpath135)
    Ir = Image.open(imggtIr)
    It = Image.open(imggtIt)

    print(img0.size)
    imgin0 = ToTensor()(img0).unsqueeze(0)
    imgin45 = ToTensor()(img45).unsqueeze(0)
    imgin90 = ToTensor()(img90).unsqueeze(0)
    imgin135 = ToTensor()(img135).unsqueeze(0)

    device = torch.device("cuda:0" if use_CUDA else "cpu")

    netG = Generator()
    netG.to(device)
    netG.apply(weights_init)

    netD = Discriminator()
    netD.to(device)
    netD.apply(weights_init)

    imgout = netG(imgin0.cuda(), imgin45.cuda(), imgin90.cuda(), imgin135.cuda())

    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    Ir = ToTensor()(Ir).unsqueeze(0)
    It = ToTensor()(It).unsqueeze(0)
    target = torch.cat((Ir, It), dim=1).cuda()
    losst = netG.loss(imgout, target)
    losst.backward()
    print(losst)
    optimizerG.step()

    imgout = imgout.cpu()
    [Ir, It] = torch.split(imgout, (3, 3), dim=1)
    print(Ir.shape, It.shape)
    imout1 = ToPILImage()(Ir.squeeze(0))
    imout2 = ToPILImage()(It.squeeze(0))
    imout1.show()
    imout2.show()


def train_epoch():
    pass


def train():
    use_CUDA = True
    from dataloader import DataLoader
    from pprint import pprint
    gtpath = '/home/lir0b/Code/perceptual-reflection-removal/synthetic'
    simupath = '/media/lir0b/关羽/Simulation_RT'
    ckptpath = '/home/lir0b/Code/TransparenceDetection/src/pid/ckpt'
    if not os.path.exists(gtpath) or not os.path.exists(simupath) or not os.path.exists(ckptpath):
        gtpath = '/ibex/scratch/lir0b/data/synthetic'
        simupath = '/ibex/scratch/lir0b/data/Simulation_RT'
        ckptpath = '/ibex/scratch/lir0b/TransparenceDetection/src/pid/ckpt'

    modelname = 'netG'
    dname = 'netD'
    dl = DataLoader(simupath, gtpath)
    #pprint(dl.simpair)
    ref_id = 49
    trans_id = 4877
    Epochs = 10
    save_epoch = True
    continue_train = False
    save_image = True
    device = torch.device("cuda:0" if use_CUDA else "cpu")

    netG = Generator()
    netG.to(device)
    netG.apply(weights_init)
    print(netG)
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    netD = Discriminator()
    netD.to(device)
    netD.apply(weights_init)
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for i in range(Epochs):
        savepathi = os.path.join(ckptpath, str(i), modelname)
        if os.path.exists(savepathi) and continue_train:
            savepathi_1 = os.path.join(ckptpath, str(i+1), modelname)
            if not os.path.exists(savepathi_1):
                netG.load_state_dict(torch.load(savepathi))
                netD.load_state_dict(torch.load(os.path.join(ckptpath, str(i), dname)))
            continue

        for ref_id, trans_id in dl.simpair[:500]:
            theta_pol_filelist = dl.get_theta_pol_filelist(ref_id, trans_id)
            ref_trans_pair = dl.get_ref_trans_pair(ref_id, trans_id)
            Ir = Image.open(ref_trans_pair[0])
            It = Image.open(ref_trans_pair[1])
            Ir = ToTensor()(Ir).unsqueeze(0)  # gt It
            It = ToTensor()(It).unsqueeze(0)  # gt Ir
            target = torch.cat((Ir, It), dim=1).cuda()

            for thetapair in theta_pol_filelist:
                imgpath0 = thetapair[0]
                imgpath45 = thetapair[1]
                imgpath90 = thetapair[2]
                imgpath135 = thetapair[3]
                #print(imgpath0)
                img0 = Image.open(imgpath0)
                img45 = Image.open(imgpath45)
                img90 = Image.open(imgpath90)
                img135 = Image.open(imgpath135)

                imgin0 = ToTensor()(img0).unsqueeze(0).cuda()
                imgin45 = ToTensor()(img45).unsqueeze(0).cuda()
                imgin90 = ToTensor()(img90).unsqueeze(0).cuda()
                imgin135 = ToTensor()(img135).unsqueeze(0).cuda()
                I_polar=[imgin0, imgin45, imgin90, imgin135]

                [IrIt, DSPr, DSPt] = netG(imgin0, imgin45, imgin90, imgin135)
                #print(len(imgout), imgout[0].shape, imgout[1].shape, imgout[2].shape)
                [Ir_pred, It_pred] = torch.split(IrIt, (3, 3), dim=1)
                #[Dr, Sr, Pr_perp, Pr_para, Dt, St, Pt_perp, Pt_para] = torch.split(imgout[1], (3, 3, 3, 3, 3, 3, 3, 3), dim=1)
                [Dr, Sr, Pr0, Pr45, Pr90, Pr135] = torch.split(DSPr, (3, 3, 3, 3, 3, 3), dim=1)
                [Dt, St, Pt0, Pt45, Pt90, Pt135] = torch.split(DSPt, (3, 3, 3, 3, 3, 3), dim=1)

                # calc generator loss
                netG.zero_grad()
                #Gloss = netG.loss(IrIt, target)
                # I_pol = [], IrIt=[], IrIr_pred=[], Ir_dsp=[], It_dsp=[]
                Ir_dsp = [Dr, Sr, Pr0, Pr45, Pr90, Pr135]
                It_dsp = [Dt, St, Pt0, Pt45, Pt90, Pt135]
                allloss = netG.loss(I_polar, [Ir.cuda(), It.cuda()], [Ir_pred, It_pred], Ir_dsp, It_dsp)
                #[loss_sep, loss_vgg_sep, loss_edge, loss_rec, loss_flat_D, loss_smooth_S, loss_sparse_P] = allloss
                [loss_sep, loss_rec, loss_flat_D] = allloss
                #total_loss = torch.sum(torch.tensor(allloss, requires_grad=True))
                total_loss = torch.sum(torch.tensor(allloss, requires_grad=True))
                total_loss.backward(retain_graph=True)
                G_x = total_loss.item()


                #print('A:%.2f Lsep:%.2f Lvgg:%.2f Le:%.2f Lrec: %.2f LD: %.2f LS:%.2f LP:%.2f '%(G_x, loss_sep.item(),
                #                                                                                 loss_vgg_sep.item(),
                #                                                                                 loss_edge.item(),
                #                                                                                 loss_rec.item(),
                #                                                                                 loss_flat_D.item(),
                #                                                                                 loss_smooth_S.item(),
                #                                                                                 loss_sparse_P.item()))

                print('A:%.2f Lsep:%.2f Lrec:%.2f LD: %.2f \n'%(G_x, loss_sep.item(), loss_rec.item(), loss_flat_D.item()))

                optimizerG.step()

                # calc discriminator loss
                netD.zero_grad()
                isreal = netD(target)
                #isrealIt = netD(It)
                ispred = netD(IrIt)
                #ispredIt = netD(It_pred)
                Dloss = netD.loss(isreal, ispred)
                Dloss.backward()
                D_x = Dloss.mean().item()
                optimizerD.step()

                print('%d L:%.2f D:%.2f' % (i, G_x, D_x))

        if save_epoch:
            savepathi = os.path.join(ckptpath, str(i))
            if not os.path.exists(savepathi):
                os.makedirs(savepathi)
            torch.save(netG.state_dict(), os.path.join(savepathi, modelname))
            torch.save(netD.state_dict(), os.path.join(savepathi, dname))

        if save_image:
            savepathi = os.path.join(ckptpath, str(i))
            if i % 1 == 0:
                idx = random.randint(1, 777)%len(dl.simpair)
                ref_id = dl.simpair[idx][0]
                trans_id = dl.simpair[idx][1]
                theta_pol_filelist = dl.get_theta_pol_filelist(ref_id, trans_id)
                ref_trans_pair = dl.get_ref_trans_pair(ref_id, trans_id)
                Ir = Image.open(ref_trans_pair[0])
                It = Image.open(ref_trans_pair[1])
                Irpath = os.path.join(ckptpath, str(i), str(ref_id)+'_r.jpg')
                #vutils.save_image(Ir, Irpath, normalize=True)
                Ir.save(Irpath)
                Itpath = os.path.join(ckptpath, str(i), str(trans_id) + '_t.jpg')
                #vutils.save_image(It, Itpath, normalize=True)
                It.save(Itpath)
                tt = 0
                for thetapair in theta_pol_filelist:
                    img0 = Image.open(thetapair[0])
                    img45 = Image.open(thetapair[1])
                    img90 = Image.open(thetapair[2])
                    img135 = Image.open(thetapair[3])

                    imgin0 = ToTensor()(img0).unsqueeze(0)
                    imgin45 = ToTensor()(img45).unsqueeze(0)
                    imgin90 = ToTensor()(img90).unsqueeze(0)
                    imgin135 = ToTensor()(img135).unsqueeze(0)

                    imgout = netG(imgin0.cuda(), imgin45.cuda(), imgin90.cuda(), imgin135.cuda())
                    [Ir_pred, It_pred] = torch.split(imgout[0], (3, 3), dim=1)

                    Irpath_pred = os.path.join(ckptpath, str(i), str(ref_id) + '_theta' + str(tt) + '_r_pred.jpg')
                    vutils.save_image(Ir_pred.detach(), Irpath_pred, normalize=True)

                    Itpath_pred = os.path.join(ckptpath, str(i), str(trans_id) + '_theta' + str(tt) + '_t_pred.jpg')
                    vutils.save_image(It_pred.detach(), Itpath_pred, normalize=True)
                    tt = tt + 1


if __name__ == '__main__':
    train()
