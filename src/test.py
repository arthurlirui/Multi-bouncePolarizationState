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


def train():
    use_CUDA = True
    from dataloader import DataLoader
    from pprint import pprint
    if True:
        gtpath = '/home/lir0b/Code/perceptual-reflection-removal/synthetic'
        simupath = '/media/lir0b/关羽/Simulation_RT'
        ckptpath = '/home/lir0b/Code/TransparenceDetection/src/pid/ckpt'
    else:
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

def train_sepmodel():
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

    modelname = 'SepNet'
    dname = 'netD'
    dl = DataLoader(simupath, gtpath)
    # pprint(dl.simpair)
    ref_id = 49
    trans_id = 4877
    Epochs = 10
    t = 0
    save_epoch = True
    continue_train = True
    save_image = True
    device = torch.device("cuda:0" if use_CUDA else "cpu")

    sepnet = Sepmodel().to(device)
    sepnet.apply(weights_init)
    print(sepnet)
    optimizerG = optim.Adam(sepnet.parameters(), lr=0.0002, betas=(0.5, 0.999))

    netD = Discriminator()
    netD.to(device)
    netD.apply(weights_init)
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for i in range(Epochs):
        savepathi = os.path.join(ckptpath, str(i), modelname)
        if os.path.exists(savepathi) and continue_train:
            savepathi_1 = os.path.join(ckptpath, str(i + 1), modelname)
            if not os.path.exists(savepathi_1):
                sepnet.load_state_dict(torch.load(savepathi))
                netD.load_state_dict(torch.load(os.path.join(ckptpath, str(i), dname)))
            continue

        for ref_id, trans_id in dl.simpair:
            theta_pol_filelist = dl.get_theta_pol_filelist(ref_id, trans_id)
            ref_trans_pair = dl.get_ref_trans_pair(ref_id, trans_id)
            Ir = Image.open(ref_trans_pair[0])
            It = Image.open(ref_trans_pair[1])
            Ir = ToTensor()(Ir).unsqueeze(0).cuda()  # gt It
            It = ToTensor()(It).unsqueeze(0).cuda()  # gt Ir
            target = torch.cat((Ir, It), dim=1).cuda()
            thetaidx = 0
            for thetapair in theta_pol_filelist:
                imgpath0 = thetapair[0]
                imgpath45 = thetapair[1]
                imgpath90 = thetapair[2]
                imgpath135 = thetapair[3]

                img0 = Image.open(imgpath0)
                img45 = Image.open(imgpath45)
                img90 = Image.open(imgpath90)
                img135 = Image.open(imgpath135)

                imgin0 = ToTensor()(img0).unsqueeze(0).cuda()
                imgin45 = ToTensor()(img45).unsqueeze(0).cuda()
                imgin90 = ToTensor()(img90).unsqueeze(0).cuda()
                imgin135 = ToTensor()(img135).unsqueeze(0).cuda()
                #print(imgin0)
                #I_polar = [imgin0, imgin45, imgin90, imgin135]
                #inputs = torch.cat([imgin0, imgin45, imgin90, imgin135], dim=1)
                #[Ir_pred, It_pred] = sepnet(imgin0, imgin45, imgin90, imgin135)
                #print(Ir_pred)
                # print(len(imgout), imgout[0].shape, imgout[1].shape, imgout[2].shape)
                #[Ir_pred, It_pred] = torch.split(IrIt, (3, 3), dim=1)
                # [Dr, Sr, Pr_perp, Pr_para, Dt, St, Pt_perp, Pt_para] = torch.split(imgout[1], (3, 3, 3, 3, 3, 3, 3, 3), dim=1)
                #[Dr, Sr, Pr0, Pr45, Pr90, Pr135] = torch.split(DSPr, (3, 3, 3, 3, 3, 3), dim=1)
                #[Dt, St, Pt0, Pt45, Pt90, Pt135] = torch.split(DSPt, (3, 3, 3, 3, 3, 3), dim=1)
                # calc generator loss
                #sepnet.zero_grad()
                # Gloss = netG.loss(IrIt, target)
                # I_pol = [], IrIt=[], IrIr_pred=[], Ir_dsp=[], It_dsp=[]
                #Ir_dsp = [Dr, Sr, Pr0, Pr45, Pr90, Pr135]
                #It_dsp = [Dt, St, Pt0, Pt45, Pt90, Pt135]
                sepnet.zero_grad()
                for ii in range(1):
                    sepnet.zero_grad()
                    [Ir_pred, It_pred] = sepnet(imgin0, imgin45, imgin90, imgin135)
                    allloss = sepnet.loss([Ir, It], [Ir_pred, It_pred])
                    #print('Ir:(%.2f, %.2f) It:(%.2f, %.2f) Irp:(%.2f, %.2f) Itp:(%.2f, %.2f)'%(torch.max(Ir), torch.min(Ir),
                    #                                                                           torch.max(It), torch.min(It),
                    #                                                                           torch.max(Ir_pred), torch.min(Ir_pred),
                    #                                                                           torch.max(It_pred), torch.min(It_pred)))

                    [loss_sep, loss_vgg_sep, loss_exclu] = allloss
                    total_loss = loss_sep+loss_vgg_sep+loss_exclu
                    total_loss.backward(retain_graph=True)
                    optimizerG.step()
                    G_x = total_loss.item()
                    print('%d_%d_%d A:%.2f Lsep:%.6f Lvgg:%.6f Lexc:%.6f \n' % (i, ii, t, total_loss.item(),
                                                                                loss_sep.item(),
                                                                                loss_vgg_sep.item(),
                                                                                loss_exclu.item()))
                    if False:
                        Irpath_pred = os.path.join(ckptpath, str(i), '%d_%d_%d_r_pred.jpg'%(ref_id, thetaidx, ii))
                        vutils.save_image(Ir_pred.detach(), Irpath_pred, normalize=True)
                        Itpath_pred = os.path.join(ckptpath, str(i), '%d_%d_%d_t_pred.jpg'%(trans_id, thetaidx, ii))
                        vutils.save_image(It_pred.detach(), Itpath_pred, normalize=True)
                    thetaidx = thetaidx + 1

                # calc discriminator loss
                netD.zero_grad()
                isreal = netD(target)
                IrIt = torch.cat([Ir_pred, It_pred], dim=1)
                ispred = netD(IrIt)
                Dloss = netD.loss(isreal, ispred)
                Dloss.backward()
                D_x = Dloss.mean().item()
                optimizerD.step()

                print('%d L:%.2f D:%.2f' % (i, G_x, D_x))

                # [loss_sep, loss_vgg_sep, loss_edge, loss_rec, loss_flat_D, loss_smooth_S, loss_sparse_P] = allloss
                #[loss_sep, loss_vgg_sep, loss_execlu] = allloss

                # total_loss = torch.sum(torch.tensor(allloss, requires_grad=True))
                #total_loss = torch.sum(torch.tensor(allloss, requires_grad=True))

                #total_loss = torch.sum(torch.tensor(allloss))

                #G_x = total_loss.item()

                # print('A:%.2f Lsep:%.2f Lvgg:%.2f Le:%.2f Lrec: %.2f LD: %.2f LS:%.2f LP:%.2f '%(G_x, loss_sep.item(),
                #                                                                                 loss_vgg_sep.item(),
                #                                                                                 loss_edge.item(),
                #                                                                                 loss_rec.item(),
                #                                                                                 loss_flat_D.item(),
                #                                                                                 loss_smooth_S.item(),
                #                                                                                 loss_sparse_P.item()))

                #print('%d_%d A:%.2f Lsep:%.2f Lvgg:%.6f Lexe:%.2f\n' % (i, t, total_loss.item(), loss_sep.item(), loss_vgg_sep.item(), loss_execlu.item()))
                #print('%d_%d A:%.2f Lsep:%.2f \n' % (i, t, total_loss.item(), loss_sep.item()))
                t = t + 1
                #optimizerG.step()
                #print('%d L:%.2f D:%.2f' % (i, G_x, D_x))
                if t % 30 == 0:
                    if not os.path.exists(os.path.join(ckptpath, str(i))):
                        os.makedirs(os.path.join(ckptpath, str(i)))
                    Irpath = os.path.join(ckptpath, str(i), str(ref_id) + '_r.jpg')
                    vutils.save_image(Ir.detach(), Irpath, normalize=True)
                    Itpath = os.path.join(ckptpath, str(i), str(trans_id) + '_t.jpg')
                    vutils.save_image(It.detach(), Itpath, normalize=True)

                    Irpath_pred = os.path.join(ckptpath, str(i), str(ref_id) + '_r_pred.jpg')
                    vutils.save_image(Ir_pred.detach(), Irpath_pred, normalize=True)
                    Itpath_pred = os.path.join(ckptpath, str(i), str(trans_id) + '_t_pred.jpg')
                    vutils.save_image(It_pred.detach(), Itpath_pred, normalize=True)

        if save_epoch:
            savepathi = os.path.join(ckptpath, str(i))
            if not os.path.exists(savepathi):
                os.makedirs(savepathi)
            torch.save(sepnet.state_dict(), os.path.join(savepathi, modelname))
            torch.save(netD.state_dict(), os.path.join(savepathi, dname))

        if save_image:
            savepathi = os.path.join(ckptpath, str(i))
            if i % 1 == 0:
                idx = random.randint(0, len(dl.simpair)-1) % len(dl.simpair)
                ref_id = dl.simpair[idx][0]
                trans_id = dl.simpair[idx][1]
                theta_pol_filelist = dl.get_theta_pol_filelist(ref_id, trans_id)
                ref_trans_pair = dl.get_ref_trans_pair(ref_id, trans_id)
                if os.path.exists(ref_trans_pair[0]) and os.path.exists(ref_trans_pair[1]):
                    Ir_tmp = Image.open(ref_trans_pair[0])
                    It_tmp = Image.open(ref_trans_pair[1])
                    Irpath = os.path.join(ckptpath, str(i), str(ref_id) + '_r.jpg')
                    #vutils.save_image(Ir_tmp, Irpath, normalize=True)
                    Ir_tmp.save(Irpath)
                    Itpath = os.path.join(ckptpath, str(i), str(trans_id) + '_t.jpg')
                    #vutils.save_image(It_tmp, Itpath, normalize=True)
                    It_tmp.save(Itpath)
                    tt = 0
                    for thetapair in theta_pol_filelist:
                        img0 = Image.open(thetapair[0])
                        img45 = Image.open(thetapair[1])
                        img90 = Image.open(thetapair[2])
                        img135 = Image.open(thetapair[3])

                        imgin0 = ToTensor()(img0).unsqueeze(0).cuda()
                        imgin45 = ToTensor()(img45).unsqueeze(0).cuda()
                        imgin90 = ToTensor()(img90).unsqueeze(0).cuda()
                        imgin135 = ToTensor()(img135).unsqueeze(0).cuda()

                        [Ir_pred, It_pred] = sepnet(imgin0, imgin45, imgin90, imgin135)
                        #[Ir_pred, It_pred] = torch.split(imgout, (3, 3), dim=1)

                        Irpath_pred = os.path.join(ckptpath, str(i), str(ref_id) + '_theta' + str(tt) + '_r_pred.jpg')
                        vutils.save_image(Ir_pred.detach(), Irpath_pred, normalize=True)

                        Itpath_pred = os.path.join(ckptpath, str(i), str(trans_id) + '_theta' + str(tt) + '_t_pred.jpg')
                        vutils.save_image(It_pred.detach(), Itpath_pred, normalize=True)
                        tt = tt + 1


def change_filename():
    import glob
    gtpath = '/home/lir0b/data/polar/realdata/feb24/without_glass'
    imglist = glob.glob(os.path.join(gtpath, '*.png'))
    namelist = []


def downsample_image():
    rootpath = '/home/lir0b/Code/TransparenceDetection/draft_eccv/figure/visual_com/real/input'
    img = Image.open('girl_0.png')
    imgin = ToTensor()(img).unsqueeze(0).cuda()
    net = downnet()
    y = net(imgin)
    y2 = net(y)
    print(imgin.shape)
    print(y.shape)
    print(y2.shape)
    vutils.save_image(y.detach(), '1.png', normalize=False)
    vutils.save_image(y2.detach(), '2.png', normalize=False)


def calc_Ir():
    import cv2
    imgpath = '/home/lir0b/Code/TransparenceDetection/draft_eccv/figure/pptfigure/polar'
    img0 = cv2.imread(os.path.join(imgpath, 'LUCID_PHX050S-Q_190100163__20200225214858973_image0_0d.png'))
    img45 = cv2.imread(os.path.join(imgpath, 'LUCID_PHX050S-Q_190100163__20200225214858973_image0_45d.png'))
    img90 = cv2.imread(os.path.join(imgpath, 'LUCID_PHX050S-Q_190100163__20200225214858973_image0_90d.png'))
    img135 = cv2.imread(os.path.join(imgpath, 'LUCID_PHX050S-Q_190100163__20200225214858973_image0_135d.png'))
    img0.astype(float)
    I_t = cv2.imread('/home/lir0b/Code/TransparenceDetection/draft_eccv/figure/pptfigure/320200225214858973_refine.png')
    I_tot = 0.25*(img0.astype(float)+img45.astype(float)+img90.astype(float)+img135.astype(float))
    I_r = I_tot - I_t
    cv2.imwrite('I_r.png', I_r)
    cv2.imwrite('I_tot.png', I_tot)


def downsample_imagefolder():
    from torchvision.transforms.functional import center_crop
    if False:
        rootpath = '/media/lir0b/赵云/simeng/Mar2/1024/transmission_layer'
        outputpath = '/media/lir0b/赵云/simeng/Mar2/512_trans'
        import glob
        filelist = glob.glob(os.path.join(rootpath, '*.png'))
        for f in filelist:
            print(f)
            img = Image.open(f)
            img = center_crop(img, (1024, 1024))
            img512 = img.resize((512, 512))
            filename = f.split('/')[-1]
            img512.save(os.path.join(outputpath, filename))

    if True:
        inputpath = '/home/lir0b/data/polar/realdata/dataset/gt'
        outputpath = '/home/lir0b/data/polar/realdata/dataset/512/gt'
        import glob
        folderlist = glob.glob(os.path.join(inputpath, '*'))
        print(folderlist)
        for folder in folderlist:
            foldername = folder.split('/')[-1]
            os.makedirs(os.path.join(outputpath, foldername))
            filelist = glob.glob(os.path.join(inputpath, folder, '*.png'))
            for f in filelist:
                print(f)
                img = Image.open(f)
                img = center_crop(img, (1024, 1024))
                img512 = img.resize((512, 512))
                filename = f.split('/')[-1]
                img512.save(os.path.join(outputpath, foldername, filename))



    # for i in range(1, 7):
    #     img = Image.open(os.path.join(rootpath, '1024', str(i)+'.png'))
    #     img = center_crop(img, (1024, 1024))
    #     img512 = img.resize((512, 512))
    #     img512.save(os.path.join(rootpath, '512', str(i)+'.png'))
        #imgin = ToTensor()(img).unsqueeze(0).cuda()

        #net = downnet()
        #y = net(imgin)
        #y2 = net(y)
        #print(imgin.shape)
        #print(y.shape)
        #print(y2.shape)
        #vutils.save_image(y.detach(), os.path.join(rootpath, '512', str(i)+'.png'), normalize=False)
        #vutils.save_image(y2.detach(), os.path.join(rootpath, '256', str(i)+'.png'), normalize=False)


class downnet(nn.Module):
    def __init__(self):
        super(downnet, self).__init__()
        self.down2 = nn.AvgPool2d(kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.down2(x)


def crop_results():
    if False:
        imgpath = '/home/lir0b/Code/TransparenceDetection/draft_eccv/figure/ghost/2_crop.png'
        img0 = Image.open(imgpath)
        box = [1000, 200, 2000, 1200]
        img1 = img0.crop(box)
        #img2 = img1.resize([512, 512])
        img1.save('/home/lir0b/Code/TransparenceDetection/draft_eccv/figure/ghost/2_crop2.png')

    if True:
        imgpath = '/home/lir0b/Code/TransparenceDetection/draft/figure/example'
        filename = 'ex0.png'
        img0 = Image.open(os.path.join(imgpath, filename))
        box = [1000, 200, 2000, 1200]
        img1 = img0.crop(box)
        # img2 = img1.resize([512, 512])
        img1.save('/home/lir0b/Code/TransparenceDetection/draft_eccv/figure/ghost/2_crop2.png')


def test_stokes_vector():
    imgpath = '/home/lir0b/Code/TransparenceDetection/draft/figure/stokes'
    imgname_s0 = 'LUCID_PHX050S-Q_190100163__20200408230211095_image0_S0.png'
    imgname_s1 = 'LUCID_PHX050S-Q_190100163__20200408230211095_image0_S1.png'
    imgname_s2 = 'LUCID_PHX050S-Q_190100163__20200408230211095_image0_S2.png'
    imgname_s3 = 'LUCID_PHX050S-Q_190100163__20200408230211095_image0_S3.png'
    import
    imgs0 =


if __name__ == '__main__':
    #downsample_image()
    from torchvision.transforms.functional import center_crop, five_crop, to_tensor, to_pil_image

    #downsample_imagefolder()
    #calc_Ir()
    crop_results()
    if False:
        rootpath = '/home/lir0b/Code/TransparenceDetection/src/pid'
        img = Image.open(os.path.join(rootpath, 'girl_0.png'))
        #img = torchvision.transforms.CenterCrop(img, (1024, 1024))

        #transform = Compose([FiveCrop((1024, 1024)), FiveCrop(256, 256)])

    if False:
        img_center = center_crop(img, (1024, 1024))
        print(img_center.size)
        #pil_center = to_pil_image(img_center)
        #pil_center.show()
        img_2x2 = five_crop(img_center, (512, 512))
        img_2x2 = img_2x2[:-1]
        output_pil = []
        for i in img_2x2:
            img_list = five_crop(i, (256, 256))
            img_2x2_1 = img_list[:-1]
            print(img_2x2_1)
            tensor_2x2_1 = [to_tensor(t) for t in img_2x2_1]
            tl, tr, bl, br = tensor_2x2_1
            print(tl.shape)
            #print(tensor_2x2_1[0].shape)
            row1 = torch.cat([tl, tr], dim=2)
            print(row1.shape)
            row2 = torch.cat([bl, br], dim=2)
            tmp = torch.cat([row1, row2], dim=1)
            img_pil = to_pil_image(tmp)
            #img_pil.show()
            output_pil.append(tmp)
        tl, tr, bl, br = output_pil
        row1 = torch.cat([tl, tr], dim=2)
        row2 = torch.cat([bl, br], dim=2)
        output = torch.cat([row1, row2], dim=1)
        img_output = to_pil_image(output)
        img_output.show()
        print(img_output.size)


            #for ii in img_list:
                #print(ii.size)
