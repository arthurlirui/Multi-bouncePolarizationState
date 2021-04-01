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
from pidmodel import *
import glob
from PIL import ImageEnhance
import time
import numpy as np
from torchvision.transforms.functional import center_crop, five_crop, to_tensor, to_pil_image


def train_sepmodel():
    use_CUDA = True
    from dataloader import DataLoader
    from pprint import pprint
    gtpath = '/home/qius0a/data/dataset/synthetic'
    #simupath = '/media/lir0b/关羽/Simulation_RT'
    simupath = '/home/qius0a/data/dataset/Simulation_HighIntensity'
    ckptpath = '/home/qius0a/Code/TransparenceDetection/ckpt'
    if not os.path.exists(gtpath) or not os.path.exists(simupath) or not os.path.exists(ckptpath):
        gtpath = '/ibex/scratch/qius0a/dataset/synthetic'
        #simupath = '/ibex/scratch/lir0b/data/Simulation_RT'
        simupath = '/ibex/scratch/qius0a/dataset/Simulation_HighIntensity'
        ckptpath = '/ibex/scratch/qius0a/pid/ckpt'

    modelname = 'SepNet'
    psfname = 'PSFNet'
    dname = 'netD'
    dl = DataLoader(simupath, gtpath)
    Epochs = 100
    t = 0
    numsample = 500
    rsz = (320, 320)
    save_epoch = True
    continue_train = False
    save_image = True
    device = torch.device("cuda:0" if use_CUDA else "cpu")

    sepnet = Sepmodel().to(device)
    sepnet.apply(weights_init)
    #psfnet = PSFmodel().to(device)
    #psfnet.apply(weights_init)

    optimizerG = optim.Adam(sepnet.parameters(), lr=0.0002, betas=(0.5, 0.999))
    #optimizerPSF = optim.Adam(sepnet.psfnet.parameters(), lr=0.0002, betas=(0.5, 0.999))
    #optimizerPSF = optim.Adam(psfnet.parameters(), lr=0.0002, betas=(0.5, 0.999))

    print(sepnet.parameters())
    #print(psfnet.parameters())

    refD = RefDiscriminator()
    transD = TransDiscriminator()
    netD = Discriminator()
    refD.to(device)
    transD.to(device)
    netD.to(device)
    refD.apply(weights_init)
    transD.apply(weights_init)

    optimrefD = optim.Adam(refD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimtransD = optim.Adam(transD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    #optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    weights = [1.0, 5000, 5000, 3, 1]
    #weights = [1.33, 0, 0, 0, 0]
    enratio = 1
    if continue_train:
        savefoldername = '1.00_1000.00_10.00_1579512652'
        ckptpath = os.path.join(ckptpath, savefoldername)
    else:
        ckptpath = os.path.join(ckptpath, '%.2f_%.2f_%.2f_%d' % (weights[0], weights[1], weights[2], int(time.time())))
    if not os.path.exists(ckptpath):
        os.makedirs(ckptpath)
    for i in range(Epochs):
        savepathi = os.path.join(ckptpath, str(i), modelname)
        if os.path.exists(savepathi) and continue_train:
            savepathi_1 = os.path.join(ckptpath, str(i + 1), modelname)
            if not os.path.exists(savepathi_1):
                sepnet.load_state_dict(torch.load(os.path.join(ckptpath, str(i), modelname)))
                #psfnet.load_state_dict(torch.load(os.path.join(ckptpath, str(i), psfname)))
                refD.load_state_dict(torch.load(os.path.join(ckptpath, str(i), 'refD')))
                transD.load_state_dict(torch.load(os.path.join(ckptpath, str(i), 'transD')))
            continue
        for idx in range(numsample):
            #idx = random.randint(0, numsample) % len(dl.simpair)
            #idx = random.randint(0, numsample) % len(dl.simpair)
            ref_id, trans_id = dl.simpair[idx]
            theta_pol_filelist = dl.get_theta_pol_filelist(ref_id, trans_id)
            ref_trans_pair = dl.get_ref_trans_pair(ref_id, trans_id)
            Ir = Image.open(ref_trans_pair[0])
            Ir = Ir.resize(rsz)
            It = Image.open(ref_trans_pair[1])
            It = It.resize(rsz)
            Ir = ToTensor()(Ir).unsqueeze(0).cuda()  # gt It
            It = ToTensor()(It).unsqueeze(0).cuda()  # gt Ir
            target = torch.cat((Ir, It), dim=1).cuda()
            theta_idx = 0
            for thetapair in theta_pol_filelist:
                imgpath0 = thetapair[0]
                imgpath45 = thetapair[1]
                imgpath90 = thetapair[2]
                imgpath135 = thetapair[3]

                if os.path.exists(imgpath0) and os.path.exists(imgpath45) \
                        and os.path.exists(imgpath90) and os.path.exists(imgpath135):
                    img0 = Image.open(imgpath0)
                    img0 = img0.resize(rsz)
                    enhb = ImageEnhance.Brightness(img0)
                    img0 = enhb.enhance(enratio)

                    img45 = Image.open(imgpath45)
                    img45 = img45.resize(rsz)
                    enhb = ImageEnhance.Brightness(img45)
                    img45 = enhb.enhance(enratio)

                    img90 = Image.open(imgpath90)
                    img90 = img90.resize(rsz)
                    enhb = ImageEnhance.Brightness(img90)
                    img90 = enhb.enhance(enratio)

                    img135 = Image.open(imgpath135)
                    img135 = img135.resize(rsz)
                    enhb = ImageEnhance.Brightness(img135)
                    img135 = enhb.enhance(enratio)
                else:
                    continue

                imgin0 = ToTensor()(img0).unsqueeze(0).cuda()
                imgin45 = ToTensor()(img45).unsqueeze(0).cuda()
                imgin90 = ToTensor()(img90).unsqueeze(0).cuda()
                imgin135 = ToTensor()(img135).unsqueeze(0).cuda()
                Itot = 0.25*(imgin0+imgin45+imgin90+imgin135)

                sepnet.zero_grad()
                #for ii in range(1):
                #sepnet.zero_grad()
                [Ir_pred, It_pred, I_obs] = sepnet(imgin0, imgin45, imgin90, imgin135)
                allloss = sepnet.loss([Ir, It], [Ir_pred, It_pred], Itot, I_obs, weights)
                [loss_sep, loss_vgg_sep, loss_exclu, loss_backward] = allloss
                ii = 0
                total_loss = loss_sep + loss_vgg_sep + loss_exclu + loss_backward
                if t % 100 == 0:
                    print('%d_%d_%d A:%.2f Lsep:%.6f Lvgg:%.6f Lexc:%.6f Lb:%.6f \n' % (i, ii, t, total_loss.item(),
                                                                                loss_sep.item(),
                                                                                loss_vgg_sep.item(),
                                                                                loss_exclu.item(),
                                                                                loss_backward.item()))

                #psfnet.zero_grad()
                #for ii in range(5):
                #    psfnet.zero_grad()
                #    [Ir_back, It_back] = psfnet(Ir_pred, It_pred)
                #    psfloss = psfnet.loss(Ir_back, It_back, Itot)
                #    psfloss.backward(retain_graph=True)
                #    optimizerPSF.step()

                #print('%d_%d_%d Lpsf:%.2f \n' % (i, ii, t, psfloss.item()))
                #total_loss = loss_sep + loss_vgg_sep + loss_exclu + loss_backward
                total_loss.backward(retain_graph=True)
                optimizerG.step()
                G_x = total_loss.item()

                #thetaidx = thetaidx + 1

                # calc discriminator loss
                refD.zero_grad()
                isreal = netD(target)
                IrIt = torch.cat([Ir_pred, It_pred], dim=1)
                ispred = netD(IrIt)
                Dloss = netD.loss(isreal, ispred)
                Dloss.backward()
                D_x = Dloss.mean().item()
                optimizerD.step()
                if t % 100 == 0:
                    print('%d L:%.2f D:%.2f' % (i, G_x, D_x))
                t = t + 1
                theta_idx = theta_idx + 1
                #optimizerG.step()
                #print('%d L:%.2f D:%.2f' % (i, G_x, D_x))
                if save_image and idx % 30 == 0:
                    if not os.path.exists(os.path.join(ckptpath, str(i))):
                        os.makedirs(os.path.join(ckptpath, str(i)))
                    Irpath = os.path.join(ckptpath, str(i), str(ref_id) + '_' + str(theta_idx) + '_r.jpg')
                    vutils.save_image(Ir.detach(), Irpath, normalize=True)
                    Itpath = os.path.join(ckptpath, str(i), str(trans_id) + '_' + str(theta_idx) + '_t.jpg')
                    vutils.save_image(It.detach(), Itpath, normalize=True)

                    Irpath_pred = os.path.join(ckptpath, str(i), str(ref_id) + '_' + str(theta_idx) + '_r_pred.jpg')
                    vutils.save_image(Ir_pred.detach(), Irpath_pred, normalize=True)
                    Itpath_pred = os.path.join(ckptpath, str(i), str(trans_id) + '_' + str(theta_idx) + '_t_pred.jpg')
                    vutils.save_image(It_pred.detach(), Itpath_pred, normalize=True)

                    I_obs_path = os.path.join(ckptpath, str(i), str(ref_id)+'_'+str(trans_id) + '_' + str(theta_idx) + '_obs.jpg')
                    vutils.save_image(I_obs.detach(), I_obs_path, normalize=True)
                    #Itpath_back = os.path.join(ckptpath, str(i), str(trans_id) + '_t_back.jpg')
                    #vutils.save_image(It_back.detach(), Itpath_back, normalize=True)

        if save_epoch:
            savepathi = os.path.join(ckptpath, str(i))
            if not os.path.exists(savepathi):
                os.makedirs(savepathi)
            torch.save(sepnet.state_dict(), os.path.join(savepathi, modelname))
            torch.save(refD.state_dict(), os.path.join(savepathi, 'refD'))
            torch.save(refD.state_dict(), os.path.join(savepathi, 'transD'))
            #torch.save(psfnet.state_dict(), os.path.join(savepathi, psfname))

        # if save_image:
        #     savepathi = os.path.join(ckptpath, str(i))
        #     if i % 1 == 0:
        #         idx = random.randint(0, len(dl.simpair)-1) % len(dl.simpair)
        #         ref_id = dl.simpair[idx][0]
        #         trans_id = dl.simpair[idx][1]
        #         theta_pol_filelist = dl.get_theta_pol_filelist(ref_id, trans_id)
        #         ref_trans_pair = dl.get_ref_trans_pair(ref_id, trans_id)
        #         if os.path.exists(ref_trans_pair[0]) and os.path.exists(ref_trans_pair[1]):
        #             Ir_tmp = Image.open(ref_trans_pair[0])
        #             It_tmp = Image.open(ref_trans_pair[1])
        #             Irpath = os.path.join(ckptpath, str(i), str(ref_id) + '_r.jpg')
        #             #vutils.save_image(Ir_tmp, Irpath, normalize=True)
        #             Ir_tmp.save(Irpath)
        #             Itpath = os.path.join(ckptpath, str(i), str(trans_id) + '_t.jpg')
        #             #vutils.save_image(It_tmp, Itpath, normalize=True)
        #             It_tmp.save(Itpath)
        #             tt = 0
        #             for thetapair in theta_pol_filelist:
        #                 if os.path.exists(thetapair[0]) and os.path.exists(thetapair[1]) \
        #                         and os.path.exists(thetapair[2]) and os.path.exists(thetapair[3]):
        #                     img0 = Image.open(thetapair[0])
        #                     img45 = Image.open(thetapair[1])
        #                     img90 = Image.open(thetapair[2])
        #                     img135 = Image.open(thetapair[3])
        #                 else:
        #                     continue
        #
        #                 imgin0 = ToTensor()(img0).unsqueeze(0).cuda()
        #                 imgin45 = ToTensor()(img45).unsqueeze(0).cuda()
        #                 imgin90 = ToTensor()(img90).unsqueeze(0).cuda()
        #                 imgin135 = ToTensor()(img135).unsqueeze(0).cuda()
        #
        #                 [Ir_pred, It_pred] = sepnet(imgin0, imgin45, imgin90, imgin135)
        #                 #[Ir_pred, It_pred] = torch.split(imgout, (3, 3), dim=1)
        #
        #                 Irpath_pred = os.path.join(ckptpath, str(i), str(ref_id) + '_theta' + str(tt) + '_r_pred.jpg')
        #                 vutils.save_image(Ir_pred.detach(), Irpath_pred, normalize=True)
        #
        #                 Itpath_pred = os.path.join(ckptpath, str(i), str(trans_id) + '_theta' + str(tt) + '_t_pred.jpg')
        #                 vutils.save_image(It_pred.detach(), Itpath_pred, normalize=True)
        #                 tt = tt + 1


def preprocess_image(imgpath, sz=(512, 512), box=(0, 0, 512, 512), data_argument=True):
    if os.path.exists(imgpath):
        if data_argument:
            img = Image.open(imgpath)
            [h, w] = img.size
            I_crop = img.crop(box)
            I_crop = I_crop.resize(sz)
            return I_crop
        else:
            img = Image.open(imgpath)
            if True:
                img = center_crop(img, (1024, 1024))
            I_center = img.resize((512, 512))
            [h, w] = I_center.size
            I_center = center_crop(I_center, (h+64, w+64))


            #I_center = center_crop(img, (min([1024, h]), min([1024, w])))
            #I_center = center_crop(img, sz)
            #I_center = I_center.resize()
            return I_center
    else:
        print('None exist files: %s' % imgpath)
        return None


def preprocess_image_realdata_grid4x4(imgpath):
    if os.path.exists(imgpath):
        img = Image.open(imgpath)
        img_center = center_crop(img, (1024, 1024))
        img_grid = five_crop(img_center, (512, 512))
        [tl, tr, bl, br, center] = img_grid

        tl_grid = five_crop(tl, (256, 256))
        tr_grid = five_crop(tr, (256, 256))
        bl_grid = five_crop(bl, (256, 256))
        br_grid = five_crop(br, (256, 256))
        cen_grid = five_crop(center, (256, 256))

        return [tl_grid, tr_grid, bl_grid, br_grid, cen_grid]
    else:
        print('None exist files: %s' % imgpath)
        return None


def to_CUDA(img):
    return ToTensor()(img).unsqueeze(0).cuda()


def define_path():
    gtpath = '/home/lir0b/Code/perceptual-reflection-removal/synthetic'
    #simupath = '/home/lir0b/Code/TransparenceDetection/src/pid/exp/synthetic_multi'
    simupath = '/home/lir0b/Code/TransparenceDetection/src/pid/exp/paperdata/partial'
    ckptpath = '/home/lir0b/Code/TransparenceDetection/src/pid/ckpt'
    if not os.path.exists(gtpath) and not os.path.exists(simupath) and not os.path.exists(ckptpath):
        gtpath = '/ibex/scratch/lir0b/data/synthetic'
        simupath = '/ibex/scratch/lir0b/data/paperdata/strong_ref'
        ckptpath = '/ibex/scratch/lir0b/TransparenceDetection/src/pid/ckpt'
    if not os.path.exists(gtpath) and not os.path.exists(simupath) and not os.path.exists(ckptpath):
        gtpath = '/ibex/scratch/lir0b/data/synthetic'
        simupath = '/ibex/scratch/lir0b/data/synthetic_multi'
        ckptpath = '/ibex/scratch/lir0b/TransparenceDetection/src/pid/ckpt'
    if not os.path.exists(gtpath) and not os.path.exists(simupath) and not os.path.exists(ckptpath):
        gtpath = '/ibex/scratch/qius0a/data/synthetic'
        simupath = '/ibex/scratch/qius0a/data/synthetic_multi'
        ckptpath = '/ibex/scratch/qius0a/TransparenceDetection/src/pid/ckpt'
    if not os.path.exists(gtpath) and not os.path.exists(simupath) and not os.path.exists(ckptpath):
        gtpath = '/ibex/scratch/qius0a/data/synthetic'
        simupath = '/ibex/scratch/qius0a/data/synthetic_multi'
        ckptpath = '/ibex/scratch/qius0a/TransparenceDetection/src/pid/ckpt'
    return gtpath, simupath, ckptpath


def define_param():
    pass


def gen_ckptpath(modelpath, suffix=None, modelname=[], continue_train=True):
    if continue_train:
        ckptpath = modelpath
    else:
        ckptpath = os.path.join(modelpath, suffix)
        if not os.path.exists(ckptpath):
            os.makedirs(ckptpath)


def test_translate_conv():
    Irpath = '/home/lir0b/Code/TransparenceDetection/10509_r.jpg'
    Itpath = '/home/lir0b/Code/TransparenceDetection/1257_t.jpg'
    Ir = Image.open(Irpath)
    It = Image.open(Itpath)
    Ir_cuda = ToTensor()(Ir).unsqueeze(0).cuda()
    It_cuda = ToTensor()(It).unsqueeze(0).cuda()
    d = 20
    theta = 70/180*3.1415926
    pe = PolarEngine(Ir_cuda, It_cuda, theta, d)
    [Ir_para_out, Ir_perp_out] = pe.polarized_trace()
    outpath = '/home/lir0b/Code/TransparenceDetection/'
    vutils.save_image(Ir_para_out.clone(), os.path.join(outpath, 'Ir_para.jpg'), normalize=False)
    vutils.save_image(Ir_perp_out.clone(), os.path.join(outpath, 'Ir_perp.jpg'), normalize=False)

    if False:
        imgpath = '/home/lir0b/Code/TransparenceDetection/test.jpg'
        img = Image.open(imgpath)
        I_cuda = ToTensor()(img).unsqueeze(0).cuda()
        #I_cuda = to_CUDA(img)
        d = 20
        theta = 0.1*3.1415926
        tx = int(d * np.cos(theta))
        ty = int(d * np.sin(theta))
        print(tx, ty)

        [b, c, h, w] = I_cuda.shape
        filternp = np.zeros([3, 1, 41, 41])
        filternp[:, :, 20 - ty, 20 - tx] = 1
        filter = torch.Tensor(filternp).cuda()
        I_trans = torch.nn.functional.conv2d(I_cuda, filter, groups=3, padding=20)
        print(I_trans.shape, I_cuda.shape)
        outpath = '/home/lir0b/Code/TransparenceDetection/'
        I_out = 0.5*(I_trans+I_cuda)
        vutils.save_image(I_out.clone(), os.path.join(outpath, 'out.jpg'), normalize=False)
        vutils.save_image(I_trans.clone(), os.path.join(outpath, 'trans.jpg'), normalize=False)


def train_sepmodel_singlenet():
    use_CUDA = True
    from dataloader import DataLoader
    from pprint import pprint
    gtpath, simupath, ckptpath = define_path()

    polar_name = 'SepNet'
    psf_name = 'PSFNet'
    discriminator_name = 'netD'
    refD_name = 'refD'
    transD_name = 'transD'

    eps = 1e-5
    dl = DataLoader(simupath, gtpath)
    Epochs = 100
    t = 0
    numsample = 100
    rsz = (320, 320)
    save_epoch = True
    continue_train = False
    save_image = True
    device = torch.device("cuda:0" if use_CUDA else "cpu")

    #sepnet = Sepmodel().to(device)
    sepnet = Sepmodelv2().to(device)
    #sepnet.apply(weights_init)

    optimizerG = optim.Adam(sepnet.parameters(), lr=0.0002, betas=(0.5, 0.999))

    print(sepnet.parameters())

    refD = RefDiscriminator()
    transD = TransDiscriminator()
    refD.to(device)
    transD.to(device)
    refD.apply(weights_init)
    transD.apply(weights_init)
    optimizer_refD = optim.Adam(refD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_transD = optim.Adam(transD.parameters(), lr=0.0002, betas=(0.5, 0.999))


    #netD = Discriminator()
    #netD.to(device)
    #netD.apply(weights_init)
    #optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    weights = [1.0, 0, 0, 3, 1]
    #weights = [1.33, 0, 0, 0, 0]
    brightness = 1
    if continue_train:
        savefoldername = '1.00_0.00_0.00_1581451445'
        ckptpath = os.path.join(ckptpath, savefoldername)
    else:
        ckptpath = os.path.join(ckptpath, '%.2f_%.2f_%.2f_%d' % (weights[0], weights[1], weights[2], int(time.time())))
    if not os.path.exists(ckptpath):
        os.makedirs(ckptpath)
    for i in range(Epochs):
        savepathi = os.path.join(ckptpath, str(i), polar_name)
        if os.path.exists(savepathi) and continue_train:
            savepathi_1 = os.path.join(ckptpath, str(i + 1), polar_name)
            if not os.path.exists(savepathi_1):
                sepnet.load_state_dict(torch.load(os.path.join(ckptpath, str(i), polar_name)))
                #psfnet.load_state_dict(torch.load(os.path.join(ckptpath, str(i), psfname)))
                refD.load_state_dict(torch.load(os.path.join(ckptpath, str(i), refD_name)))
                transD.load_state_dict(torch.load(os.path.join(ckptpath, str(i), transD_name)))
            continue
        for idx in range(numsample):
            print(idx)
            #idx = random.randint(0, numsample) % len(dl.simpair)
            #idx = random.randint(0, numsample) % len(dl.simpair)
            ref_id, trans_id = dl.simpair[idx]
            theta_pol_filelist = dl.get_theta_pol_filelist(ref_id, trans_id)
            ref_trans_pair = dl.get_ref_trans_pair(ref_id, trans_id)

            Ir = preprocess_image(ref_trans_pair[0], rsz, brightness)
            if Ir is None:
                continue
            Ir_cuda = to_CUDA(Ir)

            It = preprocess_image(ref_trans_pair[1], rsz, brightness)
            if It is None:
                continue
            It_cuda = to_CUDA(It)

            target = torch.cat((Ir_cuda, It_cuda), dim=1).cuda()
            theta_idx = 0
            t = 0

            for idxt, thetapath in enumerate(theta_pol_filelist):
                for idxp, polpath in enumerate(thetapath):
                    img = preprocess_image(polpath, rsz, brightness)
                    if img is None:
                        continue
                    img_cuda = to_CUDA(img)

                    if False:
                        sepnet.zero_grad()
                        [Ir_pred, It_pred, I_obs] = sepnet([img_cuda])
                        allloss = sepnet.loss([Ir_cuda, It_cuda], [Ir_pred, It_pred], img_cuda, I_obs, weights)
                        [loss_sep, loss_vgg_sep, loss_exclu, loss_backward] = allloss
                        total_loss = loss_sep + loss_vgg_sep + loss_exclu + loss_backward

                    if True:
                        sepnet.zero_grad()
                        [Ir_pred, It_pred] = sepnet(img_cuda)
                        allloss = sepnet.loss([Ir_pred, It_pred], [Ir_cuda, It_cuda])
                        [loss_ref, loss_trans, total_loss] = allloss

                        loss_ref_adv = refD.gloss(refD(img_cuda, Ir_pred))
                        loss_trans_adv = transD.gloss(transD(img_cuda, It_pred))
                        total_loss = total_loss+loss_ref_adv*1000000+loss_trans_adv*1000000

                        #It_pred = It_res_pred + img_cuda
                        #[loss_sep, loss_vgg_sep, loss_exclu, loss_backward] = allloss

                    total_loss.backward(retain_graph=True)
                    optimizerG.step()
                    G_x = total_loss.item()

                    # calc discriminator loss
                    refD.zero_grad()
                    transD.zero_grad()
                    isref_pred = refD(img_cuda, Ir_pred)
                    istrans_pred = transD(img_cuda, It_pred)
                    isref_gt = refD(img_cuda, Ir_cuda)
                    istrans_gt = transD(img_cuda, It_cuda)

                    refDloss = refD.dloss(isref_pred, isref_gt)
                    refDloss.backward(retain_graph=True)
                    optimizer_refD.step()

                    transDloss = transD.dloss(istrans_pred, istrans_gt)
                    transDloss.backward(retain_graph=True)
                    optimizer_transD.step()



                    #isreal = refD(target)
                    #IrIt = torch.cat([Ir_pred, It_pred], dim=1)
                    #ispred = netD(IrIt)
                    #Dloss = netD.loss(isreal, ispred)
                    #Dloss.backward()
                    #D_x = Dloss.mean().item()
                    #optimizerD.step()


                    #netD.zero_grad()
                    #isreal = netD(target)
                    #IrIt = torch.cat([Ir_pred, It_pred], dim=1)
                    #ispred = netD(IrIt)
                    #Dloss = netD.loss(isreal, ispred)
                    #Dloss.backward()
                    #D_x = Dloss.mean().item()
                    #optimizerD.step()

                    if idx % 10 == 0:
                        ps = '%d_%d_%d_%d tot:%.6f, refl:%.6f, transl:%.6f, refD:%.6f, transD:%.6f, refG:%.6f transG:%.6f\n'
                        print(ps % (i, idx, idxt, idxp,
                                    total_loss.item(),
                                    loss_ref.item(),
                                    loss_trans.item(),
                                    refDloss.item(),
                                    transDloss.item(),
                                    loss_ref_adv.item(),
                                    loss_trans_adv.item()))

                    if idx % 10 == 0 and False:
                        print('%d_%d_%d A:%.2f Lsep:%.6f Lvgg:%.6f Lexc:%.6f Lb:%.6f \n' % (i, idxt, t, total_loss.item(),
                                                                                            loss_sep.item(),
                                                                                            loss_vgg_sep.item(),
                                                                                            loss_exclu.item(),
                                                                                            loss_backward.item()))
                        print('%d L:%.2f D:%.2f' % (i, G_x, D_x))

                    if save_image and idx % 1 == 0:
                        subfoldername = str(ref_id)+'_'+str(trans_id)
                        if not os.path.exists(os.path.join(ckptpath, str(i))):
                            os.makedirs(os.path.join(ckptpath, str(i)))
                        if not os.path.exists(os.path.join(ckptpath, str(i), subfoldername)):
                            os.makedirs(os.path.join(ckptpath, str(i), subfoldername))

                        Irpath = os.path.join(ckptpath, str(i), subfoldername, str(ref_id) + '_r.jpg')
                        if not os.path.exists(Irpath):
                            vutils.save_image(Ir_cuda.clone(), Irpath, normalize=False)
                        Itpath = os.path.join(ckptpath, str(i), subfoldername, str(trans_id) + '_t.jpg')
                        if not os.path.exists(Itpath):
                            vutils.save_image(It_cuda.clone(), Itpath, normalize=False)

                        Itpath = os.path.join(ckptpath, str(i), subfoldername, str(trans_id) + '_' + str(idxt) + '_res_pred_t.jpg')
                        if not os.path.exists(Itpath):
                            pass
                            #vutils.save_image(It_res_pred.clone(), Itpath, normalize=False)

                        Irpath_pred = os.path.join(ckptpath, str(i), subfoldername, str(ref_id) + '_' + str(idxt) + '_r_pred.jpg')
                        if not os.path.exists(Irpath_pred):
                            vutils.save_image(Ir_pred.clone(), Irpath_pred, normalize=False)
                        Itpath_pred = os.path.join(ckptpath, str(i), subfoldername, str(trans_id) + '_' + str(idxt) + '_t_pred.jpg')
                        if not os.path.exists(Itpath_pred):
                            vutils.save_image(It_pred.clone(), Itpath_pred, normalize=False)

                        #ori_path = os.path.join(ckptpath, str(i), str(ref_id)+'_'+str(trans_id) + '_' + str(idxt) + '_ori.jpg')
                        #vutils.save_image(img_cuda, ori_path, normalize=True)
                        I_obs_path = os.path.join(ckptpath, str(i), subfoldername, str(ref_id)+'_'+str(trans_id) + '_' + str(idxt) + '_obs.jpg')
                        vutils.save_image(img_cuda.clone(), I_obs_path, normalize=False)

        if save_epoch:
            savepathi = os.path.join(ckptpath, str(i))
            if not os.path.exists(savepathi):
                os.makedirs(savepathi)
            torch.save(sepnet.state_dict(), os.path.join(savepathi, polar_name))
            torch.save(refD.state_dict(), os.path.join(savepathi, refD_name))
            torch.save(transD.state_dict(), os.path.join(savepathi, transD_name))


def train_sepmodel_selected():
    use_CUDA = True
    from dataloader import DataLoader
    from pprint import pprint
    gtpath, simupath, ckptpath = define_path()

    polar_name = 'SepNet'
    psf_name = 'PSFNet'
    discriminator_name = 'netD'
    refD_name = 'refD'
    transD_name = 'transD'

    eps = 1e-5
    dl = DataLoader(simupath, gtpath, suffix='out')
    Epochs = 100
    t = 0
    numsample = 79
    rsz = (320, 320)
    save_epoch = True
    continue_train = False
    save_image = True
    device = torch.device("cuda:0" if use_CUDA else "cpu")

    sepnet = Sepmodelv3().to(device)
    optimizerG = optim.Adam(sepnet.parameters(), lr=0.0002, betas=(0.5, 0.999))
    refD = RefDiscriminator()
    transD = TransDiscriminator()
    refD.to(device)
    transD.to(device)
    refD.apply(weights_init)
    transD.apply(weights_init)
    optimizer_refD = optim.Adam(refD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_transD = optim.Adam(transD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    weights = [1.0, 0, 0, 3, 1]
    brightness = 1
    if continue_train:
        savefoldername = '1.00_0.00_0.00_1582156620'
        ckptpath = os.path.join(ckptpath, savefoldername)
    else:
        ckptpath = os.path.join(ckptpath, '%.2f_%.2f_%.2f_%d' % (weights[0], weights[1], weights[2], int(time.time())))
    if not os.path.exists(ckptpath):
        os.makedirs(ckptpath)

    # loading data
    imgpath = []

    for i in range(Epochs):
        savepathi = os.path.join(ckptpath, str(i), polar_name)
        if os.path.exists(savepathi) and continue_train:
            savepathi_1 = os.path.join(ckptpath, str(i + 1), polar_name)
            if not os.path.exists(savepathi_1):
                sepnet.load_state_dict(torch.load(os.path.join(ckptpath, str(i), polar_name)))
                #psfnet.load_state_dict(torch.load(os.path.join(ckptpath, str(i), psfname)))
                refD.load_state_dict(torch.load(os.path.join(ckptpath, str(i), refD_name)))
                transD.load_state_dict(torch.load(os.path.join(ckptpath, str(i), transD_name)))
            continue
        for idx in range(numsample):
            print(idx)
            ref_id, trans_id = dl.simpair[idx]
            #theta_pol_filelist = dl.get_theta_pol_filelist(ref_id, trans_id)
            theta_filelist = dl.get_theta_filelist(ref_id, trans_id)
            ref_trans_pair = dl.get_ref_trans_pair(ref_id, trans_id)

            Ir = preprocess_image(ref_trans_pair[0], rsz, brightness)
            if Ir is None:
                continue
            Ir_cuda = to_CUDA(Ir)

            It = preprocess_image(ref_trans_pair[1], rsz, brightness)
            if It is None:
                continue
            It_cuda = to_CUDA(It)

            for idxt, thetapath in enumerate(theta_filelist):
                print(thetapath)
                img = preprocess_image(thetapath, rsz, brightness)
                if img is None:
                    continue
                img_cuda = to_CUDA(img)
                sepnet.zero_grad()
                [Ir_pred, It_pred] = sepnet(img_cuda)
                allloss = sepnet.loss([Ir_pred, It_pred], [Ir_cuda, It_cuda])
                [loss_ref, loss_trans, total_loss] = allloss

                loss_ref_adv = refD.gloss(refD(img_cuda, Ir_pred))
                loss_trans_adv = transD.gloss(transD(img_cuda, It_pred))
                total_loss = total_loss + loss_ref_adv * 1000 + loss_trans_adv * 1000

                total_loss.backward(retain_graph=True)
                optimizerG.step()
                G_x = total_loss.item()

                # calc discriminator loss
                refD.zero_grad()
                transD.zero_grad()
                isref_pred = refD(img_cuda, Ir_pred)
                istrans_pred = transD(img_cuda, It_pred)
                isref_gt = refD(img_cuda, Ir_cuda)
                istrans_gt = transD(img_cuda, It_cuda)

                refDloss = refD.dloss(isref_pred, isref_gt)
                refDloss.backward(retain_graph=True)
                optimizer_refD.step()

                transDloss = transD.dloss(istrans_pred, istrans_gt)
                transDloss.backward(retain_graph=True)
                optimizer_transD.step()

                if idx % 1 == 0:
                    ps = '%d_%d tot:%.6f, refl:%.6f, transl:%.6f, refD:%.6f, transD:%.6f, refG:%.6f transG:%.6f\n'
                    print(ps % (i, idx,
                                total_loss.item(),
                                loss_ref.item(),
                                loss_trans.item(),
                                refDloss.item(),
                                transDloss.item(),
                                loss_ref_adv.item(),
                                loss_trans_adv.item()))

                if save_image and idx % 1 == 0:
                    subfoldername = str(ref_id)+'_'+str(trans_id)
                    if not os.path.exists(os.path.join(ckptpath, str(i))):
                        os.makedirs(os.path.join(ckptpath, str(i)))
                    if not os.path.exists(os.path.join(ckptpath, str(i), subfoldername)):
                        os.makedirs(os.path.join(ckptpath, str(i), subfoldername))

                    Irpath = os.path.join(ckptpath, str(i), subfoldername, str(ref_id) + '_r.jpg')
                    if not os.path.exists(Irpath):
                        vutils.save_image(Ir_cuda.clone(), Irpath, normalize=False)
                    Itpath = os.path.join(ckptpath, str(i), subfoldername, str(trans_id) + '_t.jpg')
                    if not os.path.exists(Itpath):
                        vutils.save_image(It_cuda.clone(), Itpath, normalize=False)

                    Itpath = os.path.join(ckptpath, str(i), subfoldername, str(trans_id) + '_' + str(idxt) + '_res_pred_t.jpg')
                    if not os.path.exists(Itpath):
                        pass
                        #vutils.save_image(It_res_pred.clone(), Itpath, normalize=False)

                    Irpath_pred = os.path.join(ckptpath, str(i), subfoldername, str(ref_id) + '_' + str(idxt) + '_r_pred.jpg')
                    if not os.path.exists(Irpath_pred):
                        vutils.save_image(Ir_pred.clone(), Irpath_pred, normalize=False)
                    Itpath_pred = os.path.join(ckptpath, str(i), subfoldername, str(trans_id) + '_' + str(idxt) + '_t_pred.jpg')
                    if not os.path.exists(Itpath_pred):
                        vutils.save_image(It_pred.clone(), Itpath_pred, normalize=False)

                    #ori_path = os.path.join(ckptpath, str(i), str(ref_id)+'_'+str(trans_id) + '_' + str(idxt) + '_ori.jpg')
                    #vutils.save_image(img_cuda, ori_path, normalize=True)
                    I_obs_path = os.path.join(ckptpath, str(i), subfoldername, str(ref_id)+'_'+str(trans_id) + '_' + str(idxt) + '_obs.jpg')
                    vutils.save_image(img_cuda.clone(), I_obs_path, normalize=False)

        if save_epoch:
            savepathi = os.path.join(ckptpath, str(i))
            if not os.path.exists(savepathi):
                os.makedirs(savepathi)
            torch.save(sepnet.state_dict(), os.path.join(savepathi, polar_name))
            torch.save(refD.state_dict(), os.path.join(savepathi, refD_name))
            torch.save(transD.state_dict(), os.path.join(savepathi, transD_name))


def train_sepmodel_4input_selected():
    use_CUDA = True
    from dataloader import DataLoader
    from pprint import pprint
    gtpath, simupath, ckptpath = define_path()

    polar_name = 'SepNet'
    psf_name = 'PSFNet'
    discriminator_name = 'netD'
    refD_name = 'refD'
    transD_name = 'transD'

    eps = 1e-5
    dl = DataLoader(simupath, gtpath, suffix='filter')
    Epochs = 100
    t = 0
    numsample = 79
    rsz = (320, 320)
    save_epoch = True
    continue_train = False
    save_image = True
    device = torch.device("cuda:0" if use_CUDA else "cpu")

    sepnet = Sepmodelv3().to(device)
    optimizerG = optim.Adam(sepnet.parameters(), lr=0.0002, betas=(0.5, 0.999))
    refD = RefDiscriminator()
    transD = TransDiscriminator()
    refD.to(device)
    transD.to(device)
    refD.apply(weights_init)
    transD.apply(weights_init)
    optimizer_refD = optim.Adam(refD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_transD = optim.Adam(transD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    weights = [1.0, 0, 0, 3, 1]
    brightness = 1
    if continue_train:
        savefoldername = '1.00_0.00_0.00_1582156620'
        ckptpath = os.path.join(ckptpath, savefoldername)
    else:
        ckptpath = os.path.join(ckptpath, '%.2f_%.2f_%.2f_%d' % (weights[0], weights[1], weights[2], int(time.time())))
    if not os.path.exists(ckptpath):
        os.makedirs(ckptpath)

    # loading data
    imgpath = []

    for i in range(Epochs):
        savepathi = os.path.join(ckptpath, str(i), polar_name)
        if os.path.exists(savepathi) and continue_train:
            savepathi_1 = os.path.join(ckptpath, str(i + 1), polar_name)
            if not os.path.exists(savepathi_1):
                sepnet.load_state_dict(torch.load(os.path.join(ckptpath, str(i), polar_name)))
                # psfnet.load_state_dict(torch.load(os.path.join(ckptpath, str(i), psfname)))
                refD.load_state_dict(torch.load(os.path.join(ckptpath, str(i), refD_name)))
                transD.load_state_dict(torch.load(os.path.join(ckptpath, str(i), transD_name)))
            continue
        for idx in range(numsample):
            print(idx)
            ref_id, trans_id = dl.simpair[idx]
            theta_pol_filelist = dl.get_theta_pol_filelist(ref_id, trans_id)
            #theta_filelist = dl.get_theta_filelist(ref_id, trans_id)

            ref_trans_pair = dl.get_ref_trans_pair(ref_id, trans_id)

            Ir = preprocess_image(ref_trans_pair[0], rsz, brightness)
            if Ir is None:
                continue
            Ir_cuda = to_CUDA(Ir)

            It = preprocess_image(ref_trans_pair[1], rsz, brightness)
            if It is None:
                continue
            It_cuda = to_CUDA(It)

            for idxt, polfilelist in enumerate(theta_pol_filelist):
                img1 = preprocess_image(polfilelist[0], rsz, brightness)
                img2 = preprocess_image(polfilelist[1], rsz, brightness)
                img3 = preprocess_image(polfilelist[2], rsz, brightness)
                img4 = preprocess_image(polfilelist[3], rsz, brightness)
                if img1 is None or img2 is None or img3 is None or img4 is None:
                    continue
                img1_cuda = to_CUDA(img1)
                img2_cuda = to_CUDA(img2)
                img3_cuda = to_CUDA(img3)
                img4_cuda = to_CUDA(img4)
                I_obs = [img1_cuda, img2_cuda, img3_cuda, img4_cuda]

                sepnet.zero_grad()
                [Ir_pred, It_pred] = sepnet(img1_cuda, img2_cuda, img3_cuda, img4_cuda)
                [loss_ref, loss_trans, total_loss] = sepnet.loss(Ir_pred, Ir_cuda, It_pred, It_cuda)

                refGloss = torch.sum(torch.stack([refD.gloss(refD(ii, Ir_pred)) for ii in I_obs], dim=0))
                transGloss = torch.sum(torch.stack([transD.gloss(transD(ii, It_pred)) for ii in I_obs], dim=0))
                total_loss = total_loss + 1000 * refGloss
                total_loss = total_loss + 1000 * transGloss
                total_loss.backward(retain_graph=True)
                optimizerG.step()

                # train discriminator here
                refD.zero_grad()
                transD.zero_grad()
                refDloss = torch.sum(torch.stack([refD.dloss(refD(ii, Ir_pred), refD(ii, Ir_cuda)) for ii in I_obs], dim=0))
                transDloss = torch.sum(torch.stack([transD.dloss(transD(ii, It_pred), transD(ii, It_cuda)) for ii in I_obs], dim=0))
                refDloss.backward(retain_graph=True)
                optimizer_refD.step()
                transDloss.backward(retain_graph=True)
                optimizer_transD.step()

                if idx % 1 == 0:
                    ps = '%d_%d tot:%.6f, refl:%.6f, transl:%.6f, refD:%.6f, transD:%.6f, refG:%.6f transG:%.6f\n'
                    print(ps % (i, idx,
                                total_loss.item(),
                                loss_ref.item(),
                                loss_trans.item(),
                                refDloss.item(),
                                transDloss.item(),
                                refGloss.item(),
                                transGloss.item()))

                if save_image and idx % 1 == 0:
                    subfoldername = str(ref_id) + '_' + str(trans_id)
                    if not os.path.exists(os.path.join(ckptpath, str(i))):
                        os.makedirs(os.path.join(ckptpath, str(i)))
                    if not os.path.exists(os.path.join(ckptpath, str(i), subfoldername)):
                        os.makedirs(os.path.join(ckptpath, str(i), subfoldername))

                    Irpath = os.path.join(ckptpath, str(i), subfoldername, str(ref_id) + '_r.jpg')
                    if not os.path.exists(Irpath):
                        vutils.save_image(Ir_cuda.clone(), Irpath, normalize=False)
                    Itpath = os.path.join(ckptpath, str(i), subfoldername, str(trans_id) + '_t.jpg')
                    if not os.path.exists(Itpath):
                        vutils.save_image(It_cuda.clone(), Itpath, normalize=False)

                    Irpath_pred = os.path.join(ckptpath, str(i), subfoldername, str(ref_id) + '_' + str(idxt) + '_r_pred.jpg')
                    if not os.path.exists(Irpath_pred):
                        vutils.save_image(Ir_pred.clone(), Irpath_pred, normalize=False)
                    Itpath_pred = os.path.join(ckptpath, str(i), subfoldername, str(trans_id) + '_' + str(idxt) + '_t_pred.jpg')
                    if not os.path.exists(Itpath_pred):
                        vutils.save_image(It_pred.clone(), Itpath_pred, normalize=False)

                    for phi, i_obs in enumerate(I_obs):
                        I_obs_path = os.path.join(ckptpath, str(i), subfoldername, str(ref_id) + '_' + str(trans_id) + '_' + str(phi) + '_obs.jpg')
                        vutils.save_image(i_obs.clone(), I_obs_path, normalize=False)

        if save_epoch:
            savepathi = os.path.join(ckptpath, str(i))
            if not os.path.exists(savepathi):
                os.makedirs(savepathi)
            torch.save(sepnet.state_dict(), os.path.join(savepathi, polar_name))
            #torch.save(refD.state_dict(), os.path.join(savepathi, refD_name))
            torch.save(transD.state_dict(), os.path.join(savepathi, transD_name))


def train_sepmodel_syntheticdata():
    use_CUDA = True
    from dataloader import DataLoader
    from pprint import pprint
    import numpy as np
    gtpath, simupath, ckptpath = define_path()

    #gtlist = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    #gtlist = [6, 8, 9, 10, 11, 12, 13, 17, 18,  21, 22, 24]
    #gtlist = [8, 12, 18, 19]
    #gtlist = [16, 17]

    gtpath = '/home/lir0b/Code/perceptual-reflection-removal/synthetic/reflection_layer'
    polarpath = '/home/lir0b/Code/TransparenceDetection/src/pid/exp/paperdata/strong_ref'
    ckptpath = '/home/lir0b/Code/TransparenceDetection/src/pid/ckpt'
    if not os.path.exists(gtpath) and not os.path.exists(polarpath) and not os.path.exists(ckptpath):
        gtpath = '/ibex/scratch/lir0b/data/dataset/gt'
        polarpath = '/ibex/scratch/lir0b/data/dataset/polar'
        ckptpath = '/ibex/scratch/lir0b/TransparenceDetection/src/pid/ckpt'

    folderlist = glob.glob(os.path.join(polarpath, '*'))
    folderlist = [f.split('/')[-1] for f in folderlist]
    reflist = [f.split('_')[0] for f in folderlist]
    translist = [f.split('_')[1] for f in folderlist]

    polar_name = 'SepNet'
    #psf_name = 'PSFNet'
    discriminator_name = 'netD'
    refD_name = 'refD'
    transD_name = 'transD'
    save_epoch = True
    continue_train = True
    save_image = True

    device = torch.device("cuda:0" if use_CUDA else "cpu")
    sepnet = Sepmodelv3().to(device)
    #optimizerG = optim.Adam(sepnet.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(sepnet.parameters(), lr=0.0001, betas=(0.5, 0.999))
    #refD = RefDiscriminator()
    #transD = TransDiscriminator()
    #refD.to(device)
    #transD.to(device)
    #refD.apply(weights_init)
    #transD.apply(weights_init)
    #optimizer_refD = optim.Adam(refD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    #optimizer_transD = optim.Adam(transD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    epoch = 400
    numsample = 500
    weights = [1.0, 0, 0, 3, 1]
    rsz = (512, 512)
    #box = [0, 0, 1024, 1024]
    #thetalist = [10, 40, 60, 70, 80, 85]
    thetalist = [40, 60, 70, 80]
    if continue_train:
        ckptpath = os.path.join(ckptpath, '1.00_0.00_0.00_1583877271')
    else:
        ckptpath = os.path.join(ckptpath, '%.2f_%.2f_%.2f_%d' % (weights[0], weights[1], weights[2], int(time.time())))

    for i in range(epoch):
        savepathi = os.path.join(ckptpath, str(i), polar_name)
        if os.path.exists(savepathi) and continue_train:
            savepathi_1 = os.path.join(ckptpath, str(i + 1), polar_name)
            if not os.path.exists(savepathi_1):
                sepnet.load_state_dict(torch.load(os.path.join(ckptpath, str(i), polar_name)))
                #refD.load_state_dict(torch.load(os.path.join(ckptpath, str(i), refD_name)))
                #transD.load_state_dict(torch.load(os.path.join(ckptpath, str(i), transD_name)))
            continue

        for idx in range(numsample):
            iter = random.randint(0, numsample) % len(reflist)
            refid = reflist[iter]
            transid = translist[iter]
            refpath = os.path.join(gtpath, str(refid)+'.jpg')
            transpath = os.path.join(gtpath, str(transid)+'.jpg')
            theta = thetalist[random.randint(1, 100)%len(thetalist)]
            foldername = str(refid)+'_'+str(transid)
            filename = foldername+'_theta_'+str(theta)+'_phi'
            I0path = os.path.join(polarpath, foldername,
                                  'polar',  filename + '_0_filter.jpg')
            I45path = os.path.join(polarpath, foldername,
                                  'polar', filename + '_45_filter.jpg')
            I90path = os.path.join(polarpath, foldername,
                                  'polar', filename + '_90_filter.jpg')
            I135path = os.path.join(polarpath, foldername,
                                  'polar', filename + '_135_filter.jpg')

            left = random.randint(0, 128)
            top = random.randint(0, 128)
            right = left+512
            bottom = top+512
            box = (left, top, right, bottom)
            data_argument = random.randint(0, 100) < 10

            Igt0 = preprocess_image(transpath, rsz, box, data_argument)
            Igt45 = preprocess_image(transpath, rsz, box, data_argument)
            Igt90 = preprocess_image(transpath, rsz, box, data_argument)
            Igt135 = preprocess_image(transpath, rsz, box, data_argument)
            if Igt0 is None or Igt45 is None or Igt90 is None or Igt135 is None:
                continue
            Igt0_cuda = to_CUDA(Igt0)
            Igt45_cuda = to_CUDA(Igt45)
            Igt90_cuda = to_CUDA(Igt90)
            Igt135_cuda = to_CUDA(Igt135)
            I_gt = [Igt0_cuda, Igt45_cuda, Igt90_cuda, Igt135_cuda]
            It_gt = 0.25*(Igt0_cuda+Igt45_cuda+Igt90_cuda+Igt135_cuda)

            I0 = preprocess_image(I0path, rsz, box, data_argument)
            I45 = preprocess_image(I45path, rsz, box, data_argument)
            I90 = preprocess_image(I90path, rsz, box, data_argument)
            I135 = preprocess_image(I135path, rsz, box, data_argument)

            if I0 is None or I45 is None or I90 is None or I135 is None:
                continue
            I0_cuda = to_CUDA(I0)
            I45_cuda = to_CUDA(I45)
            I90_cuda = to_CUDA(I90)
            I135_cuda = to_CUDA(I135)
            I_obs = [I0_cuda, I45_cuda, I90_cuda, I135_cuda]

            sepnet.zero_grad()
            [It_refine, It_pred1, It_pred2, It_pred3, It_pred4] = sepnet(I0_cuda, I45_cuda, I90_cuda, I135_cuda)
            It_pred = [It_pred1, It_pred2, It_pred3, It_pred4]
            Ir_pred = [I0_cuda-It_pred1, I45_cuda-It_pred2, I90_cuda-It_pred3, I135_cuda-It_pred4]
            [Ir_pred1, Ir_pred2, Ir_pred3, Ir_pred4] = Ir_pred
            Ir_refine = 0.25*(Ir_pred1+Ir_pred2+Ir_pred3+Ir_pred4)
            [loss_polar, loss_refine, loss_vgg, total_loss] = sepnet.loss(It_refine, It_gt,  It_pred, I_gt)

            #loss_trans = sepnet.loss_trans(It_pred, It_cuda)
            #refGloss = torch.sum(torch.stack([refD.gloss(refD(ii, Ir_pred)) for ii in I_obs], dim=0))
            #print(transD(It_refine, It_gt).shape)
            #transGloss = transD.gloss(transD(It_refine, It_gt))
            #total_loss = total_loss + 1000 * refGloss
            #total_loss = total_loss
            total_loss.backward(retain_graph=False)
            optimizerG.step()

            # train discriminator here
            #refD.zero_grad()
            #transD.zero_grad()
            #refDloss = torch.sum(torch.stack([refD.dloss(refD(ii, Ir_pred), refD(ii, Ir_cuda)) for ii in I_obs], dim=0))
            #transDloss_list = []
            #for iii in range(len(I_obs)):
            #    transDloss_list.append(transD.dloss(transD(I_obs[iii], It_pred[iii]), transD(I_obs[iii], I_gt[iii])))
            #transDloss = torch.sum(torch.stack(transDloss_list), dim=0)
            #transDloss = torch.sum(torch.stack([transD.dloss(transD(ii, It_pred), transD(ii, It_cuda)) for ii in I_obs], dim=0))
            #transDloss = transD.dloss(transD(ii, It_pred), transD())
            #refDloss.backward(retain_graph=True)
            #optimizer_refD.step()
            #transDloss.backward(retain_graph=True)
            #optimizer_transD.step()

            if idx % 1 == 0:
                #ps = '%d_%d tot:%.6f, lpolar:%.6f, lref:%.6f, transD:%.6f, transG:%.12f\n'
                ps = '%d_%d tot:%.6f, lpolar:%.6f, lref:%.6f vgg:%.6f\n'
                print(ps % (i, idx, total_loss.item(), loss_polar.item(), loss_refine.item(), loss_vgg.item()))

            if save_image and idx % 1 == 0 and not data_argument:
                subfoldername = str(refid)+'_'+str(transid)
                if not os.path.exists(os.path.join(ckptpath, str(i))):
                    os.makedirs(os.path.join(ckptpath, str(i)))
                if not os.path.exists(os.path.join(ckptpath, str(i), subfoldername)):
                    os.makedirs(os.path.join(ckptpath, str(i), subfoldername))
                #It_refine, It_pred1, It_pred2, It_pred3, It_pred4
                idp = foldername+'_theta_'+str(theta)
                Itpath = os.path.join(ckptpath, str(i), subfoldername, idp + '_refine.png')
                if not os.path.exists(Itpath):
                    vutils.save_image(It_refine.clone(), Itpath, normalize=False)
                    I = Image.open(Itpath)
                    I_crop = center_crop(I, (512, 512))
                    I_crop.save(Itpath)

                Itpath = os.path.join(ckptpath, str(i), subfoldername, idp + '_pred1.png')
                if not os.path.exists(Itpath):
                    vutils.save_image(It_pred1.clone(), Itpath, normalize=False)
                    I = Image.open(Itpath)
                    I_crop = center_crop(I, (512, 512))
                    I_crop.save(Itpath)

                Itpath = os.path.join(ckptpath, str(i), subfoldername, idp + '_pred2.png')
                if not os.path.exists(Itpath):
                    vutils.save_image(It_pred2.clone(), Itpath, normalize=False)
                    I = Image.open(Itpath)
                    I_crop = center_crop(I, (512, 512))
                    I_crop.save(Itpath)

                Itpath = os.path.join(ckptpath, str(i), subfoldername, idp + '_pred3.png')
                if not os.path.exists(Itpath):
                    vutils.save_image(It_pred3.clone(), Itpath, normalize=False)
                    I = Image.open(Itpath)
                    I_crop = center_crop(I, (512, 512))
                    I_crop.save(Itpath)

                Itpath = os.path.join(ckptpath, str(i), subfoldername, idp + '_pred4.png')
                if not os.path.exists(Itpath):
                    vutils.save_image(It_pred4.clone(), Itpath, normalize=False)
                    I = Image.open(Itpath)
                    I_crop = center_crop(I, (512, 512))
                    I_crop.save(Itpath)

                # output Ir here
                Irpath = os.path.join(ckptpath, str(i), subfoldername, idp + '_refine_r.png')
                if not os.path.exists(Irpath):
                    vutils.save_image(Ir_refine.clone(), Irpath, normalize=False)
                    I = Image.open(Irpath)
                    I_crop = center_crop(I, (512, 512))
                    I_crop.save(Irpath)

                Irpath = os.path.join(ckptpath, str(i), subfoldername, idp + '_pred1_r.png')
                if not os.path.exists(Irpath):
                    vutils.save_image(Ir_pred1.clone(), Irpath, normalize=False)
                    I = Image.open(Irpath)
                    I_crop = center_crop(I, (512, 512))
                    I_crop.save(Irpath)

                Irpath = os.path.join(ckptpath, str(i), subfoldername, idp + '_pred2_r.png')
                if not os.path.exists(Irpath):
                    vutils.save_image(Ir_pred2.clone(), Irpath, normalize=False)
                    I = Image.open(Irpath)
                    I_crop = center_crop(I, (512, 512))
                    I_crop.save(Irpath)

                Irpath = os.path.join(ckptpath, str(i), subfoldername, idp + '_pred3_r.png')
                if not os.path.exists(Irpath):
                    vutils.save_image(Ir_pred3.clone(), Irpath, normalize=False)
                    I = Image.open(Irpath)
                    I_crop = center_crop(I, (512, 512))
                    I_crop.save(Irpath)

                Irpath = os.path.join(ckptpath, str(i), subfoldername, idp + '_pred4_r.png')
                if not os.path.exists(Irpath):
                    vutils.save_image(Ir_pred4.clone(), Irpath, normalize=False)
                    I = Image.open(Irpath)
                    I_crop = center_crop(I, (512, 512))
                    I_crop.save(Irpath)

        if save_epoch:
            savepathi = os.path.join(ckptpath, str(i))
            if not os.path.exists(savepathi):
                os.makedirs(savepathi)
            torch.save(sepnet.state_dict(), os.path.join(savepathi, polar_name))
            #torch.save(refD.state_dict(), os.path.join(savepathi, refD_name))
            #torch.save(transD.state_dict(), os.path.join(savepathi, transD_name))


def test_sepmodel_data():
    use_CUDA = True
    from dataloader import DataLoader
    from pprint import pprint
    import numpy as np
    gtpath, simupath, ckptpath = define_path()

    #gtlist = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    #gtlist = [6, 8, 9, 10, 11, 12, 13, 17, 18,  21, 22, 24]
    #gtlist = [8, 12, 18, 19]
    #gtlist = [16, 17]

    gtpath = '/home/lir0b/Code/perceptual-reflection-removal/synthetic/reflection_layer'
    polarpath = '/home/lir0b/Code/TransparenceDetection/src/pid/exp/paperdata/strong_ref'
    ckptpath = '/home/lir0b/Code/TransparenceDetection/src/pid/ckpt'
    if not os.path.exists(gtpath) and not os.path.exists(polarpath) and not os.path.exists(ckptpath):
        gtpath = '/ibex/scratch/lir0b/data/dataset/gt'
        polarpath = '/ibex/scratch/lir0b/data/dataset/polar'
        ckptpath = '/ibex/scratch/lir0b/TransparenceDetection/src/pid/ckpt'

    folderlist = glob.glob(os.path.join(polarpath, '*'))
    folderlist = [f.split('/')[-1] for f in folderlist]
    reflist = [f.split('_')[0] for f in folderlist]
    translist = [f.split('_')[1] for f in folderlist]

    polar_name = 'SepNet'
    #psf_name = 'PSFNet'
    discriminator_name = 'netD'
    refD_name = 'refD'
    transD_name = 'transD'
    save_epoch = True
    continue_train = True
    save_image = True

    device = torch.device("cuda:0" if use_CUDA else "cpu")
    sepnet = Sepmodelv3().to(device)
    #optimizerG = optim.Adam(sepnet.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(sepnet.parameters(), lr=0.0001, betas=(0.5, 0.999))
    #refD = RefDiscriminator()
    #transD = TransDiscriminator()
    #refD.to(device)
    #transD.to(device)
    #refD.apply(weights_init)
    #transD.apply(weights_init)
    #optimizer_refD = optim.Adam(refD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    #optimizer_transD = optim.Adam(transD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    epoch = 400
    numsample = 500
    weights = [1.0, 0, 0, 3, 1]
    rsz = (512, 512)
    #box = [0, 0, 1024, 1024]
    #thetalist = [10, 40, 60, 70, 80, 85]
    thetalist = [40, 60, 70, 80]
    foldername = '286'
    ckptpath = os.path.join(ckptpath, '1.00_0.00_0.00_1583877271', foldername)
    sepnet.load_state_dict(torch.load(os.path.join(ckptpath, polar_name)))

    polarpath = '/home/lir0b/data/polar/realdata/dataset/ghost'
    #filename = 'LUCID_PHX050S-Q_190100163__20200313194406465_image0_'
    filename = 'LUCID_PHX050S-Q_190100163__20200313194457946_image0_'
    outputname = 'ghost_2'

    I0path = os.path.join(polarpath, filename + '0d.png')
    I45path = os.path.join(polarpath, filename + '45d.png')
    I90path = os.path.join(polarpath, filename + '90d.png')
    I135path = os.path.join(polarpath, filename + '135d.png')
    Igt = I0path

    data_argument=False
    I0 = preprocess_image(I0path, sz=(512,512), data_argument=False)
    I45 = preprocess_image(I45path, sz=(512,512), data_argument=False)
    I90 = preprocess_image(I90path, sz=(512,512), data_argument=False)
    I135 = preprocess_image(I135path, sz=(512,512), data_argument=False)
    It_gt = preprocess_image(Igt, sz=(512,512), data_argument=False)
    #if I0 is None or I45 is None or I90 is None or I135 is None:
    #    continue
    I0_cuda = to_CUDA(I0)
    I45_cuda = to_CUDA(I45)
    I90_cuda = to_CUDA(I90)
    I135_cuda = to_CUDA(I135)
    It_gt_cuda = to_CUDA(It_gt)
    I_gt = [It_gt_cuda, It_gt_cuda, It_gt_cuda, It_gt_cuda]
    for i in range(10):
        sepnet.zero_grad()
        [It_refine, It_pred1, It_pred2, It_pred3, It_pred4] = sepnet(I0_cuda, I45_cuda, I90_cuda, I135_cuda)
        It_pred = [It_pred1, It_pred2, It_pred3, It_pred4]
        Ir_pred = [I0_cuda - It_pred1, I45_cuda - It_pred2, I90_cuda - It_pred3, I135_cuda - It_pred4]
        [Ir_pred1, Ir_pred2, Ir_pred3, Ir_pred4] = Ir_pred
        Ir_refine = 0.25 * (Ir_pred1 + Ir_pred2 + Ir_pred3 + Ir_pred4)
        [loss_polar, loss_refine, loss_vgg, total_loss] = sepnet.loss(It_refine, It_gt_cuda, It_pred, I_gt)
        total_loss.backward(retain_graph=False)
        optimizerG.step()
        ps = '%d tot:%.6f, lpolar:%.6f, lref:%.6f vgg:%.6f\n'
        print(ps % (i, total_loss.item(), loss_polar.item(), loss_refine.item(), loss_vgg.item()))

    if save_image:
        outputpath = '/home/lir0b/data/polar/realdata/dataset/ghost/output'
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
        idp = outputname
        Itpath = os.path.join(outputpath, idp + '_refine.png')
        #if not os.path.exists(Itpath):
        vutils.save_image(It_refine.clone(), Itpath, normalize=False)
        I = Image.open(Itpath)
        I_crop = center_crop(I, (512, 512))
        I_crop.save(Itpath)

        #Itpath = os.path.join(ckptpath, str(i), subfoldername, idp + '_pred1.png')
        #I0_cuda, I45_cuda, I90_cuda, I135_cuda
        Itpath = os.path.join(outputpath, idp + '_0.png')
        vutils.save_image(I0_cuda.clone(), Itpath, normalize=False)
        I = Image.open(Itpath)
        I_crop = center_crop(I, (512, 512))
        I_crop.save(Itpath)

        Itpath = os.path.join(outputpath, idp + '_45.png')
        vutils.save_image(I45_cuda.clone(), Itpath, normalize=False)
        I = Image.open(Itpath)
        I_crop = center_crop(I, (512, 512))
        I_crop.save(Itpath)

        Itpath = os.path.join(outputpath, idp + '_90.png')
        vutils.save_image(I90_cuda.clone(), Itpath, normalize=False)
        I = Image.open(Itpath)
        I_crop = center_crop(I, (512, 512))
        I_crop.save(Itpath)

        Itpath = os.path.join(outputpath, idp + '_135.png')
        vutils.save_image(I135_cuda.clone(), Itpath, normalize=False)
        I = Image.open(Itpath)
        I_crop = center_crop(I, (512, 512))
        I_crop.save(Itpath)

        Itpath = os.path.join(outputpath, idp + '_pred1.png')
        #if not os.path.exists(Itpath):
        vutils.save_image(It_pred1.clone(), Itpath, normalize=False)
        I = Image.open(Itpath)
        I_crop = center_crop(I, (512, 512))
        I_crop.save(Itpath)

        #Itpath = os.path.join(ckptpath, str(i), subfoldername, idp + '_pred2.png')
        Itpath = os.path.join(outputpath, idp + '_pred2.png')
        #if not os.path.exists(Itpath):
        vutils.save_image(It_pred2.clone(), Itpath, normalize=False)
        I = Image.open(Itpath)
        I_crop = center_crop(I, (512, 512))
        I_crop.save(Itpath)

        #Itpath = os.path.join(ckptpath, str(i), subfoldername, idp + '_pred3.png')
        Itpath = os.path.join(outputpath, idp + '_pred3.png')
        #if not os.path.exists(Itpath):
        vutils.save_image(It_pred3.clone(), Itpath, normalize=False)
        I = Image.open(Itpath)
        I_crop = center_crop(I, (512, 512))
        I_crop.save(Itpath)

        #Itpath = os.path.join(ckptpath, str(i), subfoldername, idp + '_pred4.png')
        Itpath = os.path.join(outputpath, idp + '_pred4.png')
        #if not os.path.exists(Itpath):
        vutils.save_image(It_pred4.clone(), Itpath, normalize=False)
        I = Image.open(Itpath)
        I_crop = center_crop(I, (512, 512))
        I_crop.save(Itpath)

        # output Ir here
        #Irpath = os.path.join(ckptpath, str(i), subfoldername, idp + '_refine_r.png')
        Irpath = os.path.join(outputpath, idp + '_refine_r.png')
        #if not os.path.exists(Irpath):
        vutils.save_image(Ir_refine.clone(), Irpath, normalize=False)
        I = Image.open(Irpath)
        I_crop = center_crop(I, (512, 512))
        I_crop.save(Irpath)

        Irpath = os.path.join(outputpath, idp + '_pred1_r.png')
        #if not os.path.exists(Irpath):
        vutils.save_image(Ir_pred1.clone(), Irpath, normalize=False)
        I = Image.open(Irpath)
        I_crop = center_crop(I, (512, 512))
        I_crop.save(Irpath)

        Irpath = os.path.join(outputpath, idp + '_pred2_r.png')
        #if not os.path.exists(Irpath):
        vutils.save_image(Ir_pred2.clone(), Irpath, normalize=False)
        I = Image.open(Irpath)
        I_crop = center_crop(I, (512, 512))
        I_crop.save(Irpath)

        Irpath = os.path.join(outputpath, idp + '_pred3_r.png')
        #if not os.path.exists(Irpath):
        vutils.save_image(Ir_pred3.clone(), Irpath, normalize=False)
        I = Image.open(Irpath)
        I_crop = center_crop(I, (512, 512))
        I_crop.save(Irpath)

        Irpath = os.path.join(outputpath, idp + '_pred4_r.png')
        #if not os.path.exists(Irpath):
        vutils.save_image(Ir_pred4.clone(), Irpath, normalize=False)
        I = Image.open(Irpath)
        I_crop = center_crop(I, (512, 512))
        I_crop.save(Irpath)


def train_sepmodel_realdata():
    use_CUDA = True
    from dataloader import DataLoader
    from pprint import pprint
    import numpy as np
    gtpath, simupath, ckptpath = define_path()

    #gtlist = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    #gtlist = [6, 8, 9, 10, 11, 12, 13, 17, 18,  21, 22, 24]
    gtlist = [8, 12, 18, 19]
    #gtlist = [16, 17]

    gtpath = '/home/lir0b/data/polar/realdata/dataset/gt'
    polarpath = '/home/lir0b/data/polar/realdata/dataset/polar'
    ckptpath = '/home/lir0b/Code/TransparenceDetection/src/pid/ckpt'
    if not os.path.exists(gtpath) and not os.path.exists(polarpath) and not os.path.exists(ckptpath):
        gtpath = '/ibex/scratch/lir0b/data/dataset/gt'
        polarpath = '/ibex/scratch/lir0b/data/dataset/polar'
        ckptpath = '/ibex/scratch/lir0b/TransparenceDetection/src/pid/ckpt'


    polar_name = 'SepNet'
    psf_name = 'PSFNet'
    discriminator_name = 'netD'
    refD_name = 'refD'
    transD_name = 'transD'
    save_epoch = True
    continue_train = True
    save_image = True

    device = torch.device("cuda:0" if use_CUDA else "cpu")
    sepnet = Sepmodelv3().to(device)
    #optimizerG = optim.Adam(sepnet.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(sepnet.parameters(), lr=0.0001, betas=(0.5, 0.999))
    #refD = RefDiscriminator()
    #transD = TransDiscriminator()
    #refD.to(device)
    #transD.to(device)
    #refD.apply(weights_init)
    #transD.apply(weights_init)
    #optimizer_refD = optim.Adam(refD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    #optimizer_transD = optim.Adam(transD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    epoch = 400
    weights = [1.0, 0, 0, 3, 1]
    #rsz = (512, 512)
    rsz = (512, 512)
    box = [0, 0, 1024, 1024]
    if continue_train:
        ckptpath = os.path.join(ckptpath, '1.00_0.00_0.00_1583022852')
    else:
        ckptpath = os.path.join(ckptpath, '%.2f_%.2f_%.2f_%d' % (weights[0], weights[1], weights[2], int(time.time())))

    for i in range(epoch):
        savepathi = os.path.join(ckptpath, str(i), polar_name)
        if os.path.exists(savepathi) and continue_train:
            savepathi_1 = os.path.join(ckptpath, str(i + 1), polar_name)
            if not os.path.exists(savepathi_1):
                sepnet.load_state_dict(torch.load(os.path.join(ckptpath, str(i), polar_name)))
                # psfnet.load_state_dict(torch.load(os.path.join(ckptpath, str(i), psfname)))
                #refD.load_state_dict(torch.load(os.path.join(ckptpath, str(i), refD_name)))
                #transD.load_state_dict(torch.load(os.path.join(ckptpath, str(i), transD_name)))
            continue

        #for idx, folder in enumerate(gtlist):
        numsample = 300
        for idx in range(numsample):
            iter = random.randint(0, numsample) % len(gtlist)
            folder = gtlist[iter]
            curgtlist = glob.glob(os.path.join(gtpath, str(folder), '*.png'))
            #pprint(curgtlist)
            curpolarlist = glob.glob(os.path.join(polarpath, str(folder), '*.png'))
            #pprint(curpolarlist)
            tmp = [os.path.split(tt)[1] for tt in curpolarlist]
            tmp2 = [tt.split('_') for tt in tmp]
            idlist = [tt[4] for tt in tmp2]
            #pprint(list(np.unique(idlist)))
            idlist = list(np.unique(idlist))
            #'LUCID_PHX050S-Q_190100163__20200224191335776_image0_0d.png'
            prefix = 'LUCID_PHX050S-Q_190100163__'
            suffix = '_image0_'
            polarlist = []
            gt0 = curgtlist[0]
            gt1 = curgtlist[1]
            gt2 = curgtlist[2]
            gt3 = curgtlist[3]

            brightness = 1

            Igt0 = preprocess_image(curgtlist[0], rsz, brightness, box)
            Igt45 = preprocess_image(curgtlist[1], rsz, brightness, box)
            Igt90 = preprocess_image(curgtlist[2], rsz, brightness, box)
            Igt135 = preprocess_image(curgtlist[3], rsz, brightness, box)
            if Igt0 is None or Igt45 is None or Igt90 is None or Igt135 is None:
                continue
            Igt0_cuda = to_CUDA(Igt0)
            Igt45_cuda = to_CUDA(Igt45)
            Igt90_cuda = to_CUDA(Igt90)
            Igt135_cuda = to_CUDA(Igt135)
            I_gt = [Igt0_cuda, Igt45_cuda, Igt90_cuda, Igt135_cuda]
            It_gt = 0.25*(Igt0_cuda+Igt45_cuda+Igt90_cuda+Igt135_cuda)

            for idxx, idp in enumerate(idlist):
                if random.randint(0, 11) > 5:
                    continue
                ttt = []
                ttt.append(prefix + idp + suffix + '0d.png')
                ttt.append(prefix + idp + suffix + '45d.png')
                ttt.append(prefix + idp + suffix + '90d.png')
                ttt.append(prefix + idp + suffix + '135d.png')
                polarlist.append(ttt)

                # pprint(polarlist)
                img0 = prefix + idp + suffix + '0d.png'
                img45 = prefix + idp + suffix + '45d.png'
                img90 = prefix + idp + suffix + '90d.png'
                img135 = prefix + idp + suffix + '135d.png'
                path0 = os.path.join(polarpath, str(folder), img0)
                path45 = os.path.join(polarpath, str(folder), img45)
                path90 = os.path.join(polarpath, str(folder), img90)
                path135 = os.path.join(polarpath, str(folder), img135)

                I0 = preprocess_image(path0, rsz, brightness, box)
                I45 = preprocess_image(path45, rsz, brightness, box)
                I90 = preprocess_image(path90, rsz, brightness, box)
                I135 = preprocess_image(path135, rsz, brightness, box)
                if I0 is None or I45 is None or I90 is None or I135 is None:
                    continue
                I0_cuda = to_CUDA(I0)
                I45_cuda = to_CUDA(I45)
                I90_cuda = to_CUDA(I90)
                I135_cuda = to_CUDA(I135)
                I_obs = [I0_cuda, I45_cuda, I90_cuda, I135_cuda]

                sepnet.zero_grad()
                #[Ir_pred, It_pred] = sepnet(I0_cuda, I45_cuda, I90_cuda, I135_cuda)
                [It_refine, It_pred1, It_pred2, It_pred3, It_pred4] = sepnet(I0_cuda, I45_cuda, I90_cuda, I135_cuda)
                It_pred = [It_pred1, It_pred2, It_pred3, It_pred4]
                Ir_pred = [I0_cuda-It_pred1, I45_cuda-It_pred2, I90_cuda-It_pred3, I135_cuda-It_pred4]
                [Ir_pred1, Ir_pred2, Ir_pred3, Ir_pred4] = Ir_pred
                Ir_refine = 0.25*(Ir_pred1+Ir_pred2+Ir_pred3+Ir_pred4)
                [loss_polar, loss_refine, loss_vgg, total_loss] = sepnet.loss(It_refine, It_gt,  It_pred, I_gt)
                #loss_trans = sepnet.loss_trans(It_pred, It_cuda)

                #refGloss = torch.sum(torch.stack([refD.gloss(refD(ii, Ir_pred)) for ii in I_obs], dim=0))
                #print(transD(It_refine, It_gt).shape)
                #transGloss = transD.gloss(transD(It_refine, It_gt))
                #total_loss = total_loss + 1000 * refGloss
                total_loss = total_loss
                total_loss.backward(retain_graph=False)
                optimizerG.step()

                # train discriminator here
                #refD.zero_grad()
                #transD.zero_grad()
                #refDloss = torch.sum(torch.stack([refD.dloss(refD(ii, Ir_pred), refD(ii, Ir_cuda)) for ii in I_obs], dim=0))
                #transDloss_list = []
                #for iii in range(len(I_obs)):
                #    transDloss_list.append(transD.dloss(transD(I_obs[iii], It_pred[iii]), transD(I_obs[iii], I_gt[iii])))
                #transDloss = torch.sum(torch.stack(transDloss_list), dim=0)
                #transDloss = torch.sum(torch.stack([transD.dloss(transD(ii, It_pred), transD(ii, It_cuda)) for ii in I_obs], dim=0))
                #transDloss = transD.dloss(transD(ii, It_pred), transD())
                #refDloss.backward(retain_graph=True)
                #optimizer_refD.step()
                #transDloss.backward(retain_graph=True)
                #optimizer_transD.step()

                if idx % 1 == 0:
                    #ps = '%d_%d tot:%.6f, lpolar:%.6f, lref:%.6f, transD:%.6f, transG:%.12f\n'
                    ps = '%d_%d tot:%.6f, lpolar:%.6f, lref:%.6f vgg:%.6f\n'
                    print(ps % (i, idx, total_loss.item(), loss_polar.item(), loss_refine.item(), loss_vgg.item()))

                if save_image and idx % 1 == 0:
                    subfoldername = str(folder)
                    if not os.path.exists(os.path.join(ckptpath, str(i))):
                        os.makedirs(os.path.join(ckptpath, str(i)))
                    if not os.path.exists(os.path.join(ckptpath, str(i), subfoldername)):
                        os.makedirs(os.path.join(ckptpath, str(i), subfoldername))
                    #It_refine, It_pred1, It_pred2, It_pred3, It_pred4

                    Itpath = os.path.join(ckptpath, str(i), subfoldername, str(iter) + idp + '_refine.png')
                    if not os.path.exists(Itpath):
                        vutils.save_image(It_refine.clone(), Itpath, normalize=False)

                    Itpath = os.path.join(ckptpath, str(i), subfoldername, str(iter) + idp + '_pred1.png')
                    if not os.path.exists(Itpath):
                        vutils.save_image(It_pred1.clone(), Itpath, normalize=False)

                    Itpath = os.path.join(ckptpath, str(i), subfoldername, str(iter) + idp + '_pred2.png')
                    if not os.path.exists(Itpath):
                        vutils.save_image(It_pred2.clone(), Itpath, normalize=False)

                    Itpath = os.path.join(ckptpath, str(i), subfoldername, str(iter) + idp + '_pred3.png')
                    if not os.path.exists(Itpath):
                        vutils.save_image(It_pred3.clone(), Itpath, normalize=False)

                    Itpath = os.path.join(ckptpath, str(i), subfoldername, str(iter) + idp + '_pred4.png')
                    if not os.path.exists(Itpath):
                        vutils.save_image(It_pred4.clone(), Itpath, normalize=False)

                    # output Ir here
                    Irpath = os.path.join(ckptpath, str(i), subfoldername, str(iter) + idp + '_refine_r.png')
                    if not os.path.exists(Irpath):
                        vutils.save_image(Ir_refine.clone(), Irpath, normalize=False)

                    Irpath = os.path.join(ckptpath, str(i), subfoldername, str(iter) + idp + '_pred1_r.png')
                    if not os.path.exists(Irpath):
                        vutils.save_image(Ir_pred1.clone(), Irpath, normalize=False)

                    Irpath = os.path.join(ckptpath, str(i), subfoldername, str(iter) + idp + '_pred2_r.png')
                    if not os.path.exists(Irpath):
                        vutils.save_image(Ir_pred2.clone(), Irpath, normalize=False)

                    Irpath = os.path.join(ckptpath, str(i), subfoldername, str(iter) + idp + '_pred3_r.png')
                    if not os.path.exists(Irpath):
                        vutils.save_image(Ir_pred3.clone(), Irpath, normalize=False)

                    Irpath = os.path.join(ckptpath, str(i), subfoldername, str(iter) + idp + '_pred4_r.png')
                    if not os.path.exists(Irpath):
                        vutils.save_image(Ir_pred4.clone(), Irpath, normalize=False)

        if save_epoch:
            savepathi = os.path.join(ckptpath, str(i))
            if not os.path.exists(savepathi):
                os.makedirs(savepathi)
            torch.save(sepnet.state_dict(), os.path.join(savepathi, polar_name))
            #torch.save(refD.state_dict(), os.path.join(savepathi, refD_name))
            #torch.save(transD.state_dict(), os.path.join(savepathi, transD_name))


def test_sepmodel_4input_selected():
    datapath = '/home/lir0b/data/polar/realdata/feb23'

    ckptpath = '/home/lir0b/Code/TransparenceDetection/src/pid/ckpt/1.00_0.00_0.00_1582382087_strong_ref/99'

    use_CUDA = True
    device = torch.device("cuda:0" if use_CUDA else "cpu")

    polar_name = 'SepNet'
    psf_name = 'PSFNet'
    discriminator_name = 'netD'
    refD_name = 'refD'
    transD_name = 'transD'

    sepnet = Sepmodelv3().to(device)
    optimizerG = optim.Adam(sepnet.parameters(), lr=0.0002, betas=(0.5, 0.999))
    refD = RefDiscriminator()
    transD = TransDiscriminator()
    refD.to(device)
    transD.to(device)
    refD.apply(weights_init)
    transD.apply(weights_init)
    optimizer_refD = optim.Adam(refD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_transD = optim.Adam(transD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    weights = [1.0, 0, 0, 3, 1]
    brightness = 1
    rsz = (320, 320)

    sepnet.load_state_dict(torch.load(os.path.join(ckptpath, polar_name)))
    refD.load_state_dict(torch.load(os.path.join(ckptpath, refD_name)))
    transD.load_state_dict(torch.load(os.path.join(ckptpath, transD_name)))
    print(sepnet)

    #filename = 'LUCID_PHX050S-Q_190100163__20200223222805374_image0'
    filename = 'LUCID_PHX050S-Q_190100163__20200223224413060_image0'
    imgpath1 = os.path.join(datapath, filename+'_0d.png')
    imgpath2 = os.path.join(datapath, filename+'_45d.png')
    imgpath3 = os.path.join(datapath, filename+'_90d.png')
    imgpath4 = os.path.join(datapath, filename+'_135d.png')

    img1 = preprocess_image(imgpath1, rsz, brightness)
    img2 = preprocess_image(imgpath2, rsz, brightness)
    img3 = preprocess_image(imgpath3, rsz, brightness)
    img4 = preprocess_image(imgpath4, rsz, brightness)

    img1_cuda = to_CUDA(img1)
    img2_cuda = to_CUDA(img2)
    img3_cuda = to_CUDA(img3)
    img4_cuda = to_CUDA(img4)
    I_obs = [img1_cuda, img2_cuda, img3_cuda, img4_cuda]

    sepnet.zero_grad()
    [Ir_pred, It_pred] = sepnet(img1_cuda, img2_cuda, img3_cuda, img4_cuda)

    outputpath = '/home/lir0b/Code/TransparenceDetection/exp/realdata'
    Irpath_pred = os.path.join(outputpath, filename, filename+'_r_pred.jpg')
    os.makedirs(os.path.join(outputpath, filename))
    if not os.path.exists(Irpath_pred):
        vutils.save_image(Ir_pred.clone(), Irpath_pred, normalize=False)
    Itpath_pred = os.path.join(outputpath, filename, filename+'_t_pred.jpg')
    #os.makedirs(os.path.join(outputpath, filename))
    if not os.path.exists(Itpath_pred):
        vutils.save_image(It_pred.clone(), Itpath_pred, normalize=False)


def pil2grid4x4(I=[]):
    pass


def grid4x42pil(grid=[]):
    blk = []
    for i in range(len(grid)):
        row1 = torch.cat([grid[i][0], grid[i][1]], dim=3)
        row2 = torch.cat([grid[i][2], grid[i][3]], dim=3)
        blk1 = torch.cat([row1, row2], dim=2)
        blk.append(blk1)
    row1 = torch.cat([blk[0], blk[1]], dim=3)
    row2 = torch.cat([blk[2], blk[3]], dim=3)
    img_tensor = torch.cat([row1, row2], dim=2)
    #print(img_tensor.shape)
    #img_pil = torchvision.transforms.functional.to_pil_image(img_tensor)
    return img_tensor


def train_sepmodel_realdata_4x4():
    use_CUDA = True
    from dataloader import DataLoader
    from pprint import pprint
    import numpy as np
    gtpath, simupath, ckptpath = define_path()

    gtlist = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]

    if True:
        gtpath = '/home/lir0b/data/polar/realdata/dataset/gt'
        polarpath = '/home/lir0b/data/polar/realdata/dataset/polar'
        ckptpath = '/home/lir0b/Code/TransparenceDetection/src/pid/ckpt'
    else:
        gtpath = '/ibex/scratch/lir0b/data/dataset/gt'
        polarpath = '/ibex/scratch/lir0b/data/dataset/polar'
        ckptpath = '/ibex/scratch/lir0b/TransparenceDetection/src/pid/ckpt'


    polar_name = 'SepNet'
    psf_name = 'PSFNet'
    discriminator_name = 'netD'
    refD_name = 'refD'
    transD_name = 'transD'
    save_epoch = True
    continue_train = False
    save_image = True

    device = torch.device("cuda:0" if use_CUDA else "cpu")
    sepnet = Sepmodelv3().to(device)
    #optimizerG = optim.Adam(sepnet.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(sepnet.parameters(), lr=0.0001, betas=(0.5, 0.999))
    epoch = 400
    weights = [1.0, 0, 0, 3, 1]
    rsz = (512, 512)
    box = [0, 0, 1024, 1024]
    brightness = 1
    if continue_train:
        ckptpath = os.path.join(ckptpath, '1.00_0.00_0.00_1582839792')
    else:
        ckptpath = os.path.join(ckptpath, '%.2f_%.2f_%.2f_%d' % (weights[0], weights[1], weights[2], int(time.time())))

    prefix = 'LUCID_PHX050S-Q_190100163__'
    suffix = '_image0_'
    for i in range(epoch):
        savepathi = os.path.join(ckptpath, str(i), polar_name)
        if os.path.exists(savepathi) and continue_train:
            savepathi_1 = os.path.join(ckptpath, str(i + 1), polar_name)
            if not os.path.exists(savepathi_1):
                sepnet.load_state_dict(torch.load(os.path.join(ckptpath, str(i), polar_name)))
            continue

        #for idx, folder in enumerate(gtlist):
        numsample = 300
        for idx in range(numsample):
            iter = random.randint(0, numsample) % len(gtlist)
            folder = gtlist[iter]
            curgtlist = glob.glob(os.path.join(gtpath, str(folder), '*.png'))
            curpolarlist = glob.glob(os.path.join(polarpath, str(folder), '*.png'))
            print(curpolarlist)
            tmp = [os.path.split(tt)[1] for tt in curpolarlist]
            tmp2 = [tt.split('_') for tt in tmp]
            idlist = [tt[4] for tt in tmp2]
            #pprint(list(np.unique(idlist)))
            idlist = list(np.unique(idlist))
            #'LUCID_PHX050S-Q_190100163__20200224191335776_image0_0d.png'
            polarlist = []

            # read gt image here
            Igt0 = preprocess_image_realdata_grid4x4(curgtlist[0])
            Igt45 = preprocess_image_realdata_grid4x4(curgtlist[1])
            Igt90 = preprocess_image_realdata_grid4x4(curgtlist[2])
            Igt135 = preprocess_image_realdata_grid4x4(curgtlist[3])

            # read polarized images here
            randidx = random.randint(0, len(idlist))
            idp = str(idlist[randidx%len(idlist)])

            path0 = os.path.join(polarpath, str(folder), prefix + idp + suffix + '0d.png')
            path45 = os.path.join(polarpath, str(folder), prefix + idp + suffix + '45d.png')
            path90 = os.path.join(polarpath, str(folder), prefix + idp + suffix + '90d.png')
            path135 = os.path.join(polarpath, str(folder), prefix + idp + suffix + '135d.png')

            I0 = preprocess_image_realdata_grid4x4(path0)
            I45 = preprocess_image_realdata_grid4x4(path45)
            I90 = preprocess_image_realdata_grid4x4(path90)
            I135 = preprocess_image_realdata_grid4x4(path135)

            idij = [(i, j) for i in range(4) for j in range(4)]
            #[It_refine, It_pred1, It_pred2, It_pred3, It_pred4]
            #Ir_pred = [Igt0ij - It_pred1, Igt45ij - It_pred2, Igt90ij - It_pred3, Igt135ij - It_pred4]
            It_refine_list = [[], [], [], []]
            It_pred1_list = [[], [], [], []]
            It_pred2_list = [[], [], [], []]
            It_pred3_list = [[], [], [], []]
            It_pred4_list = [[], [], [], []]
            for ij in idij:
                ii, jj = ij
                Igt0ij = to_CUDA(Igt0[ii][jj])
                Igt45ij = to_CUDA(Igt45[ii][jj])
                Igt90ij = to_CUDA(Igt90[ii][jj])
                Igt135ij = to_CUDA(Igt135[ii][jj])
                I_gt = [Igt0ij, Igt45ij, Igt90ij, Igt135ij]
                It_gt = 0.25 * (Igt0ij + Igt45ij + Igt90ij + Igt135ij)

                I0ij = to_CUDA(I0[ii][jj])
                I45ij = to_CUDA(I45[ii][jj])
                I90ij = to_CUDA(I90[ii][jj])
                I135ij = to_CUDA(I135[ii][jj])


                sepnet.zero_grad()
                [It_refine, It_pred1, It_pred2, It_pred3, It_pred4] = sepnet(I0ij, I45ij, I90ij, I135ij)

                It_refine_list[ii].append(It_refine)
                It_pred1_list[ii].append(It_pred1)
                It_pred2_list[ii].append(It_pred2)
                It_pred3_list[ii].append(It_pred3)
                It_pred4_list[ii].append(It_pred4)

                It_pred = [It_pred1, It_pred2, It_pred3, It_pred4]
                Ir_pred = [Igt0ij - It_pred1, Igt45ij - It_pred2, Igt90ij - It_pred3, Igt135ij - It_pred4]
                [loss_polar, loss_refine, loss_vgg, total_loss] = sepnet.loss(It_refine, It_gt, It_pred, I_gt)

                total_loss.backward(retain_graph=False)
                optimizerG.step()

                if idx % 1 == 0:
                    # ps = '%d_%d tot:%.6f, lpolar:%.6f, lref:%.6f, transD:%.6f, transG:%.12f\n'
                    ps = '%d_%d tot:%.6f, lpolar:%.6f, lref:%.6f vgg:%.6f\n'
                    print(ps % (i, idx, total_loss.item(), loss_polar.item(), loss_refine.item(), loss_vgg.item()))


            It_refine = grid4x42pil(It_refine_list)
            It_pred1 = grid4x42pil(It_pred1_list)
            It_pred2 = grid4x42pil(It_pred2_list)
            It_pred3 = grid4x42pil(It_pred3_list)
            It_pred4 = grid4x42pil(It_pred4_list)

            if save_image and idx % 1 == 0:
                subfoldername = str(folder)
                if not os.path.exists(os.path.join(ckptpath, str(i))):
                    os.makedirs(os.path.join(ckptpath, str(i)))
                if not os.path.exists(os.path.join(ckptpath, str(i), subfoldername)):
                    os.makedirs(os.path.join(ckptpath, str(i), subfoldername))
                # It_refine, It_pred1, It_pred2, It_pred3, It_pred4

                #savename = str(iter) + idp + '_' + str(ii) + '_' + str(jj) + '_refine.png'
                savename = str(iter) + '_' + idp + '_refine.png'
                Itpath = os.path.join(ckptpath, str(i), subfoldername, savename)
                if not os.path.exists(Itpath):
                    vutils.save_image(It_refine.clone(), Itpath, normalize=False)

                #savename = str(iter) + idp + '_' + str(ii) + '_' + str(jj) + '_pred1.png'
                savename = str(iter) + '_' + idp + '_pred1.png'
                Itpath = os.path.join(ckptpath, str(i), subfoldername, savename)
                if not os.path.exists(Itpath):
                    vutils.save_image(It_pred1.clone(), Itpath, normalize=False)

                #savename = str(iter) + idp + '_' + str(ii) + '_' + str(jj) + '_pred2.png'
                savename = str(iter) + '_' + idp + '_pred2.png'
                Itpath = os.path.join(ckptpath, str(i), subfoldername, savename)
                if not os.path.exists(Itpath):
                    vutils.save_image(It_pred2.clone(), Itpath, normalize=False)

                #savename = str(iter) + idp + '_' + str(ii) + '_' + str(jj) + '_pred3.png'
                savename = str(iter) + '_' + idp + '_pred3.png'
                Itpath = os.path.join(ckptpath, str(i), subfoldername, savename)
                if not os.path.exists(Itpath):
                    vutils.save_image(It_pred3.clone(), Itpath, normalize=False)

                #savename = str(iter) + idp + '_' + str(ii) + '_' + str(jj) + '_pred4.png'
                savename = str(iter) + '_' + idp + '_pred4.png'
                Itpath = os.path.join(ckptpath, str(i), subfoldername, savename)
                if not os.path.exists(Itpath):
                    vutils.save_image(It_pred4.clone(), Itpath, normalize=False)

        if save_epoch:
            savepathi = os.path.join(ckptpath, str(i))
            if not os.path.exists(savepathi):
                os.makedirs(savepathi)
            torch.save(sepnet.state_dict(), os.path.join(savepathi, polar_name))


if __name__ == '__main__':
    #train_sepmodel_singlenet()
    #test_translate_conv()
    #train_sepmodel_selected()
    #train_sepmodel_4input_selected()
    #test_sepmodel_4input_selected()
    #train_sepmodel_realdata()
    #train_sepmodel_realdata_4x4()
    #train_sepmodel_realdata()
    #train_sepmodel_syntheticdata()
    test_sepmodel_data()
