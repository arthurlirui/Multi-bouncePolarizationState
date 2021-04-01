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


def define_path():
    gtpath = '/home/lir0b/Code/perceptual-reflection-removal/synthetic'
    simupath = '/home/lir0b/Code/TransparenceDetection/src/pid/exp/synthetic_multi'
    ckptpath = '/home/lir0b/Code/TransparenceDetection/src/pid/ckpt'
    if not os.path.exists(gtpath) and not os.path.exists(simupath) and not os.path.exists(ckptpath):
        gtpath = '/ibex/scratch/lir0b/data/synthetic'
        simupath = '/ibex/scratch/lir0b/data/synthetic_multi'
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
    sepnet = Sepmodelv2().to(device)
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

                    if save_image and idx % 10 == 0:
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
