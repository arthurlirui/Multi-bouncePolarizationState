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
from pidmodel import Intmodel, Discriminator, Sepmodel, weights_init
import glob


def train_Intnet():
    use_CUDA = True
    from dataloader import DataLoader
    from pprint import pprint
    #gtpath = '/home/lir0b/data/polar/poldata'
    #simupath = '/home/lir0b/data/polar/poldata'
    ckptpath = '/home/lir0b/Code/TransparenceDetection/src/pid/ckpt_Intnet/'
    #if not os.path.exists(gtpath) or not os.path.exists(simupath) or not os.path.exists(ckptpath):
        #gtpath = '/ibex/scratch/lir0b/data/synthetic'
        #simupath = '/ibex/scratch/lir0b/data/Simulation_RT'
        #ckptpath = '/ibex/scratch/lir0b/TransparenceDetection/src/pid/ckpt'

    datapath = '/home/lir0b/data/polar/poldata'
    if not os.path.exists(datapath):
        datapath = '/ibex/scratch/lir0b/data/poldata'
        ckptpath = '/ibex/scratch/lir0b/TransparenceDetection/src/pid/ckpt_Intnet'
    folderlist = glob.glob(os.path.join(datapath, '*'))
    data = []
    name = [0, 45, 90, 135]
    for f in folderlist:
        foldername = os.path.split(f)[-1]
        filelist = []
        for n in name:
            filename = foldername+'_'+str(n)+'.png'
            filelist.append(os.path.join(f, filename))
        data.append(filelist)

    modelname = 'IntNetG'
    dname = 'IntNetD'
    Epochs = 1000
    t = 0
    save_epoch = True
    continue_train = False
    save_image = True
    device = torch.device("cuda:0" if use_CUDA else "cpu")

    intnetG = Intmodel().to(device)
    intnetG.apply(weights_init)
    print(intnetG)
    optimizerG = optim.Adam(intnetG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    #intnetD = Discriminator()
    #intnetD.to(device)
    #intnetD.apply(weights_init)
    #optimizerD = optim.Adam(intnetD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    t = 0
    sz = (256, 256)
    loss_weights = [1.5, 1, 1, 1, 1]
    subfolder = '%.2f_%.2f_%.2f_%.2f_%.2f'%(loss_weights[0], loss_weights[1], loss_weights[2], loss_weights[3], loss_weights[4])
    ckptpath = os.path.join(ckptpath, subfolder)
    if not os.path.exists(ckptpath):
        os.makedirs(ckptpath)
    # loading model
    is_loading = False
    for i in range(Epochs):
        savepathi = os.path.join(ckptpath, str(i), modelname)
        savepathi_1 = os.path.join(ckptpath, str((int(i / 100) + 1) * 100), modelname)
        if os.path.exists(savepathi) and continue_train and not os.path.exists(savepathi_1):
            intnetG.load_state_dict(torch.load(savepathi))
            is_loading = True

        if continue_train and not is_loading:
            continue

        for fl in data:
            I0 = Image.open(fl[0]).resize(sz)
            I1 = Image.open(fl[1]).resize(sz)
            I2 = Image.open(fl[2]).resize(sz)
            I3 = Image.open(fl[3]).resize(sz)
            I0 = ToTensor()(I0).unsqueeze(0).cuda()
            I1 = ToTensor()(I1).unsqueeze(0).cuda()
            I2 = ToTensor()(I2).unsqueeze(0).cuda()
            I3 = ToTensor()(I3).unsqueeze(0).cuda()
            print(I0.shape)
            target = torch.cat((I0, I1, I2, I3), dim=1)
            intnetG.zero_grad()
            for ii in range(5):
                intnetG.zero_grad()
                [D, S, P0, P1, P2, P3] = intnetG(I0, I1, I2, I3)
                [Lrec, Ld, Ls, Lsparse, Ldyn] = intnetG.loss([I0, I1, I2, I3],
                                                             [D, S, P0, P1, P2, P3], weights=loss_weights)
                # [loss_sep, loss_vgg_sep, loss_exclu] = allloss
                total_loss = Lrec + Ld + Ls + Lsparse + Ldyn
                total_loss.backward()
                optimizerG.step()
                G_x = total_loss.item()

                print('%d_%d_%d A:%.2f Lrec:%.6f Ld:%.6f Ls:%.6f Lspa:%.6f Ldyn:%.6f\n' % (i, ii, t, total_loss.item(),
                                                                                           Lrec.item(),
                                                                                           Ld.item(),
                                                                                           Ls.item(),
                                                                                           Lsparse.item(),
                                                                                           Ldyn.item()))

            #
            t = t + 1
            if t % 97 == 0:
                if not os.path.exists(os.path.join(ckptpath, str(i))):
                    os.makedirs(os.path.join(ckptpath, str(i)))
                filename = os.path.split(fl[0])[0].split('/')[-1]
                Dpath = os.path.join(ckptpath, str(i), filename + '_D.jpg')
                print(Dpath)
                vutils.save_image(D.detach(), Dpath, normalize=True)
                Spath = os.path.join(ckptpath, str(i), filename + '_S.jpg')
                vutils.save_image(S.detach(), Spath, normalize=True)
                P0path = os.path.join(ckptpath, str(i), filename + '_P0.jpg')
                vutils.save_image(P0.detach(), P0path, normalize=True)
                P1path = os.path.join(ckptpath, str(i), filename + '_P1.jpg')
                vutils.save_image(P1.detach(), P1path, normalize=True)
                P2path = os.path.join(ckptpath, str(i), filename + '_P2.jpg')
                vutils.save_image(P2.detach(), P2path, normalize=True)
                P3path = os.path.join(ckptpath, str(i), filename + '_P3.jpg')
                vutils.save_image(P3.detach(), P3path, normalize=True)
        if save_epoch and i % 100 == 0:
            savepathi = os.path.join(ckptpath, str(i))
            if not os.path.exists(savepathi):
                os.makedirs(savepathi)
            torch.save(intnetG.state_dict(), os.path.join(savepathi, modelname))
            #torch.save(intnetD.state_dict(), os.path.join(savepathi, dname))


if __name__ == '__main__':
    train_Intnet()

    if False:
        import glob
        datapath = '/home/lir0b/data/polar/poldata'
        folderlist = glob.glob(os.path.join(datapath, '*'))
        data = []
        name = [0, 45, 90, 135]
        for f in folderlist:
            foldername = os.path.split(f)[-1]
            filelist = []
            for n in name:
                filename = foldername+'_'+str(n)+'.png'
                filelist.append(os.path.join(f, filename))
            data.append(filelist)
        print(data)
        #self.translist = [f for f in glob.glob(os.path.join(self.transpath, '*.jpg'))]