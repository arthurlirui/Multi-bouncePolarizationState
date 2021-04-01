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
from torchvision.transforms.functional import center_crop


if __name__ == '__main__':
    path = '/home/lir0b/Code/TransparenceDetection/draft_eccv/figure/ablation/row1'
    import glob
    import numpy as np
    import cv2
    #fl = glob.glob(os.path.join(path, '*.png'))
    if False:
        img0 = Image.open(os.path.join(path, 'LUCID_PHX050S-Q_190100163__20200225214309862_image0_0d.png'))
        img1 = Image.open(os.path.join(path, 'LUCID_PHX050S-Q_190100163__20200225214309862_image0_45d.png'))
        img2 = Image.open(os.path.join(path, 'LUCID_PHX050S-Q_190100163__20200225214309862_image0_90d.png'))
        img3 = Image.open(os.path.join(path, 'LUCID_PHX050S-Q_190100163__20200225214309862_image0_135d.png'))
        img0 = center_crop(img0, (1024, 1024))
        img1 = center_crop(img1, (1024, 1024))
        img2 = center_crop(img2, (1024, 1024))
        img3 = center_crop(img3, (1024, 1024))

        img0 = np.asarray(img0).astype(float)
        img1 = np.asarray(img1).astype(float)
        img2 = np.asarray(img2).astype(float)
        img3 = np.asarray(img3).astype(float)

        img_tot = 0.25*(img0+img1+img2+img3)
        cv2.imwrite('/home/lir0b/Code/TransparenceDetection/draft_eccv/figure/ablation/row1/tot.png', img_tot)

    if True:
        img0 = Image.open(os.path.join(path, 'LUCID_PHX050S-Q_190100163__20200225214309862_image0_0d.png'))
        img1 = Image.open(os.path.join(path, 'LUCID_PHX050S-Q_190100163__20200225214309862_image0_45d.png'))
        img2 = Image.open(os.path.join(path, 'LUCID_PHX050S-Q_190100163__20200225214309862_image0_90d.png'))
        img3 = Image.open(os.path.join(path, 'LUCID_PHX050S-Q_190100163__20200225214309862_image0_135d.png'))
        img0 = center_crop(img0, (1024, 1024))
        img1 = center_crop(img1, (1024, 1024))
        img2 = center_crop(img2, (1024, 1024))
        img3 = center_crop(img3, (1024, 1024))

        img0 = np.asarray(img0).astype(float)
        img1 = np.asarray(img1).astype(float)
        img2 = np.asarray(img2).astype(float)
        img3 = np.asarray(img3).astype(float)

        img_tot = 0.25 * (img0 + img1 + img2 + img3)
        cv2.imwrite('/home/lir0b/Code/TransparenceDetection/draft_eccv/figure/ablation/row1/tot.png', img_tot)


    # for idx, f in enumerate(fl):
    #     print(f)
    #     img = Image.open(f)
    #     img = center_crop(img, (1024, 1024))
    #     img512 = img.resize((512, 512))
    #     filename = f.split('/')[-1]
    #     img512.save(os.path.join(path, str(idx+1)+'_rsz.png'))