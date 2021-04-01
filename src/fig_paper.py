
import cv2
from PIL import Image
import os

def ghost_effect():
    folderlist = ['GT', 'input', 'our', 'ERRNet', 'wen2019', 'zhang2018']
    rootpath = '/home/lir0b/Code/TransparenceDetection/draft_eccv/figure/ghost'
    filelist = ['1.png', '3.png']
    boxlist = [[260, 30, 320, 90]]
    if False:
        img = Image.open(os.path.join(rootpath, '1.png'))
        img0 = img.crop([260, 30, 420, 190])
        img0 = img0.resize((256, 256))
        img0.save(os.path.join(rootpath, '1_crop.png'))
    if True:
        img = Image.open(os.path.join(rootpath, '2.jpg'))
        img0 = img.crop([500, 1200, 2500, 3200])
        #img0 = img0.resize((256, 256))
        img0.save(os.path.join(rootpath, '2_crop.png'))


if __name__ == '__main__':
    ghost_effect()