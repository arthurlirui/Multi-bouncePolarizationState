import cv2
import os
import numpy as np


if __name__ == '__main__':
    rootpath = '/home/lir0b/data/polar/nips2019'
    rawpath = os.path.join(rootpath, 'polar_imgs')
    gtpath = os.path.join(rootpath, 'gt')
    polarpath = os.path.join(rootpath, 'polar')
    #savepath = '/home/lir0b/data/polar/nips2019/gt'
    folder_list = os.listdir(rawpath)
    for folder in folder_list:
        print(folder)
        flist = os.listdir(os.path.join(rawpath, folder))
        if not os.path.exists(os.path.join(polarpath, folder)):
            os.mkdir(os.path.join(polarpath, folder))
        if not os.path.exists(os.path.join(gtpath, folder)):
            os.mkdir(os.path.join(gtpath, folder))
        angles = ['000', '045', '090', '135']
        output_angles = ['0d', '45d', '90d', '135d']
        namedict = {'000': '0d', '045': '45d', '090': '90d', '135': '135d'}
        if True:
            for f in flist:
                print(f)
                fname = f.split('.')[0]
                fnl = fname.split('_')
                img = cv2.imread(os.path.join(rawpath, folder, f))
                cv2.imwrite(os.path.join(polarpath, folder, f'{folder}_{namedict[fnl[2]]}.png'), img)

        if True:
            images = []
            for i in range(len(output_angles)):
                fname = folder+f'_{output_angles[i]}.png'
                img = cv2.imread(os.path.join(polarpath, folder, fname))
                #print(img.shape)
                images.append(img)
            img3d = np.stack(images, axis=3)
            img_min = np.min(img3d, axis=3)
            print(len(images), img_min.shape)
            #img_min = np.min(images)
            #print(img_min.shape)
            for i in range(len(output_angles)):
                folderpath = os.path.join(gtpath, folder, folder+f'_{output_angles[i]}.png')
                cv2.imwrite(folderpath, img_min)