import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from PIL import ImageEnhance


def preprocess_image(img):
    pass


class OpticsUtils:
    def __init__(self, I0, I45, I90, I135):
        '''
        :param Input: I0, I45, I90, I135 should be in the range of [0,1]
        '''
        self.I0 = I0
        self.I45 = I45
        self.I90 = I90
        self.I135 = I135

    def calc_phi(self):
        phi = 0.5*np.arctan(np.divide(self.I0+self.I90-2*self.I45, self.I0-self.I90+1e-3))
        phi[phi < -1 * (np.pi / 4)] = phi[phi < -1 * (np.pi / 4)] + 0.5 * np.pi
        phi[phi > (np.pi / 4)] = phi[phi > (np.pi / 4)] - 0.5 * np.pi
        self.phi = phi
        return phi

    def calc_perp_para(self):
        I_perp = 0.5*(self.I0+self.I90)+0.5*np.divide((self.I0-self.I90), np.cos(2*self.phi))
        I_para = 0.5*(self.I0+self.I90)-0.5*np.divide((self.I0-self.I90), np.cos(2*self.phi))
        self.I_perp = I_perp
        self.I_para = I_para
        return I_perp, I_para


def output_vggfeat():
    import torchvision
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import matplotlib.pylab as pll
    # define vgg net features
    vgg19 = torchvision.models.vgg19(pretrained=True)
    device = torch.device("cuda:0")
    vgg19f = vgg19.features.to(device)
    for param in vgg19.parameters():
        param.requires_grad = False
    print(vgg19f)
    vgg19f1 = vgg19f[:3]
    vgg19f2 = vgg19f[6:11]
    vgg19f3 = vgg19f[11:20]
    vgg19f4 = vgg19f[13:18]
    vgg19f5 = vgg19f[18:]

    # loading image
    imgpath = '/home/lir0b/Code/TransparenceDetection/src/perceptual-reflection-removal/synthetic/reflection_layer/23.jpg'
    img = Image.open(imgpath)
    imgin = ToTensor()(img).unsqueeze(0).cuda()
    h = imgin.shape[2]
    w = imgin.shape[3]
    out1 = vgg19f1(imgin)
    #out2 = vgg19f2(out1)
    #out3 = vgg19f3(out2)
    #out4 = vgg19f4(out3)
    #out5 = vgg19f5(out4)
    #print(out1.shape)

    vgg19f1 = vgg19f[:3]
    vgg19f2 = vgg19f[3:8]
    vgg19f3 = vgg19f[8:13]
    vgg19f4 = vgg19f[13:24]
    vgg19f5 = vgg19f[24:34]
    out1 = vgg19f1(imgin)
    out2 = vgg19f2(out1)
    out3 = vgg19f3(out2)
    out4 = vgg19f4(out3)
    out5 = vgg19f5(out4)

    nd = out1.shape[1]
    img1 = out1[0, 16, :, :]
    img2 = out2[0, 16, :, :]
    img3 = out3[0, 16, :, :]
    img4 = out4[0, 16, :, :]
    img5 = out5[0, 16, :, :]
    # print(imgi.shape)
    plt.imshow(img1.cpu().numpy())
    plt.show()
    plt.imshow(img2.cpu().numpy())
    plt.show()
    plt.imshow(img3.cpu().numpy())
    plt.show()
    plt.imshow(img4.cpu().numpy())
    plt.show()
    plt.imshow(img5.cpu().numpy())
    plt.show()
    feat = [out1, out2, out3, out4, out5]
    import os
    savepath = '/home/lir0b/Code/TransparenceDetection/src/pid/exp/vggfeat'
    for idx, f in enumerate(feat):
        if idx == 1 or idx == 2:
            pass
        nd = f.shape[1]
        for i in range(nd):
            imgi = f[0, i, :, :]
            print('min:%.2f max:%.2f'%(torch.min(imgi), torch.max(imgi)))
            #print(imgi.shape)
            #fig = plt.imshow(imgi.cpu().numpy())
            #plt.savefig(os.path.join(savepath, '%d_%d.png'%(idx, i)))
            #pll.savefig(os.path.join(savepath, '%d_%d.png'%(idx, i)))
            mpimg.imsave(os.path.join(savepath, '%d_%d.png'%(idx, i)), imgi.cpu())
            #plt.close(fig)
            #print(vgg19f)
        #fig, axarr = plt.subplots(1, 1)
        #axarr.imshow(img.detach().numpy())
    #print(out2.shape)
    #print(out3.shape)
    #print(out4.shape)
    #print(out5.shape)
    #return [out1, out2, out3, out4, out5]


if __name__ == '__main__':
    #output_vggfeat()
    imgpath = '/media/lir0b/关羽/Simulation_RT/38_38_10/S0_theta_85_fi_0.jpg'
    img = Image.open(imgpath)
    enhc = ImageEnhance.Contrast(img)
    enhb = ImageEnhance.Brightness(img)
    enhcolor = ImageEnhance.Color(img)
    imgb = enhb.enhance(7)
    imgb.show(title='Brightness')
    #imgc = enhc.enhance(3.5)
    #imgc.show(title='Contrast')
