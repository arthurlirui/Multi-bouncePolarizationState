import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
import time
import math
from PIL import Image

from pid.models.pldmodel import PolarPhysicalModel
from pid.options.train_options import TrainOptions
from pid.data.data_loader import CreateDataLoader
from pid.models.models import create_model
#from pid.util.visualizer import Visualizer
import pid.util.util as util
from torchvision.transforms import transforms
from torchvision.utils import save_image
from pprint import pprint


def lcm(a, b):
    return abs(a * b) / fractions.gcd(a, b) if a and b else 0


def save_results(inputs=[], savepath=''):
    pass


def init_folder():
    savepath = '/home/lir0b/Code/TransparenceDetection/src/pid/exp'
    ts = time.localtime()
    foldername = 'train_output-%s-%s-%s' % (ts.tm_year, ts.tm_mon, ts.tm_mday, ts.tm_hour)
    #if not test:
    #    foldername = '%s_%s_%s_%s_%s_%s' % (ts.tm_year, ts.tm_mon, ts.tm_mday, ts.tm_hour, ts.tm_min, ts.tm_sec)
    #foldername = 'test'
    print(foldername)
    savepath = os.path.join(savepath, foldername)
    savepath1 = os.path.join(savepath, 'output')
    try:
        os.mkdir(os.path.join(savepath))
        os.mkdir(os.path.join(savepath1))
        return [savepath, savepath1]
    except OSError as error:
        print(error)
        return [savepath, savepath1]


def save_img(S, savepath, savename):
    topil = transforms.ToPILImage()
    try:
        os.mkdir(os.path.join(savepath))
    except OSError as error:
        print(error)
    for ii, s in enumerate(S):
        s[s > 1] = 1
        s_pil = topil(torch.squeeze(s.cpu()))
        s_pil.save(os.path.join(savepath, savename + str(ii) + '.png'))


def train():
    opt = TrainOptions().parse()
    path = '/home/lir0b/data/polar/nips2019'
    #opt.dataroot = '/home/lir0b/data/polar/realdata/dataset'
    opt.dataroot = path
    #savepath = '/home/lir0b/Code/TransparenceDetection/src/pid/exp/test'
    savepath = os.path.join(path, 'exp')

    # continue train model
    iter_path = os.path.join(opt.checkpoints_dir, 'iter.txt')
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0
    opt.isTrain = True
    opt.print_freq = lcm(opt.print_freq, opt.batchSize)

    # is_save_train_data = opt.save_train_data
    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1
        opt.niter = 1
        opt.niter_decay = 0
        opt.max_dataset_size = 10

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    is_sepmodel = True
    if is_sepmodel:
        opt.model = 'sepmodel'
        main_model = create_model(opt)
        #main_model = torch.nn.DataParallel(main_model, device_ids=opt.gpu_ids)
        optimizer = main_model.module.optimizer
        #optimizer_ref = main_model.module.optimizer_ref

    # loading model
    load_model = True
    if load_model:
        epoch = 90
        #main_model.module.load_network(network=main_model.module,
        #                               network_label=main_model.module.name(),
        #                               epoch_label=str(epoch))
        print('loading model')
        main_model.module.load(opt)
    print(opt.gpu_ids)
    if True:
        for epoch in range(start_epoch, opt.max_epoch):
            inputs_save = []
            outputs_save = []
            #if epoch != start_epoch:
            #    epoch_iter = epoch_iter % dataset_size
            for ind, imgs in enumerate(dataset, start=0):
                # training data process
                Igt = imgs['GT']
                I = imgs['I']

                start_time = time.time()
                outputs = main_model(I_obs=I, I_gt=Igt)
                It_ref = outputs['It_refine']
                It_pred = outputs['It_pred']
                Ir_pred = outputs['Ir_pred']
                loss_total = outputs['loss']['total']
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()
                end_time = time.time()
                msg = f'|%d|%d|%.3f|%.2f s' % (epoch, ind, loss_total, (end_time-start_time))
                print(msg)

                if opt.save_train_data:
                    savename_gt = f'%d-%d_gt_' % (epoch, ind)
                    save_img(Igt,
                             savepath=os.path.join(savepath, 'GT'),
                             savename=savename_gt)
                    savename_I = f'%d-%d_I_' % (epoch, ind)
                    save_img(I,
                             savepath=os.path.join(savepath, 'I'),
                             savename=savename_I)
                if opt.save_results:
                    inputs = {}
                    inputs['I'] = [tt.detach().cpu() for tt in I]
                    inputs['GT'] = [tt.detach().cpu() for tt in Igt]
                    inputs_save.append(inputs)

                    outputs = {}
                    outputs['It_refine'] = It_ref.detach().cpu()
                    outputs['It_pred'] = [tt.detach().cpu() for tt in It_pred]
                    outputs['Ir_pred'] = [tt.detach().cpu() for tt in Ir_pred]
                    outputs_save.append(outputs)

            if epoch % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d' % (epoch))
                #main_model.module.save('latest')
                #main_model.module.save_network(network=main_model.module,
                #                               network_label=main_model.module.name(),
                #                               epoch_label=str(epoch),
                #                               gpu_ids=opt.gpu_ids)
                np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')
                main_model.module.save(opt, which_epoch=epoch)

                # save results
                if opt.save_results:
                    main_model.module.save_results(savepath=opt.save_results_path,
                                                   epoch=epoch,
                                                   inputs=inputs_save,
                                                   outputs=outputs_save)


def demo_test_generate_multi_bounce_image():
    opt = TrainOptions().parse()
    opt.dataroot = '/home/lir0b/data/polar/realdata/dataset'
    # opt.dataroot = '/home/lir0b/data/polar/realdata/dataset/part'
    savepath = '/home/lir0b/Code/TransparenceDetection/src/pid/exp/test'
    opt.model = 'simulator'
    opt.isTrain = True

    is_polar_engine = True
    if is_polar_engine:
        opt.model = 'polar_engine'
        simulator = create_model(opt)


    # GT variable
    #weight_mirror = torch.Tensor([0.5]).cuda()
    #theta_in = torch.Tensor([0.1]).cuda()
    #thickness = torch.Tensor([10]).cuda()
    #ref_index = torch.Tensor([1.5]).cuda()
    #simulator.module.init_params(theta_in=0.1, thickness=5, weight_mirror=0.1, ref_index=1.5)

    print('Init Params')
    pprint(simulator.module.get_variables())

    optimizer = simulator.module.optimizer
    # data path
    datapath = '/home/lir0b/Code/TransparenceDetection/exp/simulator_data'
    input_path = os.path.join(datapath, 'input')
    gtpath = os.path.join(datapath, 'GT')
    ourpath = os.path.join(datapath, 'our')
    trans = transforms.Compose([transforms.ToTensor()])
    bfunc = lambda x: torch.unsqueeze(x, dim=0)
    tocuda = lambda x: x.cuda()

    # generated images
    filename = '1'
    I0_gt = Image.open(os.path.join(gtpath, filename + '_' + '0d.png'))
    I45_gt = Image.open(os.path.join(gtpath, filename + '_' + '45d.png'))
    I90_gt = Image.open(os.path.join(gtpath, filename + '_' + '90d.png'))
    I135_gt = Image.open(os.path.join(gtpath, filename + '_' + '135d.png'))
    I_gt = [I0_gt, I45_gt, I90_gt, I135_gt]
    I_gt = [tocuda(bfunc(trans(t))) for t in I_gt]
    Ir = 0.25*(I_gt[0]+I_gt[1]+I_gt[2]+I_gt[3])

    filename = '4'
    I0_gt = Image.open(os.path.join(gtpath, filename + '_' + '0d.png'))
    I45_gt = Image.open(os.path.join(gtpath, filename + '_' + '45d.png'))
    I90_gt = Image.open(os.path.join(gtpath, filename + '_' + '90d.png'))
    I135_gt = Image.open(os.path.join(gtpath, filename + '_' + '135d.png'))
    I_gt = [I0_gt, I45_gt, I90_gt, I135_gt]
    I_gt = [tocuda(bfunc(trans(t))) for t in I_gt]
    It = 0.25 * (I_gt[0] + I_gt[1] + I_gt[2] + I_gt[3])

    if True:
        gt_theta_in = 1.13
        gt_thickness = 9
        gt_weight_mirror = 0.3
        gtvars = simulator.module.calc_params(theta_in=gt_theta_in, thickness=gt_thickness,
                                              weight_mirror=gt_weight_mirror, ref_index=1.5)
        print('GT')
        pprint(gtvars)

        dx = gtvars['dx']
        weight_mirror = gtvars['weight_mirror']
        R_para = gtvars['R_para']
        R_perp = gtvars['R_perp']
        T_para = gtvars['T_para']
        T_perp = gtvars['T_perp']

        [I_para, I_perp, Ir_para, Ir_perp, It_para, It_perp] = simulator.module.polarized_trace_two(Ir=Ir, It=It,
                                                                                                    weight_mirror=weight_mirror,
                                                                                                    dx=dx,
                                                                                                    R_para=R_para,
                                                                                                    R_perp=R_perp,
                                                                                                    T_para=T_para,
                                                                                                    T_perp=T_perp)
        I_obs = I_para + I_perp
        save_image(I_obs, 'I_obs.png')
    else:
        I_obs = Image.open('I_obs.png')
        I_obs = tocuda(bfunc(trans(I_obs)))
    return
    pred_theta_in = 3.14*0.5*torch.rand(1)
    pred_thickness = 10*torch.rand(1)
    pred_weight_mirror = torch.tensor([0.3])
    alpha = 0.01
    l2 = torch.nn.MSELoss(reduction='mean')
    pre_loss = l2(It, I_obs)
    count = 0
    # 1. estimate theta_in
    # 2. estimate thickness
    print('%.3f %.3f %.3f' % (gt_theta_in, gt_thickness, gt_weight_mirror))
    print(pre_loss)
    import numpy as np

    #save_image(I_para, 'I_para.png')
    #save_image(I_perp, 'I_perp.png')
    save_image(Ir_para, 'Ir_para.png')
    save_image(Ir_perp, 'Ir_perp.png')
    save_image(It_para, 'It_para.png')
    save_image(It_perp, 'It_perp.png')
    save_image(Ir, 'Ir.png')
    save_image(It, 'It.png')

    for t in np.linspace(1.0, 1.3, 25):
        pred_theta_in = t
        #print(pred_theta_in)
        for k in np.linspace(8, 12, 25):
            pred_thickness = k
            #print(pred_thickness)
            estvars = simulator.module.calc_params(theta_in=pred_theta_in,
                                                   thickness=pred_thickness,
                                                   weight_mirror=pred_weight_mirror, ref_index=1.5)
            dx = estvars['dx']
            weight_mirror = estvars['weight_mirror']
            R_para = estvars['R_para']
            R_perp = estvars['R_perp']
            T_para = estvars['T_para']
            T_perp = estvars['T_perp']
            #print(estvars)
            [I_para, I_perp, Ir_para, Ir_perp, It_para, It_perp] = simulator.module.polarized_trace_two(Ir=Ir, It=It,
                                                                                                        weight_mirror=weight_mirror,
                                                                                                        dx=dx,
                                                                                                        R_para=R_para,
                                                                                                        R_perp=R_perp,
                                                                                                        T_para=T_para,
                                                                                                        T_perp=T_perp)
            I_out = I_para + I_perp
            #save_image(I_out, 'I_out.png')
            loss = l2(I_out, I_obs)
            #print(loss)
            if loss.item() < pre_loss.item():
                opt_theta_in = pred_theta_in
                opt_thickness = pred_thickness
                pre_loss = loss
                msg = f'%d %.4f T:%.3f D:%.3f gT:%.3f gD:%.3f' % (count, loss.item(), opt_theta_in, opt_thickness, gt_theta_in, gt_thickness)
                print(msg)
                save_image(I_out, 'I_out.png')

            count += 1

    # for i in range(1000):
    #     # update parameters here
    #     new_theta_in = pred_theta_in + 3.14*0.5*alpha * torch.randn(1)
    #     new_thickness = pred_thickness + 10*alpha * torch.randn(1)
    #     new_weight_mirror = pred_weight_mirror + alpha * torch.randn(1)
    #     estvars = simulator.module.calc_params(theta_in=new_theta_in,
    #                                            thickness=new_thickness,
    #                                            weight_mirror=new_weight_mirror, ref_index=1.5)
    #     dx = estvars['dx']
    #     weight_mirror = estvars['weight_mirror']
    #     R_para = estvars['R_para']
    #     R_perp = estvars['R_perp']
    #     T_para = estvars['T_para']
    #     T_perp = estvars['T_perp']
    #     [I_para, I_perp, Ir_para, Ir_perp, It_para, It_perp] = simulator.module.polarized_trace_two(Ir=Ir, It=It,
    #                                                                                                 weight_mirror=weight_mirror,
    #                                                                                                 dx=dx,
    #                                                                                                 R_para=R_para,
    #                                                                                                 R_perp=R_perp,
    #                                                                                                 T_para=T_para,
    #                                                                                                 T_perp=T_perp)
    #     I_out = I_para + I_perp
    #     loss = l2(I_out, I_obs)
    #     if loss.item() < pre_loss.item():
    #         pred_theta_in = new_theta_in
    #         pred_thickness = new_thickness
    #         pred_weight_mirror = new_weight_mirror
    #         pre_loss = loss
    #         print(loss.item(), pred_theta_in, pred_thickness, pred_weight_mirror)
    #     else:
    #         count += 1
    #         if count > 5:
    #             pred_theta_in = new_theta_in
    #             pred_thickness = new_thickness
    #             pred_weight_mirror = new_weight_mirror
    #             print(loss.item(), pred_theta_in, pred_thickness, pred_weight_mirror)



    # for i in range(1000):
    #     results = simulator(Ir=Ir, It=It, I_obs=I_obs)
    #     I_out = results['I']
    #     loss = results['loss']
    #     params = results['params']
    #     msg = f'L:%.8f - T:%.3f, D:%.3f, W: %.3f' % (loss,
    #                                                  simulator.module.theta_in,
    #                                                  simulator.module.thickness,
    #                                                  simulator.module.weight_mirror)
    #     print(msg)
    #     loss.backward(retain_graph=True)
    #     optimizer.step()
    #     predvars = params
    #
    #     #pprint(params)


def demo_phymodel():
    opt = TrainOptions().parse()
    opt.dataroot = '/home/lir0b/data/polar/realdata/dataset'
    # opt.dataroot = '/home/lir0b/data/polar/realdata/dataset/part'
    savepath = '/home/lir0b/Code/TransparenceDetection/src/pid/exp/test'
    opt.model = 'simulator'

    # continue train model
    iter_path = os.path.join(opt.checkpoints_dir, 'iter.txt')
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0
    opt.isTrain = True
    opt.print_freq = lcm(opt.print_freq, opt.batchSize)

    # is_save_train_data = opt.save_train_data
    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1
        opt.niter = 1
        opt.niter_decay = 0
        opt.max_dataset_size = 10

    #data_loader = CreateDataLoader(opt)
    #dataset = data_loader.load_data()
    #dataset_size = len(data_loader)
    #print('#training images = %d' % dataset_size)

    is_sepmodel = False
    if is_sepmodel:
        opt.model = 'sepmodel'
        main_model = create_model(opt)
        # main_model = torch.nn.DataParallel(main_model, device_ids=opt.gpu_ids)
        optimizer = main_model.module.optimizer
        # optimizer_ref = main_model.module.optimizer_ref

    is_polar_engine = True
    if is_polar_engine:
        opt.model = 'polar_engine'
        simulator = create_model(opt)
        optimizer = simulator.module.optimizer

    #print(simulator.module.parameters())
    print(simulator.module.theta_in)
    print(simulator.module.theta_out)
    print(simulator.module.thickness)
    print(simulator.module.weight_mirror)
    #simulator.module.theta_in
    #print(simulator.parameters())
    #for param in simulator.module.parameters():
        #print(param)

    # load data
    datapath = '/home/lir0b/Code/TransparenceDetection/exp/simulator_data'
    input_path = os.path.join(datapath, 'input')
    gtpath = os.path.join(datapath, 'GT')
    ourpath = os.path.join(datapath, 'our')
    filename = '4'
    if True:
        I0_obs = Image.open(os.path.join(input_path, filename+'_'+'0d.png'))
        I45_obs = Image.open(os.path.join(input_path, filename + '_' + '45d.png'))
        I90_obs = Image.open(os.path.join(input_path, filename + '_' + '90d.png'))
        I135_obs = Image.open(os.path.join(input_path, filename + '_' + '135d.png'))
        I_obs = [I0_obs, I45_obs, I90_obs, I135_obs]

        I0_gt = Image.open(os.path.join(gtpath, filename + '_' + '0d.png'))
        I45_gt = Image.open(os.path.join(gtpath, filename + '_' + '45d.png'))
        I90_gt = Image.open(os.path.join(gtpath, filename + '_' + '90d.png'))
        I135_gt = Image.open(os.path.join(gtpath, filename + '_' + '135d.png'))
        I_gt = [I0_gt, I45_gt, I90_gt, I135_gt]

        Ir = Image.open(os.path.join(ourpath, filename + '_' + 'r.png'))
        It = Image.open(os.path.join(ourpath, filename + '_' + 't.png'))

        trans = transforms.Compose([transforms.ToTensor()])
        bfunc = lambda x: torch.unsqueeze(x, dim=0)
        I_obs = [bfunc(trans(t)) for t in I_obs]
        I_gt = [bfunc(trans(t)) for t in I_gt]
        Ir = bfunc(trans(Ir))
        It = bfunc(trans(It))
        I = 0.25*(I_obs[0]+I_obs[1]+I_obs[2]+I_obs[3])
        print(I.shape)

        for i in range(1000):
            outputs = simulator(Ir=Ir, It=It, I_obs=I)
            #print(outputs)
            curloss = outputs['loss']
            curloss.backward(retain_graph=True)
            optimizer.step()
            msg = f'L:%.8f - T:%.3f, D:%.3f, W: %.3f' % (curloss,
                                                         simulator.module.theta_in,
                                                         simulator.module.thickness,
                                                         simulator.module.weight_mirror)
            print(msg)


def calc_ssim_psnr():
    import math
    import numpy as np
    import cv2
    def calculate_psnr(img1, img2):
        # img1 and img2 have range [0, 255]
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def calculate_ssim(img1, img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''
        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')
    path = '/home/lir0b/Code/TransparenceDetection/draft_eccv/figure/ablation'
    gtpath = os.path.join(path, 'gt', '1_0d_1_gt.png')
    Itpath = os.path.join(path, 'row2', '1.png')
    lspath = os.path.join(path, 'row3', '1.png')
    lepath = os.path.join(path, 'row3', '2.png')
    lapath = os.path.join(path, 'row3', '3.png')
    lppath = os.path.join(path, 'row3', '4.png')

    Igt = cv2.imread(gtpath)
    It = cv2.imread(Itpath)
    Ils = cv2.imread(lspath)
    Ile = cv2.imread(lepath)
    Ila = cv2.imread(lapath)
    Ilp = cv2.imread(lppath)

    #print(Igt.shape, It.shape, Ils.shape)

    res = {}
    res['It'] = calculate_ssim(Igt, It)
    res['Ils'] = calculate_ssim(Igt, Ils)
    res['Ile'] = calculate_ssim(Igt, Ile)
    res['Ila'] = calculate_ssim(Igt, Ila)
    res['Ilp'] = calculate_ssim(Igt, Ilp)
    from pprint import pprint
    pprint(res)

    res2 = {}
    res2['It'] = calculate_psnr(Igt, It)
    res2['Ils'] = calculate_psnr(Igt, Ils)
    res2['Ile'] = calculate_psnr(Igt, Ile)
    res2['Ila'] = calculate_psnr(Igt, Ila)
    res2['Ilp'] = calculate_psnr(Igt, Ilp)
    pprint(res2)

    # trans = transforms.Compose([transforms.ToTensor()])
    # Igt = Image.open(gtpath)
    # It = Image.open(Itpath)
    # Ils = Image.open(lspath)
    # Ile = Image.open(lepath)
    # Ila = Image.open(lapath)
    # Ilp = Image.open(lppath)
    #
    # Igt = trans(Igt)
    # It = trans(It)
    # Ils = trans(Ils)
    # Ila = trans(Ila)
    # Ile = trans(Ile)
    # Ilp = trans(Ilp)


if __name__ == '__main__':
    train()
    #demo_phymodel()
    #demo_test_generate_multi_bounce_image()
    #calc_ssim_psnr()
