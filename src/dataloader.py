import os
import glob
from pprint import pprint


class DataLoader:
    def __init__(self, simpath, gtpath=None, transpath=None, refpath=None,
                 d=10,
                 theta=[10, 40, 60, 70, 80, 85],
                 polangle=[0, 45, 90, 135],
                 prefix='S0',
                 suffix='filter',
                 subfolder='polar'):
        self.simpath = simpath
        self.gtpath = gtpath
        if transpath:
            self.transpath = transpath
        else:
            #self.transpath = os.path.join(gtpath, 'transmission_layer')
            self.transpath = os.path.join(gtpath, 'reflection_layer')
        if refpath:
            self.refpath = refpath
        else:
            self.refpath = os.path.join(gtpath, 'reflection_layer')
        self.translist = [f for f in glob.glob(os.path.join(self.transpath, '*.jpg'))]
        self.reflist = [f for f in glob.glob(os.path.join(self.refpath, '*.jpg'))]
        self.simfolder = os.listdir(self.simpath)
        self.simpair = [(int(f.split('_')[0]), int(f.split('_')[1])) for f in self.simfolder]
        #self.displace = [int(f.split('_')[2]) for f in self.simfolder]
        self.d = d
        self.theta = theta
        self.polangle = polangle
        self.prefix = prefix
        self.suffix = suffix
        self.subfolder = subfolder

    def get_theta_filelist(self, ref_id, trans_id):
        simfolder = str(ref_id) + '_' + str(trans_id)
        simfile_theta = []
        for t in self.theta:
            simname = simfolder + '_theta_' + str(t) + '_' + self.suffix + '.jpg'
            simpath = os.path.join(self.simpath, simfolder, simname)
            simfile_theta.append(simpath)
        return simfile_theta

    def get_theta_pol_filelist(self, ref_id, trans_id):
        refimg = ref_id
        transimg = trans_id
        #print(refimg, transimg)
        #print(os.path.join(dl.refpath, str(refimg) + '.jpg'))
        #print(os.path.join(dl.transpath, str(transimg) + '.jpg'))
        #simfolder = str(refimg) + '_' + str(transimg) + '_' + str(self.d)
        simfolder = str(refimg) + '_' + str(transimg)
        simfile = []
        for t in self.theta:
            simfile_theta = []
            for a in self.polangle:
                #simname = self.prefix + '_theta_' + str(t) + '_phi_' + str(a) + '_' + self.suffix + '.jpg'
                simname = simfolder + '_theta_' + str(t) + '_phi_' + str(a) + '_' + self.suffix + '.jpg'
                simpath = os.path.join(self.simpath, simfolder, self.subfolder, simname)
                simfile_theta.append(simpath)
            simfile.append(simfile_theta)
        return simfile

    def get_ref_trans_pair(self, ref_id, trans_id):
        refpath = os.path.join(self.refpath, str(ref_id)+'.jpg')
        transpath = os.path.join(self.transpath, str(trans_id)+'.jpg')
        return refpath, transpath

    def get_gt_polar_realdata(self):
        pass


def test():
    gtpath = '/home/lir0b/Code/perceptual-reflection-removal/synthetic'
    simupath = '/media/lir0b/关羽/Simulation_RT'
    dl = DataLoader(simupath, gtpath)
    pprint(dl.simpair)
    pprint(dl.simfolder)
    reflist = []
    translist = []
    d = 10
    theta = [10, 40, 60, 70, 80, 85]
    polangle = [0, 45,90, 135]
    prefix = 'S0_theta_'
    for f in dl.simpair:
        refimg = f[0]
        transimg = f[1]
        print(refimg, transimg)
        print(os.path.join(dl.refpath, str(refimg)+'.jpg'))
        print(os.path.join(dl.transpath, str(transimg)+'.jpg'))
        simfolder = str(refimg)+'_'+str(transimg)+'_'+str(d)
        simfile = []
        for t in theta:
            simfile_theta = []
            for a in polangle:
                simname = prefix+str(t)+'_fi_'+str(a)+'.jpg'
                simpath = os.path.join(dl.simpath, simfolder, simname)
                simfile_theta.append(simpath)
            simfile.append(simfile_theta)
        #pprint(simfile)


def check_file():
    gtpath = '/home/lir0b/Code/perceptual-reflection-removal/synthetic'
    simupath = '/media/lir0b/关羽/Simulation_RT'
    dl = DataLoader(simupath, gtpath)
    #pprint(dl.simpair)
    #pprint(dl.simfolder)
    reflist = []
    translist = []
    d = 10
    theta = [10, 40, 60, 70, 80, 85]
    polangle = [0, 45, 90, 135]
    prefix = 'S0_theta_'
    for f in dl.simpair:
        refimg = f[0]
        transimg = f[1]
        #print(refimg, transimg)
        refpath = os.path.join(dl.refpath, str(refimg)+'.jpg')
        transpath = os.path.join(dl.transpath, str(transimg)+'.jpg')
        if not os.path.exists(refpath):
            print(refpath)
        if not os.path.exists(transpath):
            print(transpath)
        simfolder = str(refimg)+'_'+str(transimg)+'_'+str(d)
        simfile = []
        for t in theta:
            simfile_theta = []
            for a in polangle:
                simname = prefix+str(t)+'_fi_'+str(a)+'.jpg'
                simpath = os.path.join(dl.simpath, simfolder, simname)
                if not os.path.exists(simpath):
                    print(simpath)
                #simfile_theta.append(simpath)
            #simfile.append(simfile_theta)
        #pprint(simfile)


def test2():
    gtpath = '/home/lir0b/Code/perceptual-reflection-removal/synthetic'
    simupath = '/media/lir0b/关羽/Simulation_RT'
    dl = DataLoader(simupath, gtpath)
    [ref_id, trans_id] = dl.simpair[5000]
    print(dl.simpair[5000])
    simpath = dl.get_theta_pol_filelist(ref_id, trans_id)
    pprint(simpath)
    refpath, transpath = dl.get_ref_trans_pair(ref_id, trans_id)
    print(refpath)
    print(transpath)


if __name__ == '__main__':
    test()
    #check_file()

    if False:
        from pprint import pprint

        gtpath = '/home/lir0b/Code/perceptual-reflection-removal/synthetic'
        # pprint(os.listdir(gtpath))
        rootpath = '/media/lir0b/关羽/Simulation_RT'

        dl = DataLoader(rootpath, gtpath)
        pprint(dl.simupair)
        pprint(dl.simufolder)
        folderlist = os.listdir(rootpath)
        pprint(folderlist)
        refname = [ff.split('_')[0] for ff in folderlist]
        transname = [ff.split('_')[1] for ff in folderlist]
        print(refname)
        print(transname)
        #print(len(tt))
        ff = os.listdir(os.path.join(rootpath, '7_21_10'))
        #pprint(ff)
        theta = [10, 40, 60, 70, 80, 85]
        phi = [0, 45, 90, 135]
        prefix = 'S0_theta_'
        folderlist = []
        filelist = ['S0_theta_%d_fi_%d.jpg'%(t, p) for t in theta for p in phi ]
        pprint(filelist)

