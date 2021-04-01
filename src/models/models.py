import torch


def create_model(opt):
    if opt.model == 'pix2pixHD':
        from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    elif opt.model == 'sepmodel':
        from pid.models.sepmodel import SepModel
        if opt.isTrain:
            model = SepModel()
        else:
            pass
    elif opt.model == 'polar_engine':
        from pid.models.sepmodel import PolarEngine
        model = PolarEngine()
    elif opt.model == 'PLD':
        from .pldmodel import PLDModel
        model = PLDModel()
    elif opt.model == 'Diffuse':
        from .pldmodel import DiffuseModel
        model = DiffuseModel()
    elif opt.model == 'Shading':
        from .pldmodel import ShadingMatrixModel
        model = ShadingMatrixModel()
    elif opt.model == 'H00':
        from .pldmodel import ShadingModel
        model = ShadingModel()
    elif opt.model == 'DePolar':
        from .pldmodel import DePolarModel
        model = DePolarModel()
    elif opt.model == 'HModel3x1':
        from .pldmodel import HModel3x1
        model = HModel3x1()
    elif opt.model == 'Smooth':
        from .pldmodel import SmoothingModel
        model = SmoothingModel()
    elif opt.model == 'SepModel':
        from .sepmodel import SepModel
        model = SepModel()
    else:
        from .ui_model import UIModel
        model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
