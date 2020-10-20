from models import resnet_ilsvrc
from models import resnet_cifar as cresnet, vgg_cifar as cvgg


def check_model(opt):
    if opt.model.startswith('resnet'):
        if opt.dataset in ['cub200', 'indoor', 'stanford40', 'flowers102', 'dog', 'tinyimagenet']:
            ResNet = resnet_ilsvrc.__dict__[opt.model]
            model = ResNet(num_classes=opt.num_classes)
        else:
            ResNet = cresnet.__dict__[opt.model]
            model = ResNet(num_classes=opt.num_classes)

        return model

    elif opt.model.startswith('vgg'):
        VGG = cvgg.__dict__[opt.model]
        model = VGG(num_classes=opt.num_classes)

        return model

    else:
        raise Exception('Unknown model')
