from models import resnet_ilsvrc


def check_model(opt):
    if opt.model.startswith('resnet'):
        ResNet = resnet_ilsvrc.__dict__[opt.model]
        model = ResNet(num_classes=opt.num_classes)

        return model
    else:
        raise Exception('Unknown model')
