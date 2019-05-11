import models
from models import resnet_ilsvrc

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

    elif opt.model.startswith('mlp'):
        model = mlp.__dict__[opt.model](num_classes=opt.num_classes)
        return model

    elif opt.model == 'smallnet':
        return small_net.SmallNet(num_classes=opt.num_classes)

    else:
        raise Exception('Unknown model')


def check_label_generator(opt):
    if opt.model.startswith('resnet'):
        n = (int(opt.model[6:])-2) // 6
        return LabelGenerator([16]*n+[32]*n+[64]*n)#, num_labels=64, average_mode='simple')
        #return LabelGenerator([16]*n+[32]*n+[64]*n, num_labels=64, average_mode='simple')

    elif opt.model.startswith('vgg'):
        if opt.model.startswith('vgg4'):
            return LabelGenerator([64, 128, 512])
        if opt.model.startswith('vgg9'):
            return LabelGenerator([64, 128, 256, 512, 512], num_labels=opt.num_labels)
            #return LabelGenerator([64, 128, 256, 512, 512], num_labels=64)

    elif opt.model.startswith('mlp'):
        n = int(opt.model[3:])
        return LabelGenerator([n, n])

    else:
        raise Exception('Unknown model')


def check_matching_module(source_generator, target_generator, opt):
    assert source_generator.num_labels == target_generator.num_labels

    return MatchingModule(len(source_generator.num_features),
                          len(target_generator.num_features),
                          source_generator.num_labels,
                          last_only=opt.last_only,
                          temperature=opt.matching_temperature)


def check_wnet(opt):
    #return WNet([64])
    if opt.model.startswith('resnet'):
        #n = (int(opt.model[6:])-2) // 6
        return WNet([16]*2+[32]*1+[64]*1)#, num_labels=64, average_mode='simple')
        #return LabelGenerator([16]*n+[32]*n+[64]*n, num_labels=64, average_mode='simple')

    elif opt.model.startswith('vgg'):
        if opt.model.startswith('vgg4'):
            return WNet([64, 128, 512])
        if opt.model.startswith('vgg9'):
            return WNet([64, 128, 256, 512, 512], num_labels=opt.num_labels)
            #return LabelGenerator([64, 128, 256, 512, 512], num_labels=64)

    elif opt.model.startswith('mlp'):
        n = int(opt.model[3:])
        return WNet([n, n])

    else:
        raise Exception('Unknown model')

