import os, argparse, random, logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from check_dataset import check_dataset
from check_model import check_model
from utils.utils import AverageMeter, accuracy, set_logging_config
from train.meta_optimizers import MetaSGD

torch.backends.cudnn.benchmark = True

def _get_num_features(model):
    if model.startswith('resnet'):
        n = int(model[6:])
        if n in [18, 34, 50, 101, 152]:
            return [64, 64, 128, 256, 512]
        else:
            n = (n-2) // 6
            return [16]*n+[32]*n+[64]*n
    elif model.startswith('vgg'):
        n = int(model[3:].split('_')[0])
        if n == 9:
            return [64, 128, 256, 512, 512]
        elif n == 11:
            return [64, 128, 256, 512, 512]

    raise NotImplementedError

class FeatureMatching(nn.ModuleList):
    def __init__(self, source_model, target_model, pairs):
        super(FeatureMatching, self).__init__()
        self.src_list = _get_num_features(source_model)
        self.tgt_list = _get_num_features(target_model)
        self.pairs = pairs

        for src_idx, tgt_idx in pairs:
            self.append(nn.Conv2d(self.tgt_list[tgt_idx], self.src_list[src_idx], 1))

    def forward(self, source_features, target_features,
                weight, beta, loss_weight):

        matching_loss = 0.0
        for i, (src_idx, tgt_idx) in enumerate(self.pairs):
            sw = source_features[src_idx].size(3)
            tw = target_features[tgt_idx].size(3)
            if sw == tw:
                diff = source_features[src_idx] - self[i](target_features[tgt_idx])
            else:
                diff = F.interpolate(source_features[src_idx],
                                     scale_factor=tw/sw,
                                     mode='bilinear') - self[i](target_features[tgt_idx])
            diff = diff.pow(2).mean(3).mean(2)
            if loss_weight is None and weight is None:
                diff = diff.mean(1).mean(0).mul(beta[i])
            elif loss_weight is None:
                diff = diff.mul(weight[i]).sum(1).mean(0).mul(beta[i])
            elif weight is None:
                diff = (diff.sum(1)*(loss_weight[i].squeeze())).mean(0).mul(beta[i])
            else:
                diff = (diff.mul(weight[i]).sum(1)*(loss_weight[i].squeeze())).mean(0).mul(beta[i])
            matching_loss = matching_loss + diff
        return matching_loss

class WeightNetwork(nn.ModuleList):
    def __init__(self, source_model, pairs):
        super(WeightNetwork, self).__init__()
        n = _get_num_features(source_model)
        for i, _ in pairs:
            self.append(nn.Linear(n[i], n[i]))
            self[-1].weight.data.zero_()
            self[-1].bias.data.zero_()
        self.pairs = pairs

    def forward(self, source_features):
        outputs = []
        for i, (idx, _) in enumerate(self.pairs):
            f = source_features[idx]
            f = F.avg_pool2d(f, f.size(2)).view(-1, f.size(1))
            outputs.append(F.softmax(self[i](f), 1))
        return outputs

class LossWeightNetwork(nn.ModuleList):
    def __init__(self, source_model, pairs, weight_type='relu', init=None):
        super(LossWeightNetwork, self).__init__()
        n = _get_num_features(source_model)
        if weight_type == 'const':
            self.weights = nn.Parameter(torch.zeros(len(pairs)))
        else:
            for i, _ in pairs:
                l = nn.Linear(n[i], 1)
                if init is not None:
                    nn.init.constant_(l.bias, init)
                    # nn.init.constant_(l.weight, 0)
                self.append(l)
        self.pairs = pairs
        self.weight_type = weight_type

    def forward(self, source_features):
        outputs = []
        if self.weight_type == 'const':
            for w in F.softplus(self.weights.mul(10)):
                outputs.append(w.view(1, 1))
        else:
            for i, (idx, _) in enumerate(self.pairs):
                f = source_features[idx]
                f = F.avg_pool2d(f, f.size(2)).view(-1, f.size(1))
                if self.weight_type == 'relu':
                    outputs.append(F.relu(self[i](f)))
                elif self.weight_type == 'relu-avg':
                    outputs.append(F.relu(self[i](f.div(f.size(1)))))
                elif self.weight_type == 'elu':
                    outputs.append(F.elu(self[i](f))+1)
                elif self.weight_type == 'sigmoid':
                    outputs.append(F.sigmoid(2*self[i](f)))
                elif self.weight_type == 'tanh':
                    outputs.append(F.tanh(self[i](f))+1)
                elif self.weight_type == 'relu6':
                    outputs.append(F.relu6(self[i](f)))
                elif self.weight_type == 'softplus':
                    outputs.append(F.softplus(self[i](f)))
        return outputs


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataroot', required=True, help='Path to the dataset')
    parser.add_argument('--dataset', default='cub200')
    parser.add_argument('--datasplit', default='cub200')
    parser.add_argument('--batchSize', type=int, default=64, help='Input batch size')
    parser.add_argument('--workers', type=int, default=4)

    parser.add_argument('--source-model', default='resnet34', type=str)
    parser.add_argument('--source-domain', default='imagenet', type=str)
    parser.add_argument('--source-path', type=str, default=None)
    parser.add_argument('--target-model', default='resnet18', type=str)
    parser.add_argument('--weight-path', type=str, default=None)
    parser.add_argument('--wnet-path', type=str, default=None)

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1,help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--schedule', action='store_true', default=True)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--pairs', type=str, default='4-4,4-3,4-2,4-1,3-4,3-3,3-2,3-1,2-4,2-3,2-2,2-1,1-4,1-3,1-2,1-1')

    parser.add_argument('--meta-lr', type=float, default=1e-4, help='Initial learning rate for meta networks')
    parser.add_argument('--meta-wd', type=float, default=1e-4)
    parser.add_argument('--meta-start', type=int, default=0)
    parser.add_argument('--meta-period', type=int, default=1)
    parser.add_argument('--loss-weight', action='store_true', default=True)
    parser.add_argument('--loss-weight-type', type=str, default='relu6')
    parser.add_argument('--loss-weight-init', type=float, default=1.0)
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--optimizer', type=str, default='adam')

    parser.add_argument('--experiment', default='logs', help='Where to store models')

    # default settings
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(opt.experiment)
    set_logging_config(opt.experiment)
    logger = logging.getLogger('main')
    logger.info(' '.join(os.sys.argv))
    logger.info(opt)

    # load source model
    if opt.source_domain == 'imagenet':
        from models import resnet_ilsvrc
        source_model = resnet_ilsvrc.__dict__[opt.source_model](pretrained=True).to(device)
    else:
        opt.model = opt.source_model
        weights = []
        source_gen_params = []
        source_path = os.path.join(opt.source_path, '{}-{}'.format(opt.source_domain, opt.source_model), 
                                '0', 'model_best.pth.tar')
        ckpt = torch.load(source_path)
        opt.num_classes = ckpt['num_classes']
        source_model = check_model(opt).to(device)
        source_model.load_state_dict(ckpt['state_dict'], strict=False)

    pairs = []
    for pair in opt.pairs.split(','):
        pairs.append(( int(pair.split('-')[0]),
                       int(pair.split('-')[1])))

    wnet = WeightNetwork(opt.source_model, pairs).to(device)
    weight_params = list(wnet.parameters())
    if opt.loss_weight:
        lwnet = LossWeightNetwork(opt.source_model, pairs, opt.loss_weight_type, opt.loss_weight_init).to(device)
        weight_params = weight_params + list(lwnet.parameters())

    if opt.wnet_path is not None:
        ckpt = torch.load(opt.wnet_path)
        wnet.load_state_dict(ckpt['w'])
        if opt.loss_weight:
            lwnet.load_state_dict(ckpt['lw'])

    if opt.optimizer == 'sgd':
        source_optimizer = optim.SGD(weight_params, lr=opt.meta_lr, weight_decay=opt.meta_wd, momentum=opt.momentum, nesterov=opt.nesterov)
    else:
        source_optimizer = optim.Adam(weight_params, lr=opt.meta_lr, weight_decay=opt.meta_wd)

    # load dataloaders
    loaders = check_dataset(opt)

    # load target model
    opt.model = opt.target_model
    target_model = check_model(opt).to(device)
    target_branch = FeatureMatching(opt.source_model,
                                    opt.target_model,
                                    pairs).to(device)
    target_params = list(target_model.parameters()) + list(target_branch.parameters())
    if opt.meta_lr == 0:
        target_optimizer = optim.SGD(target_params, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.wd)
    else:
        target_optimizer = MetaSGD(target_params,
                                   [target_model, target_branch],
                                   lr=opt.lr,
                                   momentum=opt.momentum,
                                   weight_decay=opt.wd, rollback=True, cpu=opt.T>2)

    state = {
        'target_model': target_model.state_dict(),
        'target_branch': target_branch.state_dict(),
        'target_optimizer': target_optimizer.state_dict(),
        'w': wnet.state_dict(),
        'best': (0.0, 0.0)
    }
    if opt.loss_weight:
        state['lw'] = lwnet.state_dict()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(target_optimizer, opt.epochs)


    def validate(model, loader):
        acc = AverageMeter()
        model.eval()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_pred, _ = model(x)
            acc.update(accuracy(y_pred.data, y, topk=(1,))[0].item(), x.size(0))
        return acc.avg

    def inner_objective(data, matching_only=False):
        x, y = data[0].to(device), data[1].to(device)
        y_pred, target_features = target_model.forward_with_features(x)

        with torch.no_grad():
            s_pred, source_features = source_model.forward_with_features(x)

        weights = wnet(source_features)
        state['loss_weights'] = ''
        # if opt.meta_lr == 0:
        #     weights = None
        if opt.loss_weight:
            loss_weights = lwnet(source_features)
            state['loss_weights'] = ' '.join(['{:.2f}'.format(lw.mean().item()) for lw in loss_weights])
        else:
            loss_weights = None
        beta = [opt.beta] * len(wnet)

        matching_loss = target_branch(source_features,
                                      target_features,
                                      weights, beta, loss_weights)

        state['accuracy'] = accuracy(y_pred.data, y, topk=(1,))[0].item()
        if opt.meta_lr != 0:
            state['entropy'] = weights[-1].mul(torch.log(weights[-1])).sum(1).mean().item() * (-1)
        else:
            state['entropy'] = -1.0

        if matching_only:
            return matching_loss

        loss = F.cross_entropy(y_pred, y)
        state['loss'] = loss.item()
        return loss + matching_loss

    def outer_objective(data):
        x, y = data[0].to(device), data[1].to(device)
        y_pred, _ = target_model(x)
        state['accuracy'] = accuracy(y_pred.data, y, topk=(1,))[0].item()
        state['entropy'] = -1.0
        loss = F.cross_entropy(y_pred, y)
        state['loss'] = loss.item()
        return loss

    # source generator training
    state['iter'] = 0
    for epoch in range(opt.epochs):
        if opt.schedule:
            scheduler.step()

        state['epoch'] = epoch
        target_model.train()
        source_model.eval()
        for i, data in enumerate(loaders[0]):
            target_optimizer.zero_grad()
            inner_objective(data).backward()
            target_optimizer.step(None)

            logger.info('[Epoch {:3d}] [Iter {:3d}] [Loss {:.4f}] [Acc {:.4f}] [Entropy {:.4f}] [LW {}]'.format(
                state['epoch'], state['iter'],
                state['loss'], state['accuracy'], state['entropy'], state['loss_weights']))
            state['iter'] += 1

            if opt.meta_lr > 0 and state['iter'] % opt.meta_period == 0 and opt.meta_start <= epoch:
                for _ in range(opt.T):
                    target_optimizer.zero_grad()
                    target_optimizer.step(inner_objective, data, True)

                target_optimizer.zero_grad()
                target_optimizer.step(outer_objective, data)

                target_optimizer.zero_grad()
                source_optimizer.zero_grad()
                outer_objective(data).backward()
                target_optimizer.meta_backward()
                source_optimizer.step()

        acc = (validate(target_model, loaders[1]),
               validate(target_model, loaders[2]))

        if state['best'][0] < acc[0]:
            state['best'] = acc

        if state['epoch'] % 10 == 0:
            torch.save(state, os.path.join(opt.experiment, 'ckpt-{}.pth'.format(state['epoch']+1)))

        logger.info('[Epoch {}] [val {:.4f}] [test {:.4f}] [best {:.4f}]'.format(epoch, acc[0], acc[1], state['best'][1]))

if __name__ == '__main__':
    main()
