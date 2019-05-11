# from nested_dict import nested_dict
from __future__ import division
from __future__ import print_function
from collections import OrderedDict
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.cuda.comm as comm
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torch.nn.init import kaiming_normal
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
import torch.nn.functional as F
import math
import shutil
import os
import random
import logging

#
# def normalize(input, p=2, dim=1, eps=1e-12):
#     r"""Performs :math:`L_p` normalization of inputs over specified dimension.
#     Does:
#     .. math::
#         v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}
#     for each subtensor v over dimension dim of input. Each subtensor is flattened into a vector,
#     i.e. :math:`\lVert v \rVert_p` is not a matrix norm.
#     With default arguments normalizes over the second dimension with Euclidean norm.
#     Args:
#         input: input tensor of any shape
#         p (float): the exponent value in the norm formulation
#         dim (int): the dimension to reduce
#         eps (float): small value to avoid division by zero
#     """
#     return input / input.norm(p, dim, True).clamp(min=eps).expand_as(input)

def fgsm(model, x, y, targeted=False, eps=0.03, x_val_min=1, x_val_max=1):
    x_adv = x.clone()
    x_adv.requires_grad = True
    h_adv, _ = model(x_adv)
    if targeted:
        cost = F.cross_entropy(h_adv, y)
    else:
        cost = -F.cross_entropy(h_adv, y)

    model.zero_grad()
    if x_adv.grad is not None:
        x_adv.grad.data.fill_(0)
    cost.backward()

    x_adv.grad.sign_()
    x_adv = x_adv - eps*x_adv.grad
    x_adv = torch.clamp(x_adv, x_val_min, x_val_max)

    h, _ = model(x)
    h_adv, _ = model(x_adv)

    return x_adv, h_adv, h

def accuracy_bc(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    output_pred = torch.round(output)
    correct = output_pred.eq(target)
    correct_k = correct.float().sum(0)
    res = correct_k.mul_(100.0 / batch_size)
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def give_noise(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            weight_noise = torch.zeros(m.weight.data.size()).normal_(0, 0.001)
            if m._parameters['weight'].is_cuda:
                weight_noise = weight_noise.cuda()
            m.weight.data += weight_noise
            if m.bias is not None:
                bias_noise = torch.zeros(m.bias.data.size()).normal_(0, 0.001)
                if m._parameters['bias'].is_cuda:
                    bias_noise = bias_noise.cuda()
                m.bias.data += bias_noise


def gaussian_noise(ins, is_training, device, mean=0, stddev=0.1):
    if is_training:
        noise = torch.randn(ins.size()).to(device) * stddev + mean
        return ins + noise
    return ins


def to_np(x):
    return x.data.cpu().numpy()

def adjust_learning_rate(optimizer, epoch, opt, dataset=None):
    lr = opt.lr
    if dataset in ['cifar100', 'mnist2svhn', 'cifar10', 'stl10', 'tinyimagenet']:
        # if opt.depth >= 110 and epoch == 3:
        #     opt.lr = 0.1
        #     lr = opt.lr
        if opt.model_type == 'resnet':
            if epoch > 91 and epoch < 137:
                lr = opt.lr * 0.1
            if epoch >= 137:
                lr = opt.lr * 0.01
            if epoch >= 160:
                lr = opt.lr * 0.001
        elif opt.model_type == 'vgg':
            if epoch >= 400:
                lr = opt.lr * 0.1
    elif dataset in ['mit67', 'flowers102', 'cub200']:
        lr = opt.lr * (0.1 ** (epoch // 25))
    elif dataset in ['ilsvrc']:
        lr = opt.lr * (0.1 ** (epoch // 30))
    elif dataset in ['mnist']:
        lr = opt.lr * (0.1 ** (epoch // 50))
    # elif dataset in ['cifar100']:
    #     if opt.depth >= 110 and epoch == 3:
    #         opt.lr = 0.1
    #     if epoch > 120 and epoch < 160:
    #         lr = opt.lr * 0.1
    #     if epoch >= 160:
    #         lr = opt.lr * 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr# * param_group['lr_mult']

def save_checkpoint(state, is_best, save, filename='checkpoint.pth.tar', meta=False, learner=False):
    if save:
        torch.save(state, filename)
        if is_best:
            print('save the best model up to now...')
            if meta:
                if learner:
                    shutil.copyfile(filename, '{}/lmd_learner_model_best.pth.tar'.format(os.path.dirname(filename)))
                else:
                    shutil.copyfile(filename, '{}/meta_model_best.pth.tar'.format(os.path.dirname(filename)))
            else:
                shutil.copyfile(filename, '{}/model_best.pth.tar'.format(os.path.dirname(filename)))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def to_np(x):
    return x.data.cpu().numpy()

def distillation(y, teacher_scores, labels, T, alpha):
    return F.kl_div(F.log_softmax(y/T), F.softmax(teacher_scores/T)) * (T*T * 2. * alpha) \
            + F.cross_entropy(y, labels) * (1. - alpha)

def fsp(fs):
    fsp_matrix = []
    fs0 = fs[0].view(-1, fs[0].size(1), fs[0].size(2)*fs[0].size(3))
    fs2 = fs[2].view(-1, fs[2].size(1), fs[2].size(2)*fs[2].size(3))
    fs3 = fs[3].view(-1, fs[3].size(1), fs[3].size(2)*fs[3].size(3))
    fs4 = fs[4].view(-1, fs[4].size(1), fs[4].size(2)*fs[4].size(3))
    fs5 = fs[5].view(-1, fs[5].size(1), fs[5].size(2)*fs[5].size(3))
    fs6 = fs[6].view(-1, fs[6].size(1), fs[6].size(2)*fs[6].size(3))
    g_0 = torch.bmm(fs0, fs2.permute(0,2,1))/fs0.size(2)
    g_1 = torch.bmm(fs3, fs4.permute(0,2,1))/fs3.size(2)
    g_2 = torch.bmm(fs5, fs6.permute(0,2,1))/fs5.size(2)
    fsp_matrix.append(g_0)
    fsp_matrix.append(g_1)
    fsp_matrix.append(g_2)

    return fsp_matrix

def at_wo_flat(x):
    return F.normalize(x.pow(2).mean(1)).unsqueeze_(1)


def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def feat_matching(x, y):
    if x.size(3) != y.size(3):
        x = F.interpolate(x, scale_factor=y.size(3)/x.size(3), mode='bilinear')
    return (x-y).pow(2).mean()


def at_loss(x, y, avg=True):
    if x.size(3) != y.size(3):
        x = F.interpolate(x, scale_factor=y.size(3)/x.size(3), mode='bilinear')
    if avg:
        return (at(x) - at(y)).pow(2).mean()
    else:
        return (at(x) - at(y)).pow(2)
    #return (at(x) - at(y)).abs().mean()

def at_loss_with_newlmd(x, y, lmd):
    if x.size(3) != y.size(3):
        x = F.interpolate(x, scale_factor=y.size(3)/x.size(3), mode='bilinear')
    at_loss_v = (at(x) - at(y)).pow(2).mean(1)
    at_loss_wlmd = at_loss_v * lmd#.expand_as(at_loss_v)
    return at_loss_wlmd.mean()
    #return (at(x) - at(y)).pow(2).mean()


# For synthetic data
# ref : https://www.programcreek.com/python/example/101172/torch.log
def logaddexp(x1, x2):
    """
    Elementwise logaddexp function: log(exp(x1) + exp(x2))

    Args:
        x1: A tensor.
        x2: A tensor.

    Returns:
        tensor: Elementwise logaddexp.

    """
    # log(exp(x1) + exp(x2))
    # = log( exp(x1) (1 + exp(x2 - x1))) = x1 + log(1 + exp(x2 - x1))
    # = log( exp(x2) (exp(x1 - x2) + 1)) = x2 + log(1 + exp(x1 - x2))
    diff = torch.min(x2 - x1, x1 - x2)
    return torch.max(x1, x2) + torch.log1p(torch.exp(diff)) 


def sample_gmm(gmm, n_samples):
    if type(gmm)!=list:
        X = gmm.sample(torch.Size([n_samples]))
    else:
        X = []
        r = random.randint(0, len(gmm)-1)
        for i in range(len(gmm)):
            if i != r:
                X.append(gmm[i].sample(torch.Size([n_samples//len(gmm)])))
            else:
                X.append(gmm[i].sample(torch.Size([(n_samples//len(gmm))+(n_samples%len(gmm))])))
        
        X = torch.cat(X, 0)
    
    return X


def log_prob_gmm(gmm, X):     
    if type(gmm)!=list:
        logprob = gmm.log_prob(X)
    else:
        for i in range(len(gmm)):
            if i != 0:
                logprob = logaddexp(logprob, gmm[i].log_prob(X))
            else:
                logprob = gmm[i].log_prob(X)
    
    return logprob
    
# ref : https://stackoverflow.com/questions/26079881/kl-divergence-of-two-gmms
# KL-Divergence of two GMMs (Monte Carlo)

def gmm_kl(gmm_p, gmm_q, n_samples=10**5):
    X = sample_gmm(gmm_p, n_samples)
    log_p_X = log_prob_gmm(gmm_p, X)
    log_q_X = log_prob_gmm(gmm_q, X)

    return (log_p_X - log_q_X).mean()


# Jenson-Shannon Divergence
def gmm_js(gmm_p, gmm_q, n_samples=10**5):
    X = sample_gmm(gmm_p, n_samples)
    log_p_X = log_prob_gmm(gmm_p, X)
    log_q_X = log_prob_gmm(gmm_q, X)
    log_mix_X = logaddexp(log_p_X, log_q_X)

    Y = sample_gmm(gmm_q, n_samples)
    log_p_Y = log_prob_gmm(gmm_p, Y)
    log_q_Y = log_prob_gmm(gmm_q, Y)
    log_mix_Y = logaddexp(log_p_Y, log_q_Y)

    return (log_p_X.mean() - (log_mix_X.mean() - torch.log(torch.Tensor([2])))
            + log_q_Y.mean() - (log_mix_Y.mean() - torch.log(torch.Tensor([2]))))[0] / 2


# ref : https://github.com/ast0414/adversarial-example/blob/master/craft.py
def compute_jacobian(inputs, output, idx):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    assert inputs.requires_grad

    # Make output flat, not necessary
    output = output.view(output.size(0), -1)
    
    num_classes = output.size()[1]

    jacobian = torch.zeros(*inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    zero_gradients(inputs)
    grad_output.zero_()
    grad_output[range(output.size(0)), idx] = 1
    # Calculating Jacobian in a Differentiable Way
    output.backward(grad_output, retain_graph=True) # memory leak is fixed

    jacobian = inputs.grad.data

    #return torch.transpose(jacobian, dim0=0, dim1=1)
    return jacobian

def norm_jacobian(inputs, output, idx):
    '''
    normalize a jacobian
    '''
    jacobian = compute_jacobian(inputs, output, idx)
    return F.normalize(jacobian.view(jacobian.size(0),-1))

def jac_loss(inputs, x, y, idx):
    return F.mse_loss(norm_jacobian(inputs, x, idx), norm_jacobian(inputs, y, idx))


def temp_scaling(inputs, temp):
    '''
    :param inputs : Batch X Classes (before softmax layer)
    :param temp : Temperature
    :return : Batch X Classes (temperature scaled output)
    '''
    outputs = inputs / temp 
    return F.softmax(outputs, dim=1)

def odin_perturb(inputs, outputs, temp=1000, eps=0.0012):
    grad = compute_jacobian(inputs, temp_scaling(outputs, temp), torch.max(outputs,1)[1]) 
    return (inputs - eps * torch.sign(-grad))



# ref : https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# modified for meta-training

def content_loss(input, target):
    # we 'detach' the target content from the tree used
    # to dynamically compute the gradient: this is a stated value,
    # not a variable. Otherwise the forward method of the criterion
    # will throw an error.
    loss = F.mse_loss(input, target)
    return loss

def gram_matrix(input):
    b, c, h, w = input.size()  # b=batch size(=1 if single image)
    # c=number of features
    # (h,w)=dimensions of a f. map (N=h*w)

    features = input.view(b, c, h * w)  # resise F_XL into \hat F_XL

    G = torch.bmm(features, features.permute(0,2,1))  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(c*h*w)

def style_loss(input, target_feature):
    loss = F.mse_loss(gram_matrix(input), gram_matrix(target_feature))
    return loss


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = torch.optim.Adam([input_img.requires_grad_()])
    return optimizer

def get_style_losses(fs, ft):
    style_losses = []
    for fs_l, ft_l in zip(fs, ft):
        style_losses.append(style_loss(fs_l, ft_l))
    
    return style_losses


def get_content_losses(ft, ft_ori):
    content_losses = []
    for ft_l, ft_l_ori in zip(ft, ft_ori):
        content_losses.append(content_loss(ft_l, ft_l_ori))
    
    return content_losses

def run_style_transfer(source_model, inputs_s, inputs_t, i=None, num_steps=200,
                       style_weight=100, content_weight=1, negative=False, print_log=False):
    """Run the style transfer."""
    if print_log:
        print('Building the style transfer model..')
    
    _, fs = source_model(inputs_s, i)
    _, ft = source_model(inputs_t, i)
    
    inputs_img = inputs_t.clone()
    ft_ori = []
    for ft_l in ft:
        ft_ori.append(ft_l.clone())
    optimizer = get_input_optimizer(inputs_t)
    
    b = inputs_t.size(0)
    # DeepSet setting
    fs_set = []
    for fs_l in fs:
        fs_set.append(fs_l.mean(0).repeat(b,1,1,1))

    if print_log:
        print('Optimizing..')
    run = 0

    while run <= num_steps:
        if run > 0:
            _, ft = source_model(inputs_t, i)
        
        style_losses = get_style_losses(ft, fs_set)
        content_losses = get_content_losses(ft, ft_ori)
       
        # correct the values of updated input image
        inputs_t.data.clamp_(-2, 2)

        optimizer.zero_grad()

        style_score = sum(style_losses)
        content_score = sum(content_losses)            

        style_score *= style_weight
        content_score *= content_weight

        if negative:
            loss = content_score - style_score
        else:
            loss = content_score + style_score
        
        loss.backward(retain_graph=True)

        run += 1
        if print_log and (run % 50 == 0):
            print("run {}:".format(run))
            print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                style_score.item(), content_score.item()))
            print()


        optimizer.step()

    # a last correction...
    inputs_t.data.clamp_(-2, 2)

    return inputs_img, inputs_t



def net_reg(x, y):
    if x.size(3) != y.size(3):
        x = F.interpolate(x, scale_factor=y.size(3)/x.size(3), mode='bilinear')

    return (at(x)-at(y)).pow(2).mean()

def net_reg_abs(x, y):
    if x.size(3) != y.size(3):
        x = F.interpolate(x, scale_factor=y.size(3)/x.size(3), mode='bilinear')

    return (at(x)-at(y)).abs().mean()

def residual_at_loss(x, y, fs):
    if x.size(1) == 1:
        return (x.view(x.size(0),-1) + at(fs) - at(y)).pow(2).mean()
    else:
        return (at(x) + at(fs) - at(y)).pow(2).mean()
def preprocess_gradients(x):
    p = 10 
    eps = 1e-6
    indicator = (x.abs() > math.exp(-p)).float()
    x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)
    # if x.norm() == 0:
    #     x3 = x
    # else:
    #     x3 = x/(x.norm()+1e-12)

    #return x3#torch.cat((x1, x2), 1)
    return torch.cat((x1, x2), 1)

def set_logging_config(logdir):
    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, 'log.txt')),
                                  logging.StreamHandler(os.sys.stdout)])

def gradient_similarity(grad1, grad2):
    X = 0.0
    norm1 = 0.0
    norm2 = 0.0
    with torch.no_grad():
        for g1, g2 in zip(grad1, grad2):
            if g1 is not None:
                norm1 += g1.mul(g1).sum()
            if g2 is not None:
                norm2 += g2.mul(g2).sum()
            if g1 is None or g2 is None:
                continue
            X = X + g2.mul(g1).sum()
        return X.item() / (norm1.item() ** 0.5) / (norm2.item() ** 0.5)
