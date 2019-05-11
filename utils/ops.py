import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable


def linear_p(inputs, weight, bias, delta_w, delta_b):
    return F.linear(inputs,
                    weight + delta_w * 0.0001,
                    bias + delta_b * 0.0001)


def conv2d_p(inputs, weight, bias, delta_w, delta_b):
    return F.conv2d(inputs,
                    weight + delta_w * 0.0001,
                    bias + delta_b * 0.0001)


def linear(inputs, weight, bias, meta_step_size=0.001, meta_loss=None, stop_gradient=False):
    if meta_loss is not None:

        if not stop_gradient:
            grad_weight = autograd.grad(meta_loss, weight, create_graph=True, allow_unused=True)[0]

            if bias is not None:
                grad_bias = autograd.grad(meta_loss, bias, create_graph=True, allow_unused=True)[0]
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        else:
            grad_weight = Variable(autograd.grad(meta_loss, weight, create_graph=True, allow_unused=True)[0].data, requires_grad=False)

            if bias is not None:
                grad_bias = Variable(autograd.grad(meta_loss, bias, create_graph=True, allow_unused=True)[0].data, requires_grad=False)
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        return F.linear(inputs,
                        weight - grad_weight * meta_step_size,
                        bias_adapt)
    else:
        return F.linear(inputs, weight, bias)


def conv2d(inputs, weight, bias, meta_step_size=0.001, stride=1, padding=0, dilation=1, groups=1, meta_loss=None,
           stop_gradient=False):
    if meta_loss is not None:

        if not stop_gradient:
            grad_weight = autograd.grad(meta_loss, weight, create_graph=True, allow_unused=True)[0]

            if bias is not None:
                grad_bias = autograd.grad(meta_loss, bias, create_graph=True, allow_unused=True)[0]
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        else:
            grad_weight = Variable(autograd.grad(meta_loss, weight, create_graph=True)[0].data,
                                   requires_grad=False)
            if bias is not None:
                grad_bias = Variable(autograd.grad(meta_loss, bias, create_graph=True)[0].data, requires_grad=False)
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        return F.conv2d(inputs,
                        weight - grad_weight * meta_step_size,
                        bias_adapt, stride,
                        padding,
                        dilation, groups)
    else:
        return F.conv2d(inputs, weight, bias, stride, padding, dilation, groups)


def batchnorm2d(inputs, running_mean, running_var, weight, bias, train, track_running_stats, momentum, eps, 
        meta_step_size=0.001, meta_loss=None, stop_gradient=False):
    if meta_loss is not None:

        if not stop_gradient:
            grad_weight = autograd.grad(meta_loss, weight, create_graph=True, allow_unused=True)[0]

            if bias is not None:
                grad_bias = autograd.grad(meta_loss, bias, create_graph=True, allow_unused=True)[0]
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        else:
            grad_weight = Variable(autograd.grad(meta_loss, weight, create_graph=True)[0].data,
                                   requires_grad=False)
            if bias is not None:
                grad_bias = Variable(autograd.grad(meta_loss, bias, create_graph=True)[0].data, requires_grad=False)
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        return F.batch_norm(inputs,
                            running_mean,
                            running_var,
                            weight - grad_weight * meta_step_size,
                            bias_adapt,
                            train,
                            momentum,
                            eps)
    else:
        return F.batch_norm(inputs, running_mean, running_var, weight, bias, train, momentum, eps)


def relu(inputs):
    return F.threshold(inputs, 0, 0, inplace=True)


def maxpool(inputs, kernel_size, stride=None, padding=0):
    return F.max_pool2d(inputs, kernel_size, stride, padding=padding)
