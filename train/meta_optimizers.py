import torch, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def _copy(state):
    if isinstance(state, torch.Tensor):
        return state.cpu().clone()

    elif isinstance(state, dict):
        new_state = {}
        for key in state:
            new_state[key] = _copy(state[key])

    elif isinstance(state, list):
        new_state = []
        for item in state:
            new_state.append(_copy(item))

    else:
        new_state = copy.deepcopy(state)

    return new_state

class MetaSGD(optim.SGD):
    def __init__(self, params, modules, lr=0.1, momentum=0, weight_decay=0, rollback=False, cpu=False):
        super(MetaSGD, self).__init__(params, lr, momentum=momentum, weight_decay=weight_decay)
        self.prev_states = []
        self.modules = modules + [self]
        self.rollback = rollback
        self.cpu = cpu

    def parameters(self):
        for pg in self.param_groups:
            for p in pg['params']:
                yield p

    def get_state(self):
        if self.cpu:
            return _copy([m.state_dict() for m in self.modules])
        else:
            return copy.deepcopy([m.state_dict() for m in self.modules])

    def set_state(self, state):
        for m, s in zip(self.modules, state):
            m.load_state_dict(s)

    def step(self, objective, *args, **kwargs):
        if objective is not None:
            self.prev_states.append((self.get_state(), objective, args, kwargs))
            loss = objective(*args, **kwargs)
            loss.backward()
        super(MetaSGD, self).step()

    def meta_backward(self):
        alpha_groups = []
        for pg in self.param_groups:
            alpha_groups.append([])
            for p in pg['params']:
                if p.grad is None:
                    p.grad = torch.zeros_like(p.data)
                grad = p.grad.data.clone()
                alpha_groups[-1].append((grad, torch.zeros_like(grad)))

        curr_state = self.get_state()
        for prev_state in reversed(self.prev_states):
            state, objective, args, kwargs = prev_state
            self.set_state(state)
            loss = objective(*args, **kwargs)
            grad = torch.autograd.grad(loss, list(self.parameters()),
                                       create_graph=True, allow_unused=True)
            grad = {p: g for p, g in zip(self.parameters(), grad)}
            X = 0.0
            for pg, ag in zip(self.param_groups, alpha_groups):
                lr = pg['lr']
                wd = pg['weight_decay']
                momentum = pg['momentum']
                for p, a in zip(pg['params'], ag):
                    g = grad[p]
                    if g is not None:
                        X = X+g.mul(a[0].mul(-lr)+a[1]).sum()
            self.zero_grad()
            X.backward()
            for pg, ag in zip(self.param_groups, alpha_groups):
                lr = pg['lr']
                wd = pg['weight_decay']
                momentum = pg['momentum']
                for p, a in zip(pg['params'], ag):
                    a_new = (a[0].mul(1-lr*wd).add_(wd, a[1]).add_(p.grad.data),
                             a[1].mul(momentum).add_(-lr*momentum, a[0]))
                    a[0].copy_(a_new[0])
                    a[1].copy_(a_new[1])
        self.prev_states = []
        if not self.rollback:
            self.set_state(curr_state)

    def __len__(self):
        return len(self.prev_states)

    def meta_backward_all(self, objective, outer_args):
        alpha_groups = []
        for pg in self.param_groups:
            alpha_groups.append([])
            for p in pg['params']:
                if p.grad is None:
                    p.grad = torch.zeros_like(p.data)
                grad = p.grad.data
                alpha_groups[-1].append((torch.zeros_like(grad), torch.zeros_like(grad)))

        curr_state = self.get_state()
        for prev_state, o_args in zip(reversed(self.prev_states), outer_args):
            grad = torch.autograd.grad(objective(*o_args), list(self.parameters()), allow_unused=True)
            grad = {p: g for p, g in zip(self.parameters(), grad)}
            for pg, ag in zip(self.param_groups, alpha_groups):
                for i, p in enumerate(pg['params']):
                    if grad[p] is not None:
                        ag[i][0].add_(grad[p])

            state, objective, args, kwargs = prev_state
            self.set_state(state)
            loss = objective(*args, **kwargs)
            grad = torch.autograd.grad(loss, list(self.parameters()),
                                       create_graph=True, allow_unused=True)
            grad = {p: g for p, g in zip(self.parameters(), grad)}
            X = 0.0
            for pg, ag in zip(self.param_groups, alpha_groups):
                lr = pg['lr']
                wd = pg['weight_decay']
                momentum = pg['momentum']
                for p, a in zip(pg['params'], ag):
                    g = grad[p]
                    if g is not None:
                        X = X+g.mul(a[0].mul(-lr)+a[1]).sum()
            self.zero_grad()
            X.backward()
            for pg, ag in zip(self.param_groups, alpha_groups):
                lr = pg['lr']
                wd = pg['weight_decay']
                momentum = pg['momentum']
                for p, a in zip(pg['params'], ag):
                    a_new = (a[0].mul(1-lr*wd).add_(wd, a[1]).add_(p.grad.data),
                             a[1].mul(momentum).add_(-lr*momentum, a[0]))
                    a[0].copy_(a_new[0])
                    a[1].copy_(a_new[1])
        self.prev_states = []
        self.set_state(curr_state)

def test_metaSGD():
    v1 = torch.nn.Parameter(torch.Tensor([1., 3.]))
    v2 = torch.nn.Parameter(torch.Tensor([[-1., -2.], [1., 0.]]))
    module = nn.Module()
    module.v1 = v1
    module.v2 = v2

    lmbd = torch.nn.Parameter(torch.zeros(2, 2))

    sgd = MetaSGD([v1, v2], [module], lr=0.1, momentum=0.9,  weight_decay=0.01)

    def inner_objective():
        return v1.pow(2).mean() + (lmbd*(v2**2)).sum()

    def outer_objective():
        return (v1*v2).mean()

    for _ in range(10):
        sgd.zero_grad()
        sgd.step(inner_objective)

    sgd.zero_grad()
    lmbd.grad.zero_()
    outer_objective().backward()
    sgd.meta_backward()

    print(lmbd.grad)
