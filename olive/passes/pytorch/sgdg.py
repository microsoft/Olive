# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Based on https://github.com/JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform
import random

import torch
from torch.optim.optimizer import Optimizer, required

# ruff: noqa: N802,N806


def norm(v, dim=1):
    assert len(v.size()) == 2
    return v.norm(p=2, dim=dim, keepdim=True)


def unit(v, dim=1, eps=1e-8):
    vnorm = norm(v, dim)
    return v / vnorm.add(eps), vnorm


def qr_retraction(tan_vec):  # tan_vec, p-by-n, p <= n
    tan_vec.t_()
    q, r = torch.linalg.qr(tan_vec)  # pylint: disable=not-callable
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)
    q.t_()

    return q


def matrix_norm_one(W):
    out = torch.abs(W)
    out = torch.sum(out, dim=0)
    return torch.max(out)


def Cayley_loop(X, W, tan_vec, t):
    Y = X + t * tan_vec
    for _ in range(5):
        Y = X + t * torch.matmul(W, 0.5 * (X + Y))

    return Y.t()


episilon = 1e-8


class SGDG(Optimizer):
    r"""Cayley-SGD-G optimizer.

        This optimizer updates variables with two different routines
        based on the boolean variable 'stiefel'.

        If stiefel is True, the variables will be updated by SGD-G proposed
        as decorrelated weight matrix.

        If stiefel is False, the variables will be updated by SGD.
        This routine was taken from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        -- common parameters
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        stiefel (bool, optional): whether to use SGD-G (default: False)

        -- parameters in case stiefel is False
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        -- parameters in case stiefel is True
        omega (float, optional): orthogonality regularization factor (default: 0)
        grad_clip (float, optional): threshold for gradient norm clipping (default: None)

    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        stiefel=False,
        omega=0,
        grad_clip=None,
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "stiefel": stiefel,
            "omega": omega,
            "grad_clip": grad_clip,
        }
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        """Perform a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            stiefel = group["stiefel"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                unity, _ = unit(p.data.view(p.size()[0], -1))
                if stiefel and unity.size()[0] <= unity.size()[1]:

                    weight_decay = group["weight_decay"]
                    dampening = group["dampening"]
                    nesterov = group["nesterov"]

                    rand_num = random.randint(1, 101)
                    if rand_num == 1:
                        unity = qr_retraction(unity)

                    g = p.grad.data.view(p.size()[0], -1)

                    lr = group["lr"]

                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        param_state["momentum_buffer"] = torch.zeros(g.t().size())
                        if p.is_cuda:
                            param_state["momentum_buffer"] = param_state["momentum_buffer"].cuda()

                    V = param_state["momentum_buffer"]
                    V = momentum * V - g.t()
                    MX = torch.mm(V, unity)
                    XMX = torch.mm(unity, MX)
                    XXMX = torch.mm(unity.t(), XMX)
                    W_hat = MX - 0.5 * XXMX
                    W = W_hat - W_hat.t()
                    t = 0.5 * 2 / (matrix_norm_one(W) + episilon)
                    alpha = min(t, lr)

                    p_new = Cayley_loop(unity.t(), W, V, alpha)
                    V_new = torch.mm(W, unity.t())  # n-by-p
                    #                     check_identity(p_new.t())
                    p.data.copy_(p_new.view(p.size()))
                    V.copy_(V_new)

                else:
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if "momentum_buffer" not in param_state:
                            buf = param_state["momentum_buffer"] = d_p.clone()
                        else:
                            buf = param_state["momentum_buffer"]
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    p.data.add_(-group["lr"], d_p)

        return loss
