import math
import types

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.linalg as linalg


class ActNorm(nn.Module):
    """ An implementation of a activation normalization layer
    from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(ActNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_inputs))
        self.bias = nn.Parameter(torch.zeros(num_inputs))
        self.initialized = False

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if self.initialized == False:
            self.weight.data.copy_(torch.log(1.0 / (inputs.std(0) + 1e-12)))
            self.bias.data.copy_(inputs.mean(0))
            self.initialized = True

        if mode == 'direct':
            return (
                inputs - self.bias) * torch.exp(self.weight), self.weight.sum(
                    -1, keepdim=True).unsqueeze(0).repeat(inputs.size(0), 1)
        else:
            return inputs * torch.exp(
                -self.weight) + self.bias, -self.weight.sum(
                    -1, keepdim=True).unsqueeze(0).repeat(inputs.size(0), 1)

class LUInvertibleMM(nn.Module):
    """ An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(LUInvertibleMM, self).__init__()
        self.W = torch.Tensor(num_inputs, num_inputs)
        nn.init.orthogonal_(self.W)
        self.L_mask = torch.tril(torch.ones(self.W.size()), -1)
        self.U_mask = self.L_mask.t().clone()

        P, L, U = linalg.lu(self.W.numpy())
        self.P = torch.from_numpy(P)
        self.L = nn.Parameter(torch.from_numpy(L))
        self.U = nn.Parameter(torch.from_numpy(U))

        S = np.diag(U)
        sign_S = np.sign(S)
        log_S = np.log(abs(S))
        self.sign_S = torch.from_numpy(sign_S)
        self.log_S = nn.Parameter(torch.from_numpy(log_S))

        self.I = torch.eye(self.L.size(0))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if str(self.L_mask.device) != str(self.L.device):
            self.L_mask = self.L_mask.to(self.L.device)
            self.U_mask = self.U_mask.to(self.L.device)
            self.I = self.I.to(self.L.device)
            self.P = self.P.to(self.L.device)
            self.sign_S = self.sign_S.to(self.L.device)

        L = self.L * self.L_mask + self.I
        U = self.U * self.U_mask + torch.diag(
            self.sign_S * torch.exp(self.log_S))
        W = self.P @ L @ U

        if mode == 'direct':
            return inputs @ W, self.log_S.sum().unsqueeze(0).unsqueeze(
                0).repeat(inputs.size(0), 1)
        else:
            return inputs @ torch.inverse(
                W), -self.log_S.sum().unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0), 1)

class Flatten(nn.Module):
    def forward(self, input):
        #print('Flatten:', input.size())
        return input.view(input.size()[0],-1)

class Reshape(nn.Module):
    def forward(self, input):
        return input.view(input.size()[0],1,-1)

class MuitlChannelReshape(nn.Module):
    def forward(self, input):
        N = input.size()[0]
        
        chan1 = input[:,:24].view(N, 1, -1)
        chan2 = input[:,24:].view(N, 1, -1)
        return torch.cat((chan1, chan2), 1)

class CouplingLayer(nn.Module):
    """ An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 mask,
                 num_cond_inputs=None,
                 s_act='tanh',
                 t_act='relu'):
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        self.mask = mask

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'lrelu': nn.LeakyReLU}
        s_act_func = activations[s_act]
        t_act_func = activations[t_act]

        if num_cond_inputs is not None:
            total_inputs = num_inputs + num_cond_inputs
        else:
            total_inputs = num_inputs

        self.scale_net = nn.Sequential(
            Reshape(),
            nn.Conv1d(in_channels=1, out_channels=8, 
                        kernel_size = 5, stride = 1, 
                        dilation = 1),
            nn.BatchNorm1d(8),
            s_act_func(),

            nn.Conv1d(in_channels=8, out_channels=8, 
                        kernel_size = 5, stride = 1, 
                        dilation = 1),
            nn.BatchNorm1d(8),
            s_act_func(),

            nn.MaxPool1d(3,stride = 2),

            Flatten(),
            nn.Linear(19*8, 24)
            )


        self.translate_net = nn.Sequential(
            Reshape(),
            nn.Conv1d(in_channels=1, out_channels=8, 
                        kernel_size = 5, stride = 1, 
                        dilation = 1),
            nn.BatchNorm1d(8),
            t_act_func(),

            nn.Conv1d(in_channels=8, out_channels=8, 
                        kernel_size = 5, stride = 1, 
                        dilation = 1),
            nn.BatchNorm1d(8),
            t_act_func(),

            nn.MaxPool1d(3,stride = 2),

            Flatten(),
            nn.Linear(19*8, 24)
            )

        def init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                m.bias.data.fill_(0)
                nn.init.orthogonal_(m.weight.data)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        mask = self.mask
        
        #if cond_inputs is not None:
        masked_inputs = inputs * mask
        net_inputs = torch.cat([cond_inputs,masked_inputs], -1)
        
        if mode == 'direct':
            log_s = self.scale_net(net_inputs).squeeze() * (1 - mask)
            t = self.translate_net(net_inputs).squeeze() * (1 - mask)
            s = torch.exp(log_s)

            return inputs * s + t, log_s.sum(-1, keepdim=True)
        else:
            log_s = self.scale_net(net_inputs).squeeze() * (1 - mask)
            t = self.translate_net(net_inputs).squeeze() * (1 - mask)
            s = torch.exp(-log_s)


            return (inputs - t) * s, -log_s.sum(-1, keepdim=True)



class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet

        return inputs, logdets

    def log_probs(self, inputs, cond_inputs = None):
        u, log_jacob = self.forward(inputs, cond_inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode='inverse')[0]
        return samples

