import torch
import math
from torch.optim.optimizer import Optimizer

class A2Grad(Optimizer):
    """Implements A2Grad.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        avg (string, optional): moving average (default: 'uni')
        lr (float, optional): learning rate (default: 1e-2)
        L (float, optional): Lipschitz constant (default: 1)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
    """
    def __init__(self, params, avg='uni', lr=1e-2, L=1, eps=1e-10):
        if not avg in ['uni', 'inc', 'exp']:
            raise ValueError("Invalid moving average: {}".format(avg))
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < L:
            raise ValueError("Invalid Lipschitz constant: {}".format(L))
        if not 0.0 <= eps:
            raise ValueError("Invalid eps: {}".format(eps))
        defaults = dict(avg=avg, lr=lr, L=L, eps=eps)
        super(A2Grad, self).__init__(params, defaults)
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = -1
                state['yk'] = torch.full_like(p.data, 0)
                state['vk'] = 0
                if group['avg'] == 'exp':
                    state['vk0'] = 0
                state['sum'] = torch.full_like(p.data, 0)
    
    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                state['step'] += 1

                
                if p.grad.data.is_sparse:
                    raise RuntimeError("A2Grad is not compatible with sparse gradients") # потом разберусь
                    
                alpha = 2/(state['step'] + 2)
                alpha_nxt = 2/(state['step'] + 3)
                delta = torch.dist(grad - state['sum']/(1 + state['step']), torch.full_like(state['sum'], 0))**2
                if group['avg'] == 'uni':
                    state['vk'] = state['vk'] + delta
                elif group['avg'] == 'inc':
                    k = state['step']
                    state['vk'] = state['vk']*(k**2)/((k+1)**2) + delta
                else:
                    if state['step'] == 0:
                        state['vk0'] = delta
                    else:
                        state['vk0'] = 0.5 * delta + 0.5 * state['vk0']
                    state['vk'] = max(state['vk'], state['vk0'])
                hk = 0
                if group['avg'] == 'exp':
                    hk = math.sqrt((state['step'] + 1) * state['vk'])
                else:
                    hk = math.sqrt(state['vk'])
                denominator = 2*group['L']/(1 + state['step']) + hk * group['lr'] + group['eps']
                xk = p.data - (1/denominator)*grad
                yk = (1 - alpha_nxt)*state['yk'] + alpha_nxt * xk - ((1 - alpha_nxt) * alpha)/denominator * grad
                p.data = xk
                state['yk'] = yk
                state['sum'] += grad
