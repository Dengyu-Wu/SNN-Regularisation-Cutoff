import torch
from torch import nn
from torch.autograd import Function
from typing import Callable, List, Type
from gradients import GradFloor

class QCFS(nn.Module):
    def __init__(self, T: int = 4, 
                 L: int = 4, 
                 vthr: float = 8.0, 
                 tau: float = 1.0, 
                 gradient: Type[Function] = GradFloor, 
                 regularizer: Type[nn.Module] = None, momentum=0.9):
        super().__init__()
        self.vthr = nn.Parameter(torch.tensor([vthr]), requires_grad=True)
        self.regularizer = regularizer
        self.T = T
        self.L = L
        self.tau = tau
        self.momentum = momentum
        self.gradient = gradient.apply
        self.relu = nn.ReLU(inplace=True)
        
    def _forward(self, x):
        x = self.relu(x)
        x = x / self.vthr
        x = self.gradient(x*self.L+0.5)/self.L
        x = torch.clamp(x, 0, 1)
        x = x * self.vthr
        return x

    def forward(self, x):
        if self.regularizer is not None:
            loss = self.regularizer(x.clone())
        x = self._forward(x)
        return x 