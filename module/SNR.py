import math, pdb
import torch
import torch.nn as nn

class PositiveLinear(nn.Module):
    def __init__(self,input_size,output_size,param_init_offset=-2.0):
        super(PositiveLinear,self).__init__()
        self.weight = nn.Parameter(torch.empty(output_size,input_size))
        self.register_parameter('bias',None)
        self.param_init_offset = param_init_offset
        self.reset_parameters()
    
    def reset_parameters(self):
        self.weight = torch.nn.init.kaiming_uniform(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight = nn.Parameter(self.weight + torch.full_like(self.weight,self.param_init_offset))


    def forward(self,x):
        positive_weight = nn.Parameter(torch.nn.functional.softplus(self.weight))
        return nn.functional.linear(x,positive_weight,self.bias)



    
class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""
    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        #self.gamma_0 = torch.tensor([-5.],requires_grad=True)
        #self.gamma_1 = torch.tensor([10.],requires_grad=True)

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma