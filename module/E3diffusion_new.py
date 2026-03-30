import torch
import torch.nn as nn
from SNR import GammaNetwork

def remove_mean(x:torch.tensor,batch_index=None):
    if batch_index is None:
        mean = torch.mean(x,dim=0,keepdim=True)
        x = x - mean
    else:
        num_graph = batch_index.max().item()+1
        for i in range(num_graph):
            mean = torch.mean(x[batch_index==i],dim=0,keepdim=True)
            x[batch_index==i] = x[batch_index==i] - mean
    return x

class E3DiffusionProcess():
    def __init__(self,s,power,num_diffusion_timestep:int,noise_schedule:str='predefined'):
        self.noise_schedule = noise_schedule
        if noise_schedule =='predefined':
            self.noise_precision = s
            self.num_diffusion_timestep = num_diffusion_timestep
            self.t = torch.linspace(0,num_diffusion_timestep,num_diffusion_timestep+1) #０からnum_diffusion_timestepまでの5001個の整数
            #self.alpha_schedule1 = (1-2*s) * (1-(self.t / num_diffusion_timestep)**2) + s
            #self.sigma_schedule1 = torch.sqrt(1-self.alpha_schedule1**2)
            self.alpha_schedule = self.polynomial_schedule(num_diffusion_timestep,s=s,power=power)
            self.sigma_schedule = torch.sqrt(1-self.alpha_schedule**2)
        elif noise_schedule =='learned':
            self.gamma = GammaNetwork() 
            self.num_diffusion_timestep = num_diffusion_timestep
            self.t = torch.linspace(0,1,num_diffusion_timestep+1).view(num_diffusion_timestep+1,1)
    
    def gamma_schedule(self):
        return self.gamma(self.t)

    def alpha(self,t:int):
        if self.noise_schedule == 'predefined':
            return self.alpha_schedule[t]
        elif self.noise_schedule == 'learned':
            return torch.sqrt(torch.sigmoid(-self.gamma(self.t)[t]))

    def sigma(self,t:int):
        if self.noise_schedule == 'predefined':
            return self.sigma_schedule[t]
        elif self.noise_schedule == 'learned':
            return torch.sqrt(torch.sigmoid(self.gamma(self.t))[t])
        
        """
        print(self.alpha_schedule)
        print(self.t)
        
        assert torch.equal(self.alpha_schedule1,self.alpha_schedule), 'alpha_schedule1 is not equal to alpha_schedule'
        assert torch.equal(self.sigma_schedule1,self.sigma_schedule), 'sigma_schedule1 is not equal to sigma_schedule'
        pdb.set_trace()
        """

    def diffuse_zero_to_t(self,pos:torch.tensor,t:int):
        noise = torch.zeros_like(pos,dtype=torch.float32)
        noise.normal_(mean=0,std=1)
        noise = remove_mean(noise)
        pos_after_diffuse = self.alpha(t) * pos + self.sigma(t) * noise
        return pos_after_diffuse , noise
    
    def calculate_mu(self,pos:torch.tensor,epsilon:torch.tensor,t:int):
        alpha_t = self.alpha(t)
        """
        print('t:',self.t[5000])
        print('alpha_t:',alpha_t)
        print('sigma_t:',self.sigma_schedule[t])
        print('alpha_schedule:',self.alpha_schedule)
        print('sigma_schedule:',self.sigma_schedule)
        """
        alpha_s = self.alpha(t-1)
        squared_sigma_t = 1 - alpha_t**2
        squared_sigma_s = 1 - alpha_s**2
        alpha_ts = alpha_t / alpha_s
        squared_sigma_ts = squared_sigma_t - torch.pow(alpha_ts,2) * squared_sigma_s
        x_hat = (pos - self.sigma(t) * epsilon) / alpha_t
        """
        print('x_hat:',x_hat)
        print(alpha_s*squared_sigma_ts/squared_sigma_t)
        """
        mu = alpha_ts * squared_sigma_s * pos / squared_sigma_t + alpha_s * squared_sigma_ts * x_hat / squared_sigma_t
        return mu
    
    def reverse_diffuse_one_step(self,mu,t:int):
        alpha_t = self.alpha(t)
        alpha_s = self.alpha(t-1)
        squared_sigma_t = 1 - alpha_t**2
        squared_sigma_s = 1 - alpha_s**2
        alpha_ts = alpha_t / alpha_s
        squared_sigma_ts = squared_sigma_t - torch.pow(alpha_ts,2) * squared_sigma_s
        std = torch.sqrt(squared_sigma_ts * squared_sigma_s / squared_sigma_t)
        noise = torch.zeros_like(mu)
        noise.normal_(mean=0,std=1)
        noise = remove_mean(noise)
        #print('mu:',mu)
        pos = mu + std * noise
        return pos

    def clip_noise_schedule(self,alphas2,clip_value=0.001):
        alphas2 = torch.cat([torch.ones(1),alphas2],dim=0)
        alphas_step = (alphas2[1:] / alphas2[:-1])
        alphas_step = torch.clamp(alphas_step,min=clip_value,max=1.)
        alphas2 = torch.cumprod(alphas_step,dim=0)
        return alphas2
    
    def polynomial_schedule(self,timesteps:int,s=1e-4,power=3.):
        steps = timesteps
        x = torch.linspace(0,steps,steps+1)
        alphas2 = torch.pow(1 - torch.pow(x / steps,power),2)
        alphas2 = self.clip_noise_schedule(alphas2,clip_value=0.001)
        precision = 1 - 2 * s
        alphas2 = precision * alphas2 + s
        return alphas2

