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

class E3DiffusionProcess(torch.nn.Module):
    def __init__(self,s,power,num_diffusion_timestep:int,noise_schedule:str='predefined',deterministic=False):
        super(E3DiffusionProcess,self).__init__()
        self.noise_schedule = noise_schedule
        self.deterministic_denoising = deterministic
        if noise_schedule =='predefined':
            self.noise_precision = s
            self.power = power
            self.num_diffusion_timestep = num_diffusion_timestep
            self.t = torch.linspace(0,num_diffusion_timestep,num_diffusion_timestep+1) #０からnum_diffusion_timestepまでの5001個の整数
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
            return torch.sqrt(torch.sigmoid(self.gamma(self.t)[t]))
        
    def get_SNR(self,t:int):
        SNR = self.alpha(t)**2 / self.sigma(t)**2
        return SNR




    def diffuse_zero_to_t(self,z:torch.tensor,t:int,mode='pos'):
        noise = torch.zeros_like(z,dtype=torch.float).to(z.device)        
        noise.normal_(mean=0,std=1)
        if mode == 'pos':
            noise = remove_mean(noise)
        alpha_t = self.alpha(t).to(z.device)
        sigma_t = self.sigma(t).to(z.device)
        z_after_diffuse = alpha_t * z + sigma_t * noise
        return z_after_diffuse, noise
    
    def calculate_mu(self,z:torch.tensor,epsilon:torch.tensor,t:int):
        alpha_t = self.alpha(t)
        alpha_s = self.alpha(t-1)
        #alpha_s = self.alpha(t)
        squared_sigma_t = 1 - alpha_t**2
        sigma_t = torch.sqrt(squared_sigma_t)
        squared_sigma_s = 1 - alpha_s**2
        sigma_s = torch.sqrt(squared_sigma_s)
        alpha_ts = alpha_t / alpha_s
        squared_sigma_ts = squared_sigma_t - torch.pow(alpha_ts,2) * squared_sigma_s

        mu = z / alpha_ts - squared_sigma_ts * epsilon / alpha_ts / sigma_t
        return mu
    
    def reverse_diffuse_one_step(self,z,epsilon:torch.tensor,t:int,mode='pos'):
        mu = self.calculate_mu(z,epsilon,t)
        alpha_t = self.alpha(t)
        #alpha_s = self.alpha(t)
        alpha_s = self.alpha(t-1)
        squared_sigma_t = 1 - alpha_t**2
        squared_sigma_s = 1 - alpha_s**2
        alpha_ts = alpha_t / alpha_s
        squared_sigma_ts = squared_sigma_t - torch.pow(alpha_ts,2) * squared_sigma_s
        if self.deterministic_denoising: # reverse diffusion by DDIM denoising equation 
            #sigma = self.sigma(t-1) / self.sigma(t) * torch.sqrt(1-self.alpha(t)**2/self.alpha(t-1)**2)
            sigma = 0
            noise = torch.zeros_like(z).to(z.device)
            noise.normal_(mean=0,std=1)
            if mode =='pos':
                noise = remove_mean(noise)
            z = self.alpha(t-1) / self.alpha(t) *(z - self.sigma(t) * epsilon) + torch.sqrt(1-self.alpha(t-1)**2-sigma**2) * epsilon + sigma * noise
            return z
        else:
            std = torch.sqrt(squared_sigma_ts * squared_sigma_s / squared_sigma_t)
            noise = torch.zeros_like(z).to(z.device)
            noise.normal_(mean=0,std=1)
            if mode =='pos':
                noise = remove_mean(noise)
            z = mu + std * noise
            return z

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

