import torch
import torch.nn as nn

class SpectrumCompressor(nn.Module):
    def __init__(self,original_spectrum_dim,hidden_dim:list,compressed_spectrum_dim):
        super(SpectrumCompressor,self).__init__()

        assert isinstance(hidden_dim,list)
        self.original_spectrum_dim = original_spectrum_dim
        mlp_layers = []
        mlp_layers.append(nn.Linear(original_spectrum_dim,hidden_dim[0]))
        mlp_layers.append(nn.ReLU())
        if len(hidden_dim) > 1:
            for i in range(1,len(hidden_dim)):
                mlp_layers.append(nn.Linear(hidden_dim[i-1],hidden_dim[i]))
                mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(hidden_dim[-1],compressed_spectrum_dim))
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self,spectrum:torch.tensor):
        assert spectrum.shape[1] == self.original_spectrum_dim
        return self.mlp(spectrum)
