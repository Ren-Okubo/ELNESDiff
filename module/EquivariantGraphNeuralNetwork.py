import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class EGCLold(MessagePassing):
    def __init__(self,m_input,m_hidden,m_output,
                 x_input,x_hidden,x_output,
                 h_input,h_hidden,h_output,
                 flow='target_to_source',aggr='sum',activation='SiLU'):
        super(EGCLold,self).__init__(aggr=aggr,flow=flow)
        
        self.mlp_m = nn.Sequential(
            nn.Linear(m_input,m_hidden),
            nn.SiLU(),
            #nn.Dropout(0.3),
            nn.Linear(m_hidden,m_output),
            nn.SiLU()
            )
        self.mlp_x = nn.Sequential(
            nn.Linear(x_input,x_hidden),
            nn.SiLU(),
            #nn.Dropout(0.3),
            nn.Linear(x_hidden,x_hidden),
            nn.SiLU(),
            nn.Linear(x_hidden,x_output)
            )
        self.mlp_h = nn.Sequential(
            nn.Linear(h_input,h_hidden),
            nn.SiLU(),
            #nn.Dropout(0.3),
            nn.Linear(h_hidden,h_output)
            )
        self.attention = nn.Sequential(
            nn.Linear(m_output,1),
            nn.Sigmoid()
            )
        """
        self.mlp_m = self._build_mlp(m_input,m_hidden,m_output)
        self.mlp_x = self._build_mlp(x_input,x_hidden,x_output)
        self.mlp_h = self._build_mlp(h_input,h_hidden,h_output)
        self.attention = nn.Sequential(
            nn.Linear(m_output,1),
            nn.Sigmoid()
        )
        """
    def _build_mlp(self,input_dim,hidden_dim,output_dim):
        mlp = []
        mlp.append(nn.Linear(input_dim,hidden_dim[0]))
        mlp.append(nn.SiLU())
        if len(hidden_dim) > 1:
            for i in range(1,len(hidden_dim)):
                mlp.append(nn.Linear(hidden_dim[i-1],hidden_dim[i]))
                mlp.append(nn.SiLU())
        mlp.append(nn.Linear(hidden_dim[-1],output_dim))
        return nn.Sequential(*mlp)
    
    def message(self,h_i,h_j,coords_i,coords_j,mode,attention):
        input_data = torch.cat((h_i,h_j,torch.norm(coords_i-coords_j,dim=1,keepdim=True)**2),dim=1)
        if mode == 'h':
            out = self.mlp_m(input_data)
            if attention == True:
                out = out * self.attention(out)
            return out
        elif mode == 'x':
            input_data = torch.cat((h_i,h_j,torch.norm(coords_i-coords_j,dim=1,keepdim=True)**2),dim=1)
            #out = (coords_i - coords_j) * self.mlp_x(input_data) / (torch.norm(coords_i - coords_j,dim=1,keepdim=True)+1)
            out = (coords_i - coords_j) * self.mlp_x(input_data) / (torch.norm(coords_i - coords_j)+1)
            return out

    def forward(self,edge_index,h,coords):
        sum_message = self.propagate(edge_index=edge_index, h=h, coords=coords, mode='h',attention=True)
        updated_h = self.mlp_h(torch.cat((h,sum_message),dim=1))
        updated_x = coords + self.propagate(edge_index=edge_index, h=h, coords=coords, mode='x',attention=True)
        return updated_h, updated_x

class EquivariantGNNold(nn.Module):
    def __init__(
        self,L,m_input,m_hidden,m_output,
        x_input,x_hidden,x_output,
        h_input,h_hidden,h_output
        ):
        super(EquivariantGNNold,self).__init__()
        self.L = L
        self.egcl_list = nn.ModuleList([EGCLold(m_input,m_hidden,m_output,
                                          x_input,x_hidden,x_output,
                                          h_input,h_hidden,h_output) for i in range(L)])
    
    def forward(self,edge_index,h,x):
        for l in range(self.L):
            h, x = self.egcl_list[l](edge_index,h,x)
        return h, x
    


class EGCL(MessagePassing):
    def __init__(self,m_input,m_hidden,m_output,
                 x_input,x_hidden,x_output,
                 h_input,h_hidden,h_output,
                 flow='target_to_source',aggr='mean',activation='SiLU'):
        super(EGCL,self).__init__(aggr=aggr,flow=flow)
        
        self.mlp_m = nn.Sequential(
            nn.Linear(m_input,m_hidden),
            nn.SiLU(),
            #nn.Dropout(0.3),
            nn.Linear(m_hidden,m_output),
            nn.SiLU()
            )
        self.mlp_x = nn.Sequential(
            nn.Linear(x_input,x_hidden),
            nn.SiLU(),
            #nn.Dropout(0.3),
            nn.Linear(x_hidden,x_hidden),
            nn.SiLU(),
            nn.Linear(x_hidden,x_output)
            )
        self.mlp_h = nn.Sequential(
            nn.Linear(h_input,h_hidden),
            nn.SiLU(),
            #nn.Dropout(0.3),
            nn.Linear(h_hidden,h_output)
            )
        self.attention = nn.Sequential(
            nn.Linear(m_output,1),
            nn.Sigmoid()
            )
        """
        self.mlp_m = self._build_mlp(m_input,m_hidden,m_output)
        self.mlp_x = self._build_mlp(x_input,x_hidden,x_output)
        self.mlp_h = self._build_mlp(h_input,h_hidden,h_output)
        self.attention = nn.Sequential(
            nn.Linear(m_output,1),
            nn.Sigmoid()
        )
        """
    def _build_mlp(self,input_dim,hidden_dim,output_dim):
        mlp = []
        mlp.append(nn.Linear(input_dim,hidden_dim[0]))
        mlp.append(nn.SiLU())
        if len(hidden_dim) > 1:
            for i in range(1,len(hidden_dim)):
                mlp.append(nn.Linear(hidden_dim[i-1],hidden_dim[i]))
                mlp.append(nn.SiLU())
        mlp.append(nn.Linear(hidden_dim[-1],output_dim))
        return nn.Sequential(*mlp)
    
    def message(self,h_i,h_j,coords_i,coords_j,mode,attention):
        input_data = torch.cat((h_i,h_j,torch.norm(coords_i-coords_j,dim=1,keepdim=True)**2),dim=1)
        if mode == 'h':
            out = self.mlp_m(input_data)
            if attention == True:
                out = out * self.attention(out)
            return out
        elif mode == 'x':
            input_data = torch.cat((h_i,h_j,torch.norm(coords_i-coords_j,dim=1,keepdim=True)**2),dim=1)
            out = (coords_i - coords_j) / (torch.norm(coords_i-coords_j,dim=1,keepdim=True) + 1) * self.mlp_x(input_data)
            return out

    def forward(self,edge_index,h,coords):
        sum_message = self.propagate(edge_index=edge_index, h=h, coords=coords, mode='h',attention=True)
        updated_h = self.mlp_h(torch.cat((h,sum_message),dim=1))
        updated_x = coords + self.propagate(edge_index=edge_index, h=h, coords=coords, mode='x',attention=True)
        return updated_h, updated_x

class EquivariantGNN(MessagePassing):
    def __init__(
        self,L,m_input,m_hidden,m_output,
        x_input,x_hidden,x_output,
        h_input,h_hidden,h_output
        ):
        super(EquivariantGNN,self).__init__()
        self.L = L
        self.egcl_list = nn.ModuleList([EGCL(m_input,m_hidden,m_output,
                                          x_input,x_hidden,x_output,
                                          h_input,h_hidden,h_output) for i in range(L)])
        self.h_embedding_out = nn.Sequential(
            nn.Linear(h_output, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )
        self.eps_prediction = nn.Sequential(
            nn.Linear(x_input, x_hidden),
            nn.SiLU(),
            nn.Linear(x_hidden, x_hidden),
            nn.SiLU(),
            nn.Linear(x_hidden, x_output)  # Output dimension is 1 for epsilon prediction
        )
        
    def message(self, h_i, h_j, coords_i, coords_j):
        dij = torch.norm(coords_i - coords_j, dim=1, keepdim=True)
        input_data = torch.cat((h_i, h_j, dij ** 2), dim=1)
        out = (coords_i - coords_j) / (dij + 1) * self.eps_prediction(input_data)
        return out
    
    def forward(self,edge_index,h,x):
        for l in range(self.L):
            h, x = self.egcl_list[l](edge_index,h,x)

        # Final output processing
        output_h = self.h_embedding_out(h)
        #output_x = self.propagate(edge_index=edge_index, h=h, coords=x)
        h = output_h
        #x = output_x
        return h, x