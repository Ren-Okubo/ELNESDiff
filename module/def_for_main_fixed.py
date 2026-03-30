import torch
import matplotlib.pyplot as plt
from diffusion_x_h import E3DiffusionProcess, remove_mean
import os
from transformers import get_cosine_schedule_with_warmup
from loss_calculation import kabsch_torch
from schedulefree import RAdamScheduleFree

def noise_schedule_for_GammaNetwork(model_state_path,params,target:str):
    assert target in ['gamma','alpha','sigma','SNR'], 'target must be one of gamma, alpha, sigma, or SNR'

    #パラメータの設定
    num_diffusion_timestep = params['num_diffusion_timestep']
    noise_schedule = params['noise_schedule']
    noise_precision = params['noise_precision']
    power = params['noise_schedule_power']

    #diffusion_processの定義
    diffusion_process = E3DiffusionProcess(s=noise_precision,power=power,num_diffusion_timestep=num_diffusion_timestep,noise_schedule=noise_schedule)

    #モデルの読み込み
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dicts = torch.load(model_state_path, map_location=device, weights_only=False)
    if 'GammaNetwork' in state_dicts:
        diffusion_process.gamma.load_state_dict(state_dicts['GammaNetwork'])
    
    #targetの計算
    fig, ax = plt.subplots()
    t = diffusion_process.t
    ax.set_xlabel('t')
    if target == 'gamma':
        target = diffusion_process.gamma_schedule()
        ax.set_ylabel('gamma')
        ax.set_title('gamma schedule')
    elif target == 'alpha':
        if noise_schedule == 'predefined':
            target = diffusion_process.alpha_schedule
        elif noise_schedule == 'learned':
            target = torch.sqrt(torch.sigmoid(-diffusion_process.gamma(t)))
        ax.set_ylabel('alpha')
        ax.set_title('alpha schedule')
    elif target == 'sigma':
        if noise_schedule == 'predefined':
            target = diffusion_process.sigma_schedule
        elif noise_schedule == 'learned':
            target = torch.sqrt(torch.sigmoid(diffusion_process.gamma(t)))
        ax.set_ylabel('sigma')
        ax.set_title('sigma schedule')
    elif target == 'SNR':
        if noise_schedule == 'predefined':
            target = diffusion_process.alpha_schedule ** 2 / diffusion_process.sigma_schedule ** 2
        elif noise_schedule == 'learned':
            target = torch.exp(-diffusion_process.gamma(t))
        ax.set_ylabel('SNR')
        ax.set_title('SNR')

    #targetのプロット
    target = target.to('cpu')
    t = t.to('cpu')
    ax.plot(t.detach().numpy(),target.detach().numpy())
    
    return fig


def load_model_state(nn_dict,model_save_path,params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dicts = torch.load(model_save_path, map_location=device, weights_only=False)
    if params['EMA']['use_ema']:
        nn_dict['egnn'].load_state_dict(state_dicts['egnn_ema'])
    else:
        nn_dict['egnn'].load_state_dict(state_dicts['egnn'])
    if params['NN_encoder']['spectrum_to_latent']:
        nn_dict['spectrum_compressor'].load_state_dict(state_dicts['spectrum_compressor'])
    if params['noise_schedule'] == 'learned':
        nn_dict['gamma'].load_state_dict(state_dicts['gamma'])
    return nn_dict


def evaluate_by_rmsd(original_graph_list,generated_graph_list):
    id_list = []
    rmsd_value_list = []
    original_coords_list, generated_coords_list = [],[]
    for i in range(len(original_graph_list)):
        original_graph = original_graph_list[i]
        generated_graph = generated_graph_list[i][-1]
        if original_graph.pos.shape[0] == 1:
            continue
        _,_,rmsd_value = kabsch_torch(original_graph.pos,generated_graph.pos)
        rmsd_value_list.append(rmsd_value)
        id_list.append(original_graph.id)
        original_coords_list.append(original_graph)
        generated_coords_list.append(generated_graph)
    id_rmsd_original_generated_list = list(zip(id_list,rmsd_value_list,original_coords_list,generated_coords_list)) #rmsdの値でソート
    sorted_id_rmsd_original_generated_list = sorted(id_rmsd_original_generated_list,key=lambda x:x[1])
    return sorted_id_rmsd_original_generated_list

def evaluate_by_rmsd_and_atom_type_eval(original_graph_list,generated_graph_list):
    id_list = []
    rmsd_value_list = []
    atom_type_eval_list = []
    original_coords_list, generated_coords_list = [],[]
    for i in range(len(original_graph_list)):
        original_graph = original_graph_list[i]
        generated_graph = generated_graph_list[i][-1]
        if original_graph.pos.shape[0] == 1:
            continue
        _,_,rmsd_value = kabsch_torch(original_graph.pos,generated_graph.pos)
        rmsd_value_list.append(rmsd_value)
        num_of_O_for_original = 0
        num_of_O_for_generated = 0
        device = original_graph.x.device
        O_tensor = torch.tensor([1,0],dtype=torch.long, device=device)
        for i in range(original_graph.x.shape[0]):
            if torch.equal(O_tensor,original_graph.x[i].to(device)):
                num_of_O_for_original += 1
            if torch.equal(O_tensor,generated_graph.x[i].to(device)):
                num_of_O_for_generated += 1
        atom_type_eval_list.append([num_of_O_for_original/original_graph.x.shape[0],num_of_O_for_generated/generated_graph.x.shape[0]])
        id_list.append(original_graph.id)
        original_coords_list.append(original_graph)
        generated_coords_list.append(generated_graph)
    id_rmsd_original_generated_list = list(zip(id_list,rmsd_value_list,atom_type_eval_list,original_coords_list,generated_coords_list)) #rmsdの値でソート
    sorted_id_rmsd_atomeval_original_generated_list = sorted(id_rmsd_original_generated_list,key=lambda x:x[1])
    return sorted_id_rmsd_atomeval_original_generated_list

def define_optimizer(params,nn_dict,diffusion_process,optim_type:str):
    assert optim_type in ['Adam','AdamW','RAdamScheduleFree']
    lr = params['optimizer']['lr']
    weight_decay = params['optimizer']['weight_decay']
    if params['NN_encoder']['spectrum_to_latent']:
        if params['noise_schedule'] == 'learned':
            param_list_for_optim = list(nn_dict['egnn'].parameters())+list(nn_dict['spectrum_compressor'].parameters())+list(diffusion_process.parameters())
        else:
            param_list_for_optim = list(nn_dict['egnn'].parameters())+list(nn_dict['spectrum_compressor'].parameters())
    else:
        if params['noise_schedule'] == 'learned':
            param_list_for_optim = list(nn_dict['egnn'].parameters())+list(diffusion_process.parameters())
        else:
            param_list_for_optim = list(nn_dict['egnn'].parameters())
    if optim_type == 'Adam':
        optimizer = torch.optim.Adam(param_list_for_optim,lr=lr,weight_decay=weight_decay)
    elif optim_type == 'AdamW':
        optimizer = torch.optim.AdamW(param_list_for_optim,lr=lr,weight_decay=weight_decay,amsgrad=True)
    elif optim_type == 'RAdamScheduleFree':
        optimizer = RAdamScheduleFree(param_list_for_optim,lr=lr)
    return optimizer

def define_lr_scheduler(optimizer,params):
    train_dataset_size = params['dataset_size']
    size_for_warmup = int(train_dataset_size / 20)
    num_training_steps = train_dataset_size / params['batch_size'] * params['num_epochs']
    num_warmup_steps = size_for_warmup / params['batch_size'] * params['num_epochs']
    #num_training_steps = 3800 / params['batch_size'] * params['num_epochs']
    #num_warmup_steps = 190 / params['batch_size'] * params['num_epochs']
    #num_training_steps = 3000
    #num_warmup_steps = params['optimizer']['num_warmup_steps']
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=num_warmup_steps,num_training_steps=num_training_steps)
    if params['lr_scheduler']['use_scheduler']:
        if params['lr_scheduler']['type'] == 'OneCycleLR':
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,params['optimizer']['lr'],epochs=params['num_epochs'],steps_per_epoch=int(train_dataset_size/params['batch_size']))
        elif params['lr_scheduler']['type'] == 'CosineAnnealingLR':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=int(params['num_epochs']*train_dataset_size/params['batch_size']),eta_min=0)
        elif params['lr_scheduler']['type'] == 'ReduceLROnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=30,min_lr=1e-6)
    return lr_scheduler

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
    

class EdgeIndexJudge ():
    def __init__(self, cutoff: float = 3.0, knn = 8):
        self.cutoff = cutoff
        self.knn = knn
        self.graph = None

    def set_graph(self, graph):
        self.graph = graph

    def pass_edges_cutoff(self):
        if self.graph is None:
            raise ValueError("Graph is not set. Please set the graph using set_graph method.")
        pos = self.graph.pos_at_t
        num_graph = self.graph.num_graphs
        batch_index = self.graph.batch
        source_index, target_index = [], []
        current_node = 0
        for num in range(num_graph):
            num_node = torch.eq(batch_index, num).sum().item()
            for i in range(current_node, current_node + num_node):
                for j in range(current_node, current_node + num_node):
                    if i != j:
                        source_index.append(i)
                        target_index.append(j)
            current_node += num_node
        source_index = torch.tensor(source_index, dtype=torch.long)
        target_index = torch.tensor(target_index, dtype=torch.long)
        edge_index = torch.stack([source_index, target_index], dim=0)
        assert source_index.shape[0] == target_index.shape[0]
        source_pos = pos[source_index]
        target_pos = pos[target_index]
        edge_length = torch.norm(source_pos - target_pos, dim=1)
        mask = edge_length < self.cutoff
        new_edge_index = edge_index[:, mask]
        new_graph = self.graph.clone()
        new_graph.edge_index = new_edge_index
        return new_graph
    
    def pass_edges_knn(self):
        if self.graph is None:
            raise ValueError("Graph is not set. Please set the graph using set_graph method.")
        pos = self.graph.pos_at_t
        num_graph = self.graph.num_graphs
        batch_index = self.graph.batch
        source_index, target_index = [], []
        current_node = 0
        for num in range(num_graph):
            num_node = torch.eq(batch_index, num).sum().item()
            if num_node < self.knn + 1:
                knn = num_node-1
            else:
                knn = self.knn
            for i in range(current_node, current_node + num_node):
                knn_indices = torch.topk(torch.cdist(pos[current_node:current_node + num_node], pos[current_node:current_node + num_node]), knn+1, largest=False).indices 
                for j in range(current_node, current_node + num_node):
                    if i != j and j in knn_indices[i - current_node]:
                        source_index.append(i)
                        target_index.append(j)
            current_node += num_node
        source_index = torch.tensor(source_index, dtype=torch.long)
        target_index = torch.tensor(target_index, dtype=torch.long)
        edge_index = torch.stack([source_index, target_index], dim=0)
        assert source_index.shape[0] == target_index.shape[0]
        new_graph = self.graph.clone()
        new_graph.edge_index = edge_index
        if hasattr(self.graph, 'num_nodes'):
            del new_graph.num_nodes
        return new_graph


    
