import torch, copy, itertools, random, datetime, pdb, sys, yaml, os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from torchvision import datasets
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import torch.nn.init as init
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, confusion_matrix, f1_score
import split_to_train_and_test
from split_to_train_and_test import SetUpData
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from E3diffusion_new import E3DiffusionProcess, remove_mean
from EquivariantGraphNeuralNetwork import EGCL, EquivariantGNN
from CN2_evaluate import calculate_angle_for_CN2
from PIL import Image, ImageFilter


def gen_per_iretation(parameters_yaml_file:str, model_state_name:str, result_save_name:str):

    with open(parameters_yaml_file,'r') as file:
        params = yaml.safe_load(file)
    
    seed = params['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    num_diffusion_timestep = params['num_diffusion_timestep']
    noise_precision = params['noise_precision']
    num_epochs = params['num_epochs']

    batch_size = params['batch_size']

    conditional = params['conditional']

    atom_type_size = params['atom_type_size']
    spectrum_size = params['spectrum_size']
    t_size = params['t_size']
    x_size = params['x_size']
    m_size = params['m_size']
    d_size = params['d_size']
    if conditional:
        h_size = spectrum_size + atom_type_size + t_size
    else:
        h_size = atom_type_size + t_size
    
    onehot_scaling_factor = params['onehot_scaling_factor']

    L= params['L'] #Lの値を大きくしすぎるとnanが出る（15のときに）
    lr = params['lr']
    weight_decay = params['weight_decay']

    max_grad_norm = params['max_grad_norm']

    m_input_size = h_size + h_size + d_size
    m_output_size = m_size
    m_hidden_size = params['m_hidden_size']

    h_input_size = h_size + m_size
    h_output_size = h_size
    h_hidden_size = params['h_hidden_size']

    if params['mlp_x_input'] == 'E3':
        x_input_size = h_size + h_size + d_size
    else:
        x_input_size = m_size
    x_output_size = 1
    x_hidden_size = params['x_hidden_size']
    
    if torch.cuda.is_available():
        torch.set.default_tensor_type(torch.cuda.FloatTensor)
    else
        torch.set.default_tensor_type(torch.FloatTensor)
    
    if to_compress_spectrum:
        spectrum_compressor = SpectrumCompressor(spectrum_size,compressor_hidden_dim,compressed_spectrum_size).to(device)
        h_size = compressed_spectrum_size + atom_type_size + t_size
        m_input_size = h_size + h_size + d_size
        h_input_size = h_size + m_size
        h_output_size = h_size
        x_input_size = h_size + h_size + d_size

    egnn = EquivariantGNN(L,m_input_size,m_hidden_size,m_output_size,x_input_size,x_hidden_size,x_output_size,h_input_size,h_hidden_size,h_output_size)

    diffusion_process = E3DiffusionProcess(s=noise_precision,num_diffusion_timestep=num_diffusion_timestep)
    


    criterion = nn.MSELoss()

    model_path = model_state_name

    checkpoint = torch.load('/home/rokubo/data/diffusion_model/model_state/model_to_predict_epsilon/'+model_path+'.pth')

    egnn.load_state_dict(checkpoint)

    setupdata = SetUpData(seed,conditional)

    data = np.load(dataset,allow_pickle=True)
    dataset = setupdata.npy_to_graph(data)
    dataset = setupdata.resize_spectrum(dataset,resize=spectrum_size)

    dataset_only_CN2 = []
    for i in range(len(dataset)):
        if dataset[i].pos.shape[0] == 3:
            dataset_only_CN2.append(dataset[i])
    dataset = dataset_only_CN2

    train_data, val_data, test_data = setupdata.split(dataset)


    original_graph_list = []
    generated_graph_list = []

    if conditional:
        for data in test_data:
            seed_value = 0
            num_of_generated_coords = 0
            how_many_gen = 10 #一つのグラフに対して何種類のseedから生成するか　パラメータで定めたい
            while num_of_generated_coords != how_many_gen:
                
                torch.manual_seed(seed_value)
                np.random.seed(seed_value)
                random.seed(seed_value)
                num_atom = data.spectrum.shape[0] #


                edge_index = []
                for i in range(num_atom):
                    for j in range(num_atom):
                        if i != j:
                            edge_index.append([i, j])
                initial_coords = torch.zeros(size=(num_atom,3))
                initial_h = torch.zeros(size=(num_atom,h_size))
                initial_coords.normal_()
                initial_h.normal_()
                initial_coords = initial_coords - torch.mean(initial_coords,dim=0,keepdim=True)
                graph = Data(x=initial_h,edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous(),pos=initial_coords,spectrum=data.spectrum)#

                egnn.eval()
                transition_of_graph_per_100step = []
                with torch.no_grad():
                    for time in list(range(num_diffusion_timestep,0,-1)):
                        if time%100 == 0:
                            transition_of_graph_per_100step.append(graph)
                    
                        time_tensor = torch.tensor([[time/num_diffusion_timestep] for d in range(num_atom)],dtype=torch.float32)
                        graph.h = torch.cat((onehot_scaling_factor*graph.x[:,:2],graph.x[:,2:]),dim=1)

                        new_h, new_x = egnn(graph.edge_index,torch.cat((graph.h,time_tensor),dim=1),graph.pos)

                        epsilon_x = remove_mean(new_x - graph.pos)
                        epsilon_h = new_h
                        mu_x = diffusion_process.calculate_mu(graph.pos,epsilon_x,time)
                        mu_h = diffusion_process.calculate_mu(graph.h,epsilon_h,time)
                        graph.pos = diffusion_process.reverse_diffuse_one_step(mu_x,time,mode='x')
                        graph.h = diffusion_process.reverse_diffuse_one_step(mu_h,time,mode='h')
                        if not torch.isfinite(graph.pos).all():
                            #raise ValueError('nan')
                            print('nan')
                            seed_value += 1
                            break
                        if not torch.isfinite(graph.h).all():
                            #raise ValueError('nan')
                            print('nan')
                            seed_value += 1
                            break
                
                    time_tensor = torch.tensor([[0] for d in range(num_atom)],dtype=torch.float32)
                    graph.h = torch.cat((onehot_scaling_factor*graph.x[:,:2],graph.x[:,2:]),dim=1)
                    new_h, new_x = egnn(graph.edge_index,torch.cat((graph.h,time_tensor),dim=1),graph.pos)
                    epsilon_x = remove_mean(new_x - graph.pos)
                    epsilon_h = new_h
                    alpha_0 = diffusion_process.alpha_schedule[0]
                    sigma_0 = diffusion_process.sigma_schedule[0]
                    mu_x = graph.pos / alpha_0 - sigma_0 * epsilon_x / alpha_0
                    noise_x = torch.zeros_like(graph.pos)
                    noise_x.normal_(mean=0,std=1)
                    noise_x = remove_mean(noise_x)
                    graph.pos = mu_x + sigma_0 * noise_x / alpha_0
                    mu_h = graph.h / alpha_0 - sigma_0 * epsilon_h / alpha_0
                    noise_h = torch.zeros_like(graph.h)
                    noise_h.normal_(mean=0,std=1)
                    graph.h = mu_h + sigma_0 * noise_h / alpha_0
                    h_onehot = graph.h[:,:2]
                    h_spectrum = graph.[:,2:]
                    h_atoms = nn.functional.one_hot(torch.argmax(h_onehot,dim=1),num_classes=2)
                    graph.h = torch.cat((h_atoms,h_spectrum),dim=1)


                if torch.isfinite(graph.pos).all():
                    transition_of_graph_per_100step.append(graph)
                    transition_of_graph_per_100step = torch.stack(transition_of_graph_per_100step)
                    transition_of_graph_per_100step = np.array(transition_of_graph_per_100step)
                    num_of_generated_coords += 1
                    seed_value += 1
                
                    original_graph_list.append(data)
                    generated_graph_list.append(graph)
    else:
        seed_value = 0
        num_of_generated_coords = 0
        how_many_gen = 1000 #一つのグラフに対して何種類のseedから生成するか　パラメータで定めたい
        while num_of_generated_coords != how_many_gen:
            
            torch.manual_seed(seed_value)
            np.random.seed(seed_value)
            random.seed(seed_value)
            num_atom = data.spectrum.shape[0] #現在はベクトルのサイズを参照しているが、実際は予測する


            edge_index = []
            for i in range(num_atom):
                for j in range(num_atom):
                    if i != j:
                        edge_index.append([i, j])
            initial_coords = torch.zeros(size=(num_atom,3))
            initial_h = torch.zeros(size=(num_atom,h_size))
            initial_coords.normal_()
            initial_h.normal_()
            initial_coords = initial_coords - torch.mean(initial_coords,dim=0,keepdim=True)
            graph = Data(x=initial_h,edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous(),pos=initial_coords,spectrum=data.spectrum)#

            egnn.eval()
            transition_of_graph_per_100step = []
            with torch.no_grad():
                for time in list(range(num_diffusion_timestep,0,-1)):
                    if time%100 == 0:
                        transition_of_graph_per_100step.append(graph)
                
                    time_tensor = torch.tensor([[time/num_diffusion_timestep] for d in range(num_atom)],dtype=torch.float32)
                    graph.h = onehot_scaling_factor*graph.x[:,:2]

                    new_h, new_x = egnn(graph.edge_index,torch.cat((graph.h,time_tensor),dim=1),graph.pos)

                    epsilon_x = remove_mean(new_x - graph.pos)
                    epsilon_h = new_h
                    mu_x = diffusion_process.calculate_mu(graph.pos,epsilon_x,time)
                    mu_h = diffusion_process.calculate_mu(graph.h,epsilon_h,time)
                    graph.pos = diffusion_process.reverse_diffuse_one_step(mu_x,time,mode='x')
                    graph.h = diffusion_process.reverse_diffuse_one_step(mu_h,time,mode='h')
                    if not torch.isfinite(graph.pos).all():
                        #raise ValueError('nan')
                        print('nan')
                        seed_value += 1
                        break
                    if not torch.isfinite(graph.h).all():
                        #raise ValueError('nan')
                        print('nan')
                        seed_value += 1
                        break
            
                time_tensor = torch.tensor([[0] for d in range(num_atom)],dtype=torch.float32)
                graph.h = onehot_scaling_factor*graph.x[:,:2]
                new_h, new_x = egnn(graph.edge_index,torch.cat((graph.h,time_tensor),dim=1),graph.pos)
                epsilon_x = remove_mean(new_x - graph.pos)
                epsilon_h = new_h
                alpha_0 = diffusion_process.alpha_schedule[0]
                sigma_0 = diffusion_process.sigma_schedule[0]
                mu_x = graph.pos / alpha_0 - sigma_0 * epsilon_x / alpha_0
                noise_x = torch.zeros_like(graph.pos)
                noise_x.normal_(mean=0,std=1)
                noise_x = remove_mean(noise_x)
                graph.pos = mu_x + sigma_0 * noise_x / alpha_0
                mu_h = graph.h / alpha_0 - sigma_0 * epsilon_h / alpha_0
                noise_h = torch.zeros_like(graph.h)
                noise_h.normal_(mean=0,std=1)
                graph.h = mu_h + sigma_0 * noise_h / alpha_0
                h_atoms = nn.functional.one_hot(torch.argmax(graph.h,dim=1),num_classes=2)
                graph.h = h_atoms


            if torch.isfinite(graph.pos).all():
                transition_of_graph_per_100step.append(graph)
                transition_of_graph_per_100step = torch.stack(transition_of_graph_per_100step)
                transition_of_graph_per_100step = np.array(transition_of_graph_per_100step)
                num_of_generated_coords += 1
                seed_value += 1
            
                original_graph_list.append(-1)
                generated_graph_list.append(graph)
                
    np.savez(result_save_name + '.npz',original_coords_list=original_coords_list,generated_coords_list=generated_coords_list)
