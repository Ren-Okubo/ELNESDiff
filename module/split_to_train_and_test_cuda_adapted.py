import torch, copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from torchvision import datasets
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import torch.nn.init as init
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, confusion_matrix, f1_score
import itertools
from diffusion_process import DiffusionProcess
from torch_geometric.data import Data

def npy_to_CN_binary_dataset(npy_data:np.array) -> TensorDataset:
    input_data = []
    output_data = [] #label : {0,1,3,4:0, 2:1}
    label_reference = {0:0,1:0,3:0,4:0,2:1}
    for data in npy_data:
        input_data.append(data[1])
        output_data.append(label_reference[int(len(data[2])-1)])
    input_data = torch.tensor(input_data,dtype=torch.float32)
    output_data = torch.tensor(output_data,dtype=torch.long)
    return TensorDataset(input_data,output_data)

def npy_to_CN_0134_dataset(npy_data:np.array) -> TensorDataset:
    input_data = []
    output_data = [] #label : {0:0, 1:1, 3:2, 4:3}
    label_reference = {0:0,1:1,3:2,4:3}
    for data in npy_data:
        input_data.append(data[1])
        output_data.append(int(len(data[2])-1))
    input_data = torch.tensor(input_data,dtype=torch.float32)
    output_data = torch.tensor(output_data,dtype=torch.long)
    return TensorDataset(input_data,output_data)

def npy_to_FFN_input(npy_data:np.array, diffusion_process) -> np.array: #npy_data : [d,fitted_intensities,local_env_list]
    graphs, conformation_to_diffuse, spectrums, local_env_to_epsilon = [],[],[],[]
    time_data, graph_data, spectrum_data, conformation_data, equivaliant_epsilon_data = [],[],[],[],[]
    data = []
    for i in range(npy_data.shape[0]):
        node_list = []
        conformation_list = []
        for graph_com in npy_data[i][2]:
            node_list.append(graph_com[0])
            conformation_list.append(graph_com[1])
        #conformation_tuple = tuple(conformation_list)
        conformation_to_diffuse.append(np.vstack(conformation_list))
        graphs.append(node_list)
        spectrums.append(npy_data[i][1])
        #local_env_to_epsilon.append(npy_data[i][2])
        
    for C_at_zero, graph, spectrum in zip(conformation_to_diffuse,graphs,spectrums):
        for time in range(1,diffusion_process.num_diffusion_timestep+1):
            C_at_zero = torch.tensor(C_at_zero,dtype=torch.float32)
            data.append([graph,spectrum,time,diffusion_process.diffuse_zero_to_t_torch(C_at_zero,time),diffusion_process.equivaliant_epsilon_torch(graph,C_at_zero,time)])
    return data


class SetUpData():
    def __init__(self,seed=None,conditional=False):
        self.seed = seed
        self.conditional = conditional

    
    def npy_to_graph(self,npy_data:np.array) -> list:
        """
        graph.x : atom type,sperctrum, graph.pos = coords
        npy : list(mp_id:str, spectrum:np.array, local_atom_list:[[atom_type:[1,0]or[0,1],coord:np.array],...])
        """
        npy_data_except_CN0 = np.array([d for d in npy_data if len(d[2]) != 1])
        dataset = []
        a = 0
        for data in npy_data_except_CN0:
            x, edge_index, pos, spectrums = [],[],[],[]

            for graph_info in data[2]:
                atom_type = torch.tensor([graph_info[0]],dtype=torch.float32)
                spectrum = torch.tensor(np.array([data[1]]),dtype=torch.float32)
                x.append(atom_type)
                spectrums.append(spectrum)
                pos.append(torch.tensor(graph_info[1],dtype=torch.float32))

            x = torch.cat(x,dim=0)
            pos = torch.stack(pos)
            spectrums = torch.cat(spectrums,dim=0)
            node_num_list = [i for i in range(x.shape[0])]
            permutations = list(itertools.permutations(node_num_list,2))
            for permutation in permutations:
                edge_index.append(list(permutation))
            edge_index = torch.tensor(edge_index,dtype=torch.long)
            graph = Data(x=x,edge_index=edge_index.t().contiguous(),pos=pos,spectrum=spectrums)
            graph.id = data[0]
            dataset.append(graph)
        return dataset
    
    def split(self,dataset:list,train_ratio=0.8,val_ratio=0.1,test_ratio=0.1):
        generator = torch.Generator()
        if self.seed is not None:
            torch.manual_seed(self.seed)
            generator.manual_seed(self.seed)
        
        data_length = len(dataset)
        train_data_length = int(data_length*train_ratio)
        val_data_length = int(data_length*val_ratio)
        test_data_length = data_length - train_data_length - val_data_length
        train_val_data , test_data = random_split(dataset,[train_data_length+val_data_length,test_data_length],generator=generator)
        train_data, val_data = random_split(train_val_data,[train_data_length,val_data_length],generator=generator)
        return train_data, val_data, test_data

    def resize_spectrum(self,dataset:list,resize:int=200):
        for data in dataset:
            spectrum = data.spectrum
            resized_spectrum = spectrum[:,:resize]
            data.spectrum = resized_spectrum
        return dataset
