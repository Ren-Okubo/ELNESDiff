import torch, copy, itertools, random, datetime, pdb, yaml, pytz, os, copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import numpy as np
import torch.nn.init as init
import split_to_train_and_test_cuda_adapted as split_to_train_and_test
from tqdm import tqdm
import torch_geometric
from split_to_train_and_test_cuda_adapted import SetUpData
from EquivariantGraphNeuralNetwork import EquivariantGNN, EGCL
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from E3diffusion_new import E3DiffusionProcess, remove_mean
from CN2_evaluate import calculate_angle_for_CN2
from DataPreprocessor import SpectrumCompressor
import wandb
from def_for_main_cuda_adapted import EdgeIndexJudge

class EarlyStopping():
    def __init__(self, patience=0):
        self._step= 0
        self._loss=float('inf')
        self._patience=patience

    def validate(self,loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self._patience:
                return True
        else:
            self._step = 0
            self._loss = loss
       
        return False
    
def log_constant_Z(diffusion_process,M):
    alpha_0 = diffusion_process.alpha(0)
    sigma_0 = diffusion_process.sigma(0)
    log_z = (M-1) * 3 * torch.log(torch.sqrt(torch.tensor(2*torch.pi))*sigma_0/alpha_0)
    return log_z

def get_L0(diffusion_process,predicted_epsilon, target_epsilon, t, batch_index):
    mask_for_0 = torch.zeros_like(t)
    mask_for_0[t==0] = 1
    mask_for_t = torch.zeros_like(t)
    mask_for_t[t!=0] = 1
    if mask_for_0.sum().item() == 0:
        return torch.tensor(0).to(predicted_epsilon.device)
    num_graph = torch.unique(batch_index[mask_for_0==1]).numel()
    L_0_total = 0
    for i in range(num_graph):
        predicted = predicted_epsilon[batch_index==torch.unique(batch_index[mask_for_0==1])[i]][:,:3]
        target = target_epsilon[batch_index==torch.unique(batch_index[mask_for_0==1])[i]][:,:3]
        M = predicted.shape[0]
        log_Z = log_constant_Z(diffusion_process,M)
        L_0 = -log_Z - 0.5 * torch.mean((predicted - target)**2)
        L_0_total += L_0
    L_0 = L_0_total / num_graph
    return L_0

def gaussian_KL_divergence_with_dimension(mu1, sigma1, mu2, sigma2, dim):
    # KL divergence between two Gaussian distributions
    kl_div = dim * torch.log(sigma2/sigma1) + 0.5 * ((dim * sigma1**2 + (mu1 - mu2)**2) / sigma2**2 - dim)
    return kl_div

def get_L_T(diffusion_process,batch_data):
    #batch_data内のグラフ数を取得
    num_graph = batch_data.batch.max().item()+1
    #graphごとにkl divergenceを計算
    kl_div_list = []
    for i in range(num_graph):
        #t=Tにおけるmuとsigmaを取得
        x = batch_data.pos_at_0[batch_data.batch==i]
        mu_x = diffusion_process.alpha(diffusion_process.num_diffusion_timestep) * x
        sigma_x = diffusion_process.sigma(diffusion_process.num_diffusion_timestep)
        h = batch_data.h_at_0[batch_data.batch==i]
        mu_h = diffusion_process.alpha(diffusion_process.num_diffusion_timestep) * h
        sigma_h = diffusion_process.sigma(diffusion_process.num_diffusion_timestep)
        #xのkl divergenceを計算
        kl_x = gaussian_KL_divergence_with_dimension(mu_x, sigma_x, torch.zeros_like(mu_x), torch.ones_like(sigma_x),mu_x.shape[0]-1)
        #hのkl divergenceを計算
        kl_h = gaussian_KL_divergence_with_dimension(mu_h, sigma_h, torch.zeros_like(mu_h), torch.ones_like(sigma_h),mu_h.shape[0])
        #全体のkl divergenceのスカラー値を計算
        kl_value = torch.mean(kl_x) + torch.mean(kl_h)
        kl_div_list.append(kl_value.item())
    #全体のkl divergenceの平均値を計算
    L_T = -sum(kl_div_list) / len(kl_div_list)
    return L_T

def get_L_t(diffusion_process, predicted_epsilon, target_epsilon,t,batch_index):
    mask_for_t = torch.zeros_like(t)
    mask_for_t[t!=0] = 1
    if mask_for_t.sum().item() == 0:
        return None
    num_graph = torch.unique(batch_index[mask_for_t==1]).numel()
    L_t_total = 0
    for i in range(num_graph):
        predicted = predicted_epsilon[batch_index==torch.unique(batch_index[mask_for_t==1])[i]]
        target = target_epsilon[batch_index==torch.unique(batch_index[mask_for_t==1])[i]]
        w_t = 1 - diffusion_process.get_SNR(t-1)/diffusion_process.get_SNR(t)
        w_t = w_t[batch_index==torch.unique(batch_index[mask_for_t==1])[i]][0].item()
        L_t = 0.5 * w_t * torch.mean((predicted - target)**2)
        L_t_total += L_t
    L_t = L_t_total / num_graph
    return L_t
    

def get_loss(prediction,target,batch_index):
    #loss = torch.sum(torch.norm(target-prediction,dim=1)**2)
    #loss = torch.mean(torch.norm(target-prediction,dim=1)**2)
    loss_total = (target-prediction)**2
    num_graph = batch_index.max().item()+1
    loss = 0
    for i in range(num_graph):
        loss += torch.mean(loss_total[batch_index==i])
    loss = loss / num_graph
    return loss

def get_exact_loss(prediction,target,diffusion_process,t,batch_data):
    batch_index = batch_data.batch
    L_0 = get_L0(diffusion_process,prediction,target,t,batch_index)
    L_T = get_L_T(diffusion_process,batch_data)
    L_t = get_L_t(diffusion_process,prediction,target,t,batch_index)
    if L_t is None:
        ELBO = L_0 * (diffusion_process.num_diffusion_timestep+1) + L_T
    else:
        ELBO = L_0 + diffusion_process.num_diffusion_timestep * L_t + L_T
    negative_loss = -ELBO
    return negative_loss




def diffuse_as_batch(batch_data,graph_index,params,diffusion_process:E3DiffusionProcess):
    #parameterの定義
    num_diffusion_timestep = params['num_diffusion_timestep']
    spectrum_size = params['spectrum_size']
    onehot_scaling_factor = params['onehot_scaling_factor']

    #各グラフの拡散時間をランダムに選ぶためのリスト
    #time_list = [i for i in range(1,num_diffusion_timestep+1)]
    time_list = [i for i in range(num_diffusion_timestep+1)]

    num_graph = graph_index.max().item()+1
    pos_before_diffusion, h_before_diffusion, pos_after_diffusion,h_after_diffusion, y_for_noise_pos, y_for_noise_h, each_time_list = [],[],[],[],[],[],[]
    
    # graph_indexをGPUに移動
    graph_index = graph_index.to(batch_data.pos.device)    
    
    #バッチ内のグラフごとに処理を行う
    for i in range(num_graph):
        num_atom = batch_data.pos[graph_index==i].shape[0]

        #拡散させる時間をランダムに選ぶ
        time = random.choice(time_list)

        #posの拡散
        pos_before_dif = batch_data.pos[graph_index==i]
        pos_after_dif, noise_pos = diffusion_process.diffuse_zero_to_t(pos_before_dif,time,mode='pos')
        pos_before_diffusion.append(pos_before_dif)
        pos_after_diffusion.append(pos_after_dif)
        y_for_noise_pos.append(noise_pos)

        #hの拡散 hは連結によって元素種、spectrum、時間の情報を持つが拡散させるのは元素種のみ
        x_before_dif = batch_data.x[graph_index==i]
        h_after_dif, noise_h = diffusion_process.diffuse_zero_to_t(x_before_dif,time,mode='h')
        h_before_diffusion.append(x_before_dif)
        h_after_diffusion.append(h_after_dif)
        y_for_noise_h.append(noise_h)
        #グラフごとに拡散させた時間を記録
        each_time_list.append(torch.stack([torch.tensor([time]) for j in range(num_atom)]))

    #グラフごとに処理したデータをバッチとしてまとめる
    pos_before_diffusion = torch.cat(pos_before_diffusion,dim=0)
    h_before_diffusion = torch.cat(h_before_diffusion,dim=0)
    pos_after_diffusion = torch.cat(pos_after_diffusion,dim=0)
    h_after_diffusion = torch.cat(h_after_diffusion,dim=0)
    y_for_noise_pos = torch.cat(y_for_noise_pos,dim=0)
    y_for_noise_h = torch.cat(y_for_noise_h,dim=0)
    each_time_list = torch.cat(each_time_list,dim=0)

    #batch_dataに拡散後のデータを格納
    batch_data.pos_at_t = pos_after_diffusion 
    batch_data.h_at_t = h_after_diffusion
    batch_data.pos_at_0 = pos_before_diffusion
    batch_data.h_at_0 = h_before_diffusion
    batch_data.y_for_noise_pos = y_for_noise_pos
    batch_data.y_for_noise_h = y_for_noise_h
    batch_data.each_time_list = each_time_list

    return batch_data
    



        

def train_epoch(nn_dict,train_loader,params,diffusion_process,optimizer,scheduler=None,ema=None,ema_model=None,flag='normal'):
    egnn = nn_dict['egnn']
    egnn.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    egnn.to(device)
    param_list = list(egnn.parameters())
    if params['optimizer']['type'] == 'RAdamScheduleFree':
        optimizer.train()
    criterion = nn.MSELoss(reduction='sum')

    if params['NN_encoder']['spectrum_to_latent']:
        spectrum_compressor = nn_dict['spectrum_compressor']
        spectrum_compressor.train()
        param_list += list(spectrum_compressor.parameters())

    if params['noise_schedule'] == 'learned':
        diffusion_process.gamma.train()
        param_list += list(diffusion_process.gamma.parameters())

    #parameterの定義
    num_diffusion_timestep = params['num_diffusion_timestep']
    atom_type_size = params['atom_type_size']
    conditional = params['conditional']

    epoch_loss_train = 0
    total_num_train_node = 0   

    judge = EdgeIndexJudge(cutoff=3.0, knn=8)

    for train_data in train_loader:
        train_data = train_data.to(device)
        optimizer.zero_grad()
        total_num_train_node += train_data.num_nodes
        graph_index = train_data.batch
        num_graph = graph_index.max().item()+1

        #拡散後のデータを取得
        train_data = diffuse_as_batch(train_data,graph_index,params,diffusion_process)

        #judge.set_graph(train_data)
        #train_data = judge.pass_edges_knn()
        #if hasattr(train_data, 'num_nodes'):
            #print(f"Number of nodes in the batch: {train_data.num_nodes}")
            #del train_data.num_nodes

        
        train_data.edge_index = torch_geometric.nn.knn_graph(train_data.pos_at_t.cpu(), k=8, batch=train_data.batch.cpu()).cpu()
        train_data.edge_index = train_data.edge_index.to(device)

        
        #時間のデータをnum_diffusion_timestepで正規化
        time_data = train_data.each_time_list / num_diffusion_timestep

        #egnnに渡すhのデータを定義
        h_for_input = train_data.h_at_t.to(device)
        if conditional:
            if params['NN_encoder']['spectrum_to_latent']:
                compressed_spectrum = spectrum_compressor(train_data.spectrum.to(device))
                h_for_input = torch.cat((h_for_input,compressed_spectrum.to(device)),dim=1).to(device)
            elif params["classifier_free"]["use_classifier_free"]:
                if torch.rand(1).item() < params['classifier_free']['dropout_rate']:
                    h_for_input = torch.cat((h_for_input,torch.zeros_like(train_data.spectrum.to(device))),dim=1).to(device)
                else:
                    h_for_input = torch.cat((h_for_input,train_data.spectrum.to(device)),dim=1).to(device)
            else:
                h_for_input = torch.cat((h_for_input,train_data.spectrum.to(device)),dim=1).to(device)
        if params['give_excited_atom']:
            h_for_input = torch.cat((h_for_input,train_data.excited_atom.to(device)),dim=1)
        h_for_input = torch.cat((h_for_input,time_data.to(device)),dim=1)
        
        
        
        #equivariant graph neural networkによる予測
        if flag == 'gemnet':
            epsilon_h, epsilon_x = egnn(train_data)
        elif flag == 'dimenet':
            train_data = train_data.to(device)
            h = egnn.initial_embedding(train_data.h_at_t, train_data.pos_at_t, train_data.spectrum, train_data.each_time_list, train_data.edge_index[0], train_data.edge_index[1])
            epsilon_h, epsilon_x = egnn(train_data.edge_index, h, train_data.pos_at_t)
        elif flag == 'normal':
            epsilon_h, epsilon_x = egnn(train_data.edge_index.to(device),h_for_input,train_data.pos_at_t.to(device))
        
                
        #lossの計算
        predicted_epsilon = torch.cat((epsilon_x,epsilon_h),dim=1) #予測したepsilon
        target_epsilon = torch.cat((train_data.y_for_noise_pos,train_data.y_for_noise_h),dim=1) #正解のepsilon
        loss = get_loss(predicted_epsilon,target_epsilon,graph_index)

        if loss > 1000:
            print(train_data.id)

        #lossのbackward逆伝播
        loss.backward()

        #勾配のクリッピング
        grad_norm = nn.utils.clip_grad_norm_(param_list,params['max_grad_norm'])
        if grad_norm > 1000:
            print('grad norm is too large :',grad_norm)

        #schedulerによる学習率の更新
        if scheduler is not None and params['lr_scheduler']['type'] != 'ReduceLROnPlateau':
            scheduler.step()

        #optimizerによるパラメータの更新
        optimizer.step()

        #emaの更新
        if ema is not None and ema_model is not None:
            ema.update_model_average(ema_model,egnn)

        #lossの記録
        epoch_loss_train += loss.item() * train_data.num_graphs
        
    avg_loss_train = epoch_loss_train / len(train_loader.dataset)

    return avg_loss_train

def eval_epoch(nn_dict,eval_loader,params,diffusion_process,optimizer,scheduler=None,ema=None,ema_model=None,flag='normal'):
    if params['EMA']['use_ema']:
        egnn = ema_model
    else:
        egnn = nn_dict['egnn']
    egnn.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    egnn.to(device)
    if params['optimizer']['type'] == 'RAdamScheduleFree':
        optimizer.eval()
    criterion = nn.MSELoss(reduction='sum')

    if params['NN_encoder']['spectrum_to_latent']:
        spectrum_compressor = nn_dict['spectrum_compressor']
        spectrum_compressor.eval()

    if params['noise_schedule'] == 'learned':
        diffusion_process.gamma.eval()

    #parameterの定義
    num_diffusion_timestep = params['num_diffusion_timestep']
    conditional = params['conditional']
    atom_type_size = params['atom_type_size']

    epoch_loss_val = 0
    total_num_val_node = 0

    judge = EdgeIndexJudge(cutoff=3.0, knn=8)

    with torch.no_grad():
        for val_data in eval_loader:
            val_data = val_data.to(device)
            total_num_val_node += val_data.num_nodes
            graph_index = val_data.batch
            num_diffusion_timestep = params['num_diffusion_timestep']                

            #拡散後のデータを取得
            val_data = diffuse_as_batch(val_data,graph_index,params,diffusion_process)

            #judge.set_graph(val_data)
            #val_data = judge.pass_edges_knn()

            val_data.edge_index = torch_geometric.nn.knn_graph(val_data.pos_at_t.cpu(), k=8, batch=val_data.batch.cpu()).cpu()
            val_data.edge_index = val_data.edge_index.to(device)

            #時間のデータをnum_diffusion_timestepで正規化
            time_data = val_data.each_time_list / num_diffusion_timestep

            #egnnに渡すのデータを定義
            h_for_input = val_data.h_at_t.to(device)
            if conditional:
                if params['NN_encoder']['spectrum_to_latent']:
                    compressed_spectrum = spectrum_compressor(val_data.spectrum.to(device))
                    h_for_input = torch.cat((h_for_input,compressed_spectrum.to(device)),dim=1).to(device)
                else:
                    h_for_input = torch.cat((h_for_input,val_data.spectrum.to(device)),dim=1).to(device)
            if params['give_excited_atom']:
                h_for_input = torch.cat((h_for_input,val_data.excited_atom.to(device)),dim=1)
            h_for_input = torch.cat((h_for_input,time_data.to(device)),dim=1)
            
            
            
            #equivariant graph neural networkによる予測
            if flag == 'gemnet':
                epsilon_h, epsilon_x = egnn(val_data)
            elif flag == 'dimenet':
                val_data = val_data.to(device)
                h = egnn.initial_embedding(val_data.h_at_t, val_data.pos_at_t, val_data.spectrum, val_data.each_time_list, val_data.edge_index[0], val_data.edge_index[1])
                epsilon_h, epsilon_x = egnn(val_data.edge_index, h, val_data.pos_at_t)
            elif flag == 'normal':
                epsilon_h, epsilon_x = egnn(val_data.edge_index.to(device),h_for_input,val_data.pos_at_t.to(device))
            

            #lossの計算
            predicted_epsilon = torch.cat((epsilon_x,epsilon_h),dim=1)
            target_epsilon = torch.cat((val_data.y_for_noise_pos,val_data.y_for_noise_h),dim=1)
            loss = get_loss(predicted_epsilon,target_epsilon,graph_index)

            #lossの記録
            epoch_loss_val += loss.item() * val_data.num_graphs

    avg_loss_val = epoch_loss_val / len(eval_loader.dataset) #各ノードごとのlossの平均

    if params['lr_scheduler']['type'] == 'ReduceLROnPlateau':
        scheduler.step(avg_loss_val)

    return avg_loss_val

def generate(nn_dict,test_data,params,diffusion_process,gen_num_per_spectrum=5,ema=None,ema_model=None,check=False,flag='normal'):
    #モデルの読み込み
    if params['EMA']['use_ema']:
        egnn = ema_model
    else:
        egnn = nn_dict['egnn']
    egnn.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    egnn.to(device)
    if params['NN_encoder']['spectrum_to_latent']:
        spectrum_compressor = nn_dict['spectrum_compressor']
        spectrum_compressor.eval()
        spectrum_compressor.to(device)
    if params['noise_schedule'] == 'learned':
        diffusion_process.gamma.eval()
        diffusion_process.gamma.to(device)
    
    #parameterの定義
    num_diffusion_timestep = params['num_diffusion_timestep']
    conditional = params['conditional']
    atom_type_size = params['atom_type_size']
    spectrum_size = params['spectrum_size']
    onehot_scaling_factor = params['onehot_scaling_factor']

    judge = EdgeIndexJudge(cutoff=3.0, knn=8)
    
    original_graph_list ,generated_graph_list = [],[]

    with torch.no_grad():
        for batch in tqdm(test_data):
            data = batch.to(device)

            #生成するデータサイズを取得（本来は予測する）
            num_of_atoms = data.x.shape[0]

            #一つの条件に対して生成するデータ数と実際に生成したデータ数、nanの数をカウント
            how_many_gen = gen_num_per_spectrum
            num_of_generated_data = 0
            num_of_generated_nan = 0


            while num_of_generated_data != how_many_gen and num_of_generated_nan < how_many_gen:
                #初期値の設定
                initial_pos = torch.zeros(size=(num_of_atoms,3), device=device)
                initial_pos.normal_(mean=0,std=1)
                initial_pos = remove_mean(initial_pos)
                initial_h = torch.zeros(size=(num_of_atoms,atom_type_size), device=device)
                initial_h.normal_(mean=0,std=1)

                #グラフデータのedge_indexを生成
                edge_index = []
                for i in range(num_of_atoms):
                    for j in range(num_of_atoms):
                        if i != j:
                            edge_index.append([i,j])
                edge_index = torch.tensor(edge_index,dtype=torch.long, device=device).t().contiguous()


                #初期値をData型に変換
                graph = Data(x=initial_h,edge_index=edge_index,pos=initial_pos)
                if conditional:
                    graph.spectrum = data.spectrum
                if params['give_excited_atom']:
                    graph.excited_atom = data.excited_atom

                graph.num_graphs = 1
                graph.batch = torch.tensor([0 for i in range(num_of_atoms)],dtype=torch.long, device=device)



                
                #100stepごとのデータを格納するリスト
                transition_data_per_100step = []

                time_list = range(num_diffusion_timestep, 0, -1) #逆拡散のために時間を降順にする


                #逆拡散
                for t in time_list:
                    if t%100 == 0:
                        #print(copy.deepcopy(graph).pos)
                        transition_data_per_100step.append(copy.deepcopy(graph))

                    #judge.set_graph(graph)
                    #graph = judge.pass_edges_knn()

                    #edge_index = torch_geometric.nn.knn_graph(graph.pos.cpu(), k=8, batch=graph.batch.cpu()).cpu()
                    #graph.edge_index = edge_index.to('cuda')


                    #時間のデータをnum_diffusion_timestepで正規化
                    time_tensor = torch.tensor([[t/num_diffusion_timestep] for d in range(num_of_atoms)],dtype=torch.float32, device=device)

                    #特徴量ベクトルの定義
                    graph.h = onehot_scaling_factor*graph.x
                    if conditional:
                        if params['NN_encoder']['spectrum_to_latent']:
                            compressed_spectrum = spectrum_compressor(graph.spectrum)
                            graph.h = torch.cat((graph.h,compressed_spectrum),dim=1)
                        else:
                            graph.h = torch.cat((graph.h,graph.spectrum),dim=1)
                    if params['give_excited_atom']:
                        graph.h = torch.cat((graph.h,graph.excited_atom),dim=1)
                    graph.h = torch.cat((graph.h,time_tensor),dim=1)

                    
                    #equivariant graph neural networkによる予測
                    if flag == 'gemnet':
                        epsilon_h, epsilon_x = egnn(graph)
                    elif flag == 'dimenet':
                        graph = graph.to(device)
                        graph.each_time_list = torch.tensor([[t] for _ in range(time_tensor.shape[0])],dtype=torch.long, device=device)
                        h = egnn.initial_embedding(graph.x, graph.pos, graph.spectrum, graph.each_time_list, graph.edge_index[0], graph.edge_index[1])
                        graph.h = graph.x
                        assert graph.h.shape[1] == atom_type_size
                        epsilon_h, epsilon_x = egnn(graph.edge_index, h, graph.pos)
                        assert epsilon_h.shape[1] == atom_type_size
                    elif flag == 'normal':
                        epsilon_h, epsilon_x = egnn(graph.edge_index,graph.h,graph.pos)
                    epsilon_x = remove_mean(epsilon_x)  #並進を考慮

                    #classifier-free guidance
                    if params["classifier_free"]["use_classifier_free"]:
                        #classifier-freeのための無条件特徴量ベクトル
                        unconditional_h = onehot_scaling_factor*graph.x
                        unconditional_spectrum = torch.zeros_like(graph.spectrum)
                        if conditional:
                            if params['NN_encoder']['spectrum_to_latent']:
                                compressed_spectrum = spectrum_compressor(unconditional_spectrum)
                                unconditional_h = torch.cat((unconditional_h,unconditional_spectrum),dim=1)
                            else:
                                unconditional_h = torch.cat((unconditional_h,unconditional_spectrum),dim=1)
                        if params['give_excited_atom']:
                            unconditional_h = torch.cat((unconditional_h,graph.excited_atom),dim=1)
                        unconditional_h = torch.cat((unconditional_h,time_tensor),dim=1)

                        #無条件特徴量ベクトルを用いて予測
                        if flag == 'gemnet':
                            epsilon_unconditional_h, epsilon_unconditional_x = egnn(graph)
                        elif flag == 'dimenet':
                            pass
                        elif flag == 'normal':
                            epsilon_unconditional_h, epsilon_unconditional_x = egnn(graph.edge_index,unconditional_h,graph.pos)
                        epsilon_unconditional_x = remove_mean(epsilon_unconditional_x)
                        
                        #epsilon_for_cfg
                        classifier_free_weight = params['classifier_free']['classifier_free_weight']
                        epsilon_x = (1 + classifier_free_weight) * epsilon_x - classifier_free_weight * epsilon_unconditional_x
                        epsilon_h = (1 + classifier_free_weight) * epsilon_h - classifier_free_weight * epsilon_unconditional_h

                    #逆拡散
                    graph.pos = diffusion_process.reverse_diffuse_one_step(graph.pos,epsilon_x,t,mode='pos')
                    graph.x = diffusion_process.reverse_diffuse_one_step(graph.h[:,:atom_type_size],epsilon_h,t,mode='h')

                    #nanが出力されていないかの確認
                    if not torch.isfinite(graph.x).all():
                        num_of_generated_nan += 1
                        #print(f'NaN was generated at timestep {t}')
                        #seed_value += 1
                        if num_of_generated_nan == how_many_gen:
                            print('too much nan was generated for atom type generation')
                            print(f'NaN was generated at timestep {t}')
                            print(f'structure id is {data.id}') 
                            continue
                            #exit()
                        break
                    elif not torch.isfinite(graph.pos).all():
                        num_of_generated_nan += 1
                        #print(f'NaN was generated at timestep {t}')
                        #seed_value += 1
                        if num_of_generated_nan == how_many_gen:
                            print('too much nan was generated for coordinate generation')
                            print(f'NaN was generated at timestep {t}')
                            print(f'structure id is {data.id}')
                            continue
                            #exit()
                        break
                    else:
                        #print(f'timestep {t} is done')
                        pass

                
                time_tensor = torch.tensor([[0] for d in range(num_of_atoms)],dtype=torch.float32)
                graph.h = onehot_scaling_factor*graph.x
                if conditional:
                    if params['NN_encoder']['spectrum_to_latent']:
                        compressed_spectrum = spectrum_compressor(graph.spectrum)
                        graph.h = torch.cat((graph.h,compressed_spectrum),dim=1)
                    else:
                        graph.h = torch.cat((graph.h,graph.spectrum),dim=1)
                if params['give_excited_atom']:
                    graph.h = torch.cat((graph.h,graph.excited_atom),dim=1)
                graph.h = torch.cat((graph.h,time_tensor),dim=1)
                if flag == 'gemnet':
                    new_h, new_x = egnn(graph)
                elif flag == 'dimenet':
                    h = egnn.initial_embedding(graph.x, graph.pos, graph.spectrum, graph.each_time_list, graph.edge_index[0], graph.edge_index[1])
                    new_h, new_x = egnn(graph.edge_index, h, graph.pos)
                    graph.h = graph.x
                    assert new_h.shape[1] == atom_type_size
                elif flag == 'normal':
                    new_h, new_x = egnn(graph.edge_index,graph.h,graph.pos)
                epsilon_x = remove_mean(new_x - graph.pos)
                graph.h = graph.h[:,:atom_type_size]
                epsilon_h = new_h[:,:atom_type_size]
                alpha_0 = diffusion_process.alpha(0)
                sigma_0 = diffusion_process.sigma(0)
                mu_x = graph.pos / alpha_0 - sigma_0 * epsilon_x / alpha_0
                noise_x = torch.zeros_like(mu_x)
                noise_x.normal_(mean=0,std=1)
                noise_x = remove_mean(noise_x)
                graph.pos = mu_x + sigma_0 * noise_x / alpha_0
                mu_h = graph.h / alpha_0 - sigma_0 * epsilon_h / alpha_0
                noise_h = torch.zeros_like(mu_h)
                noise_h.normal_(mean=0,std=1)
                graph.h = mu_h + sigma_0 * noise_h / alpha_0
                h_atoms = nn.functional.one_hot(torch.argmax(graph.h,dim=1),num_classes=2)
                graph.x = h_atoms

                #t=0におけるデータを格納
                if torch.isfinite(graph.x).all() and torch.isfinite(graph.pos).all():
                    transition_data_per_100step.append(graph)
                    #座標の値が現実的でないものは除外(1000Å以上の座標が出力された場合)
                    if torch.any(graph.pos > 1000):
                        continue
                    num_of_generated_data += 1
                    


                    if conditional:
                        original_graph_list.append(data)
                    else:
                        original_graph_list.append(-1)
                    generated_graph_list.append(transition_data_per_100step)
                
    return original_graph_list,generated_graph_list


                    

                        



        
        
