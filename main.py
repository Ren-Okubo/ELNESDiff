import argparse
import sys
import os
import wandb
import yaml
import random
import torch
import datetime
import pytz
import numpy as np
import matplotlib.pyplot as plt
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.datasets as datasets
from torch.utils.data import random_split
from schedulefree import RAdamScheduleFree

sys.path.append('./module/')
sys.path.append('./utils/')
from train_per_iretation import diffuse_as_batch, train_epoch, eval_epoch, generate, EarlyStopping
from loss_calculation import kabsch_torch
from def_for_main import load_model_state, evaluate_by_rmsd, noise_schedule_for_GammaNetwork, evaluate_by_rmsd_and_atom_type_eval, define_optimizer, define_lr_scheduler, EMA

from EquivariantGraphNeuralNetwork import EquivariantGNN
from diffusion_x_h import E3DiffusionProcess
from split_to_train_and_test import SetUpData
from DataPreprocessor import SpectrumCompressor
from make_xyz_from_wandb_run import write_xyz


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name',type=str,default='Si_O_ELNES')
    parser.add_argument('--run_name',type=str,default=None)
    parser.add_argument('--dataset_path',type=str,default="./dataset/Si_O_ELNES.pt")
    parser.add_argument('--mode',type=str,default='train_and_generate') #train_and_generate, train_only, generate_only, evaluate_only
    parser.add_argument('--training_run_id',type=str,default=None) #train_onlyやgenerate_onlyのとき、モデルの状態を読み込むために必要
    parser.add_argument('--record_schedule',type=bool,default=False)
    parser.add_argument('--create_xyz_file',type=bool,default=False)
    parser.add_argument('--note',type=str,default=None)
    parser.add_argument('--scaling', type=str, default="area") #max or area
    parser.add_argument('--model_version', type=str, default='newest') #newest or old
    args = parser.parse_args()

    if args.model_version == 'old':
        from EquivariantGraphNeuralNetwork import EquivariantGNNold as EquivariantGNN

    #parameterの読み込み
    with open ('parameters.yaml','r') as file:
        prms = yaml.safe_load(file)
    jst=pytz.timezone('Asia/Tokyo')
    now = datetime.datetime.now(jst)
    prms['now'] = now.strftime("%Y%m%d%H%M")
    assert prms['NN_encoder']['spectrum_to_latent'] == False or prms['Encoder_Decoder']['spectrum_to_latent'] == False

    #wandbの設定
    assert args.mode in ['train_and_generate','train_only','generate_only','evaluate_only']
    if args.mode == 'train_and_generate' or args.mode == 'train_only':
        run = wandb.init(project=args.project_name,config=prms,name=args.run_name)
    elif args.mode == 'generate_only' or args.mode == 'evaluate_only':
        run_id = args.training_run_id
        run = wandb.init(project=args.project_name,id=run_id,resume='must')
        prms = run.config

    #メモがあればnoteに記録
    if args.note:
        wandb.run.notes = args.note

    if torch.__version__ >= "2.7.1":
        torch.serialization.add_safe_globals([
            torch_geometric.data.data.DataEdgeAttr,
            torch_geometric.data.data.DataTensorAttr,
            torch_geometric.data.storage.GlobalStorage,
            torch_geometric.nn.aggr.basic.SumAggregation,
            torch_geometric.inspector.Inspector,
            torch_geometric.inspector.Signature,
            torch_geometric.inspector.Parameter,
            torch.nn.modules.container.ModuleList,
            torch.nn.modules.container.Sequential,
            torch.nn.modules.linear.Linear,
            torch.nn.modules.activation.SiLU,
            torch.nn.modules.activation.Sigmoid,
            int
        ])


    #パラメータの設定
    #全体のパラメータ
    conditional=prms['conditional']
    seed = prms['seed']
    num_epochs = prms['num_epochs']
    batch_size = prms['batch_size']
    #optimizerのパラメータ
    lr = prms['optimizer']['lr']
    weight_decay = prms['optimizer']['weight_decay']
    optim_type = prms['optimizer']['type']
    scheduler = prms['optimizer']['scheduler']
    #early stoppingのパラメータ
    patience = prms['patience']
    #diffusion_processのパラメータ
    num_diffusion_timestep = prms['num_diffusion_timestep']
    noise_schedule = prms['noise_schedule']
    noise_precision = prms['noise_precision']
    power = prms['noise_schedule_power']
    #spectrum_compressorのパラメータ
    to_compress_spectrum = prms['NN_encoder']['spectrum_to_latent']
    compressor_hidden_dim = prms['NN_encoder']['compressed_hidden_dim']
    compressed_spectrum_size = prms['NN_encoder']['compressed_spectrum_size']
    #egnnのパラメータ
    L = prms['L']
    atom_type_size = prms['atom_type_size']
    spectrum_size = prms['spectrum_size']
    if prms['Encoder_Decoder']['spectrum_to_latent']:
        spectrum_size = prms['Encoder_Decoder']['latent_dim']
    d_size = prms['d_size']
    t_size = prms['t_size']
    excited_atom_size = prms['excited_atom_size']
    if conditional:
        if prms['NN_encoder']['spectrum_to_latent']:
            h_size = atom_type_size + compressed_spectrum_size + t_size
        else:
            h_size = atom_type_size + spectrum_size + t_size
    else:
        h_size = atom_type_size + t_size
    if prms['give_excited_atom']:
        h_size = h_size + excited_atom_size
    x_size = prms['x_size']
    m_size = prms['m_size']   
    m_input_size = h_size + h_size + d_size
    m_hidden_size = prms['m_hidden_size']
    m_output_size = m_size
    h_input_size = h_size + m_size
    h_hidden_size = prms['h_hidden_size']
    h_output_size = h_size 
    x_input_size = h_size + h_size + d_size
    x_hidden_size = prms['x_hidden_size']
    x_output_size = 1
    onehot_scaling_factor = prms['onehot_scaling_factor']

    #default_tensor_typeの設定
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    #seedの設定
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


    setupdata = SetUpData(seed=seed,conditional=conditional)    

    #deviceの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 乱数生成器を作成し、CUDAデバイスに設定
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)  # 任意のシード値を設定

    #datasetの読み込み
    dataset_path = args.dataset_path
    wandb.config.update({'dataset_path':dataset_path})
    if os.path.isdir(dataset_path):
        dataset_files = os.listdir(dataset_path)
        dataset_files = [file for file in dataset_files if file.endswith('.pt')]
        train_dataset_files = [file for file in dataset_files if 'train' in file][0]
        valid_dataset_files = [file for file in dataset_files if 'valid' in file][0]
        test_dataset_files = [file for file in dataset_files if 'test' in file][0]
        
        train_data = torch.load(os.path.join(dataset_path,train_dataset_files))
        val_data = torch.load(os.path.join(dataset_path,valid_dataset_files))
        test_data = torch.load(os.path.join(dataset_path,test_dataset_files))
        dataset_length = len(train_data) + len(val_data) + len(test_data)
        wandb.config.update({'dataset_size':dataset_length})
        prms = run.config

        #normalize spectrum and set spectrum size
        for dataset in [train_data, val_data, test_data]:
            for data in dataset:
                if args.scaling == None:  
                    spectrum = data.spectrum_raw
                elif args.scaling == 'max':
                    spectrum = data.spectrum_raw[0].cpu().numpy()
                    spectrum_scaling_max = spectrum / np.max(spectrum)
                    new_spectrum = torch.zeros_like(data.spectrum_raw)
                    new_spectrum[0] = torch.tensor(spectrum_scaling_max).to('cuda')
                    data.spectrum = new_spectrum
                    if prms['give_excited_atom']:
                        spectrum[:] = spectrum[0]
                elif args.scaling == 'area':
                    spectrum = data.spectrum_raw[0].cpu().numpy()
                    spectrum_scaling_area = spectrum / np.sum(spectrum)
                    new_spectrum = torch.zeros_like(data.spectrum_raw)
                    new_spectrum[0] = torch.tensor(spectrum_scaling_area).to('cuda')
                    data.spectrum = new_spectrum
                    spectrum = data.spectrum
                resized_spectrum = spectrum[:,:spectrum_size]
                data.spectrum = resized_spectrum


    else:
        dataset = torch.load(dataset_path)
        for data in dataset: #spectrumサイズの設定（-1~19eVがデフォルト）
            if args.scaling == None:  
                spectrum = data.spectrum_raw
            elif args.scaling == 'max':
                spectrum = data.spectrum_raw[0].cpu().numpy()
                spectrum_scaling_max = spectrum / np.max(spectrum)
                new_spectrum = torch.zeros_like(data.spectrum_raw)
                new_spectrum[0] = torch.tensor(spectrum_scaling_max).to('cuda')
                data.spectrum = new_spectrum
                if prms['give_excited_atom']:
                    spectrum[:] = spectrum[0]
            elif args.scaling == 'area':
                spectrum = data.spectrum_raw[0].cpu().numpy()
                spectrum_scaling_area = spectrum / np.sum(spectrum)
                new_spectrum = torch.zeros_like(data.spectrum_raw)
                new_spectrum[0] = torch.tensor(spectrum_scaling_area).to('cuda')
                data.spectrum = new_spectrum
                spectrum = data.spectrum
            resized_spectrum = spectrum[:,:spectrum_size]
            data.spectrum = resized_spectrum



        #dataset内のグラフに孤立原子からなるグラフがある場合、削除
        dataset = [data for data in dataset if data.pos.shape[0] > 1]
        wandb.config.update({'dataset_size':len(dataset)},allow_val_change=True)
        prms = run.config

        #datasetの分割
        train_data, val_data, test_data = setupdata.split(dataset)

    #additional datasetの追加
    if prms['dataset']['add_to_train_data']:
        train_data_list = [train_data.dataset[i] for i in train_data.indices]
        for i in range(len(prms['dataset']['additional_dataset_list'])):
            additional_dataset = torch.load(prms['dataset']['additional_dataset_list'][i])
            additional_dataset = [data for data in additional_dataset if data.pos.shape[0] > 1]
            for data in additional_dataset:
                if args.scaling == None:
                    continue
                elif args.scaling == 'max':
                    spectrum = data.spectrum_raw[0].cpu().numpy()
                    spectrum_scaling_max = spectrum / np.max(spectrum)
                    new_spectrum = torch.zeros_like(data.spectrum_raw)
                    new_spectrum[0] = torch.tensor(spectrum_scaling_max).to('cuda')
                    data.spectrum = new_spectrum
                elif args.scaling == 'area':
                    spectrum = data.spectrum_raw[0].cpu().numpy()
                    spectrum_scaling_area = spectrum / np.sum(spectrum)
                    new_spectrum = torch.zeros_like(data.spectrum_raw)
                    new_spectrum[0] = torch.tensor(spectrum_scaling_area).to('cuda')
                    data.spectrum = new_spectrum
            train_data_list += additional_dataset
        print(f'train dataset size : {len(train_data_list)}')
        train_data = train_data_list
        wandb.config.update({'dataset_size':len(train_data)+len(val_data)+len(test_data)},allow_val_change=True)
    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,generator=generator)
    val_loader = DataLoader(val_data,batch_size=batch_size,shuffle=True,generator=generator)


    #modelの定義
    diffusion_process = E3DiffusionProcess(s=noise_precision,power=power,num_diffusion_timestep=num_diffusion_timestep,noise_schedule=noise_schedule)
    egnn = EquivariantGNN(L,m_input_size,m_hidden_size,m_output_size,x_input_size,x_hidden_size,x_output_size,h_input_size,h_hidden_size,h_output_size)
    if to_compress_spectrum:
        spectrum_compressor = SpectrumCompressor(spectrum_size,compressor_hidden_dim,compressed_spectrum_size)
    early_stopping = EarlyStopping(patience=patience)

    #使用するモデルをまとめた辞書nn_dictを定義
    if to_compress_spectrum:
        nn_dict = {'egnn':egnn,'spectrum_compressor':spectrum_compressor}
    else:
        nn_dict = {'egnn':egnn,'spectrum_compressor':None}
    


    #optimizerの設定
    optimizer = define_optimizer(prms,nn_dict,diffusion_process,optim_type=optim_type)

    #schedulerの設定
    if scheduler:
        scheduler = define_lr_scheduler(optimizer,prms)
    else:
        scheduler = None

    #EMAの設定
    if prms['EMA']['use_ema']:
        ema = EMA(beta=prms['EMA']['beta'])
        ema_model = EquivariantGNN(L,m_input_size,m_hidden_size,m_output_size,x_input_size,x_hidden_size,x_output_size,h_input_size,h_hidden_size,h_output_size)
    else:
        ema = None
        ema_model = None

    if "train" in args.mode:
        #train,val
        for epoch in range(num_epochs):
            avg_loss_train = train_epoch(nn_dict,train_loader,prms,diffusion_process,optimizer,scheduler=scheduler,ema=ema,ema_model=ema_model)
            avg_loss_val = eval_epoch(nn_dict,val_loader,prms,diffusion_process,optimizer,scheduler=scheduler,ema=ema,ema_model=ema_model)

            #各epochにおけるlossを記録
            print(f'epoch:{epoch}    train_loss:{avg_loss_train}     val_loss:{avg_loss_val}')
            wandb.log({'train_loss':avg_loss_train,'val_loss':avg_loss_val})

            #early stopping
            if early_stopping.validate(avg_loss_val):
                break
        
        #モデルの保存
        model_states = {'egnn':egnn.state_dict()}
        if prms['NN_encoder']['spectrum_to_latent']:
            model_states['spectrum_compressor'] = spectrum_compressor.state_dict()
        if prms['noise_schedule'] == 'learned':
            model_states['gamma'] = diffusion_process.gamma.state_dict()
        if prms['EMA']['use_ema']:
            model_states['egnn_ema'] = ema_model.state_dict()
        
        model_save_path = os.path.join(wandb.run.dir,'model.pth')
        torch.save(model_states,model_save_path)
        wandb.config.update({'model_save_path':model_save_path})
        print(f'model saved at {model_save_path}')

        #trainのみの場合
        if args.mode == 'only_train':
            wandb.finish()
            sys.exit()

    #generateのみの場合モデルの状態を読み込む
    if args.mode == 'generate_only':
        model_save_path = wandb.config['model_save_path']
        load_model_state(nn_dict,model_save_path,prms)

    if "generate" in args.mode:
        #generate
        original_graph_list, generated_graph_list = generate(nn_dict,test_data,prms,diffusion_process,gen_num_per_spectrum=5,ema=ema,ema_model=ema_model)

        #生成したグラフを保存
        generated_graph_save_path = os.path.join(wandb.run.dir,'generated_graph.pt')
        torch.save(generated_graph_list,generated_graph_save_path)
        wandb.config.update({'generated_graph_save_path':generated_graph_save_path})
        print(f'generated graph saved at {generated_graph_save_path}')
        if conditional:
            original_graph_save_path = os.path.join(wandb.run.dir,'original_graph.pt')
            torch.save(original_graph_list,original_graph_save_path)
            wandb.config.update({'original_graph_save_path':original_graph_save_path})
            print(f'original graph saved at {original_graph_save_path}')
    
    #evaluateのみの場合
    if args.mode == 'evaluate_only':
        original_graph_list = torch.load(wandb.config['original_graph_save_path'])
        generated_graph_list = torch.load(wandb.config['generated_graph_save_path'])


    #生成したデータの評価
    if conditional:
        
        #生成したグラフのrmsdを計算し、ソート
        sorted_id_rmsd_original_generated_list = evaluate_by_rmsd(original_graph_list,generated_graph_list)
        
        #rmsdの値を保存 [(id,rmsd,original_graph,generated_graph),...]
        #rmsd_save_path = os.path.join(wandb.run.dir,'rmsd.pt')
        #torch.save(sorted_id_rmsd_original_generated_list,rmsd_save_path)
        #wandb.config.update({'rmsd_save_path':rmsd_save_path},allow_val_change=True)
        #print(f'rmsd saved at {rmsd_save_path}')

        #ソートしたrmsdの描画
        sorted_id_list, sorted_rmsd_list,sorted_original_graph_list,sorted_generated_graph_list = zip(*sorted_id_rmsd_original_generated_list)
        sorted_rmsd_list = torch.tensor(sorted_rmsd_list).cpu().numpy()
        fig, ax = plt.subplots()
        ax.plot(sorted_rmsd_list,marker='o',linestyle='None')
        ax.set_xlabel('sorted_index')
        ax.set_ylabel('rmsd')
        ax.set_yscale('log')
        ax.set_title('rmsd')
        #wandb.log({'rmsd':wandb.Image(fig)})
        plt.close()
        
        #生成したグラフのrmsd,atom_type_scoreを計算し、ソート
        sorted_id_rmsd_atomeval_original_generated_list = evaluate_by_rmsd_and_atom_type_eval(original_graph_list,generated_graph_list)

        soretd_id_list, sorted_rmsd_list, sorted_atom_type_eval_list, sorted_original_graph_list, sorted_generated_graph_list = zip(*sorted_id_rmsd_atomeval_original_generated_list)
        sorted_rmsd_list = torch.tensor(sorted_rmsd_list).cpu().numpy()
        sorted_atom_type_eval_list = torch.tensor(sorted_atom_type_eval_list).cpu().numpy()
        density_of_O_for_original = [i[0] for i in sorted_atom_type_eval_list]
        density_of_O_for_generated = [i[1] for i in sorted_atom_type_eval_list]
        fig, ax = plt.subplots(figsize=(10,10))
        ax.plot([0,1],[0,1],'-',color='red',alpha=0.5)
        ax.plot(density_of_O_for_original,density_of_O_for_generated,'o',alpha=0.5)
        ax.set_xlabel('density of O for original')
        ax.set_ylabel('density of O for generated')
        ax.set_title('atom_type_eval')
        plt.text(0.05, 0.95, 'accuracy: {:.5f}'.format(len([1 for i in range(len(density_of_O_for_original)) if density_of_O_for_generated[i] == density_of_O_for_original[i]])/len(density_of_O_for_original)), 
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        wandb.log({'atom_type_eval':wandb.Image(fig)})
        plt.close()

        
        
        #xyzファイルの作成
        if args.create_xyz_file:
            first_min_rmsd_data = sorted_id_rmsd_original_generated_list[0]
            second_min_rmsd_data = sorted_id_rmsd_original_generated_list[1]
            third_min_rmsd_data = sorted_id_rmsd_original_generated_list[2]
            mid_rmsd_data = sorted_id_rmsd_original_generated_list[int(len(sorted_id_rmsd_original_generated_list)/2)]
            max_rmad_data = sorted_id_rmsd_original_generated_list[-1]
            write_xyz(os.path.join(run.dir,'first_min_rmsd.xyz'),first_min_rmsd_data[2],first_min_rmsd_data[3],comment='first_min_rmsd ' + str(first_min_rmsd_data[0]) + ' rmsd: ' + str(first_min_rmsd_data[1].item()))
            write_xyz(os.path.join(run.dir,'second_min_rmsd.xyz'),second_min_rmsd_data[2],second_min_rmsd_data[3],comment='second_min_rmsd ' + str(second_min_rmsd_data[0]) + ' rmsd: ' + str(second_min_rmsd_data[1].item()))
            write_xyz(os.path.join(run.dir,'third_min_rmsd.xyz'),third_min_rmsd_data[2],third_min_rmsd_data[3],comment='third_min_rmsd ' + str(third_min_rmsd_data[0]) + ' rmsd: ' + str(third_min_rmsd_data[1].item()))
            write_xyz(os.path.join(run.dir,'mid_rmsd.xyz'),mid_rmsd_data[2],mid_rmsd_data[3],comment='mid_rmsd ' + str(mid_rmsd_data[0]) + ' rmsd: ' + str(mid_rmsd_data[1].item()))
            write_xyz(os.path.join(run.dir,'max_rmsd.xyz'),max_rmad_data[2],max_rmad_data[3],comment='max_rmsd ' + str(max_rmad_data[0]) + ' rmsd: ' + str(max_rmad_data[1].item()))
            wandb.config.update({'rmsd_xyz_path':run.dir})
            print('xyz file created')

    #noise_scheduleの記録
    if args.record_schedule:
        fig_alpha = noise_schedule_for_GammaNetwork(run.config.model_save_path,prms,'alpha')
        fig_sigma = noise_schedule_for_GammaNetwork(run.config.model_save_path,prms,'sigma')
        fig_SNR = noise_schedule_for_GammaNetwork(run.config.model_save_path,prms,'SNR')
        if prms['noise_schedule'] == 'learned':
            fig_gamma = noise_schedule_for_GammaNetwork(run.config.model_save_path,prms,'gamma')
            wandb.log({'alpha':wandb.Image(fig_alpha),'sigma':wandb.Image(fig_sigma),'gamma':wandb.Image(fig_gamma),'SNR':wandb.Image(fig_SNR)})
            plt.close(fig_gamma)
        elif prms['noise_schedule'] == 'predefined':
            wandb.log({'alpha':wandb.Image(fig_alpha),'sigma':wandb.Image(fig_sigma),'SNR':wandb.Image(fig_SNR)})
        print('noise_schedule saved')
        plt.close(fig_alpha)
        plt.close(fig_sigma)
        plt.close(fig_SNR)

    #wandbの終了
    wandb.finish()





    
