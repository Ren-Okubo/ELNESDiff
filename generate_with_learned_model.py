import os, torch, wandb, argparse, sys, random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from split_to_train_and_test import SetUpData
from diffusion_x_h import E3DiffusionProcess
from EquivariantGraphNeuralNetwork import EquivariantGNN, EquivariantGNNold

sys.path.append('parts/')
from def_for_main import load_model_state
from train_per_iretation import generate





if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--learned_model', type=str, required=True, help='project_name/run_id')
    argparser.add_argument('--project_name', type=str, required=True)
    argparser.add_argument('--run_name', type=str, required=True)
    argparser.add_argument('--dataset_path', type=str, default=None)
    argparser.add_argument('--scaling', required=True, choices=['max', 'area', None])
    args = argparser.parse_args()
    
    #learned_modelのconfigを取得
    api = wandb.Api()
    learned_model = api.run(args.learned_model)
    config = learned_model.config


    #seedの設定
    seed = config['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    #datasetの読み込み
    if args.dataset_path is None:
        dataset_path = config['dataset_path']
        dataset = torch.load(dataset_path)
        dataset = [data for data in dataset if data.x.shape[0] > 1]  # Remove graphs with only one atom
        setupdata = SetUpData(config['seed'])
        train_data, valid_data, test_data = setupdata.split(dataset)
    else:
        dataset_path = args.dataset_path
        test_data = torch.load(dataset_path)
        test_data = [data for data in test_data if data.x.shape[0] > 1]  # Remove graphs with only one atomJ

    #モデルの定義、状態のロード
    diffusion_process = E3DiffusionProcess(s=config['noise_precision'],
                                           power=config['noise_schedule_power'],
                                           num_diffusion_timestep=config['num_diffusion_timestep'],
                                           noise_schedule=config['noise_schedule'],
                                           deterministic=True)
    
    #egnnのパラメータ
    L = config['L']
    atom_type_size = config['atom_type_size']
    spectrum_size = config['spectrum_size']
    if config['Encoder_Decoder']['spectrum_to_latent']:
        spectrum_size = config['Encoder_Decoder']['latent_dim']
    d_size = config['d_size']
    t_size = config['t_size']
    excited_atom_size = config['excited_atom_size']
    config['excited_atom_size'] = excited_atom_size

    if config['conditional']:
        h_size = atom_type_size + spectrum_size + t_size
    else:
        h_size = atom_type_size + t_size
    if config['give_excited_atom']:
        h_size = h_size + excited_atom_size
    x_size = config['x_size']
    m_size = config['m_size']   
    m_input_size = h_size + h_size + d_size
    m_hidden_size = config['m_hidden_size']
    m_output_size = m_size
    h_input_size = h_size + m_size
    h_hidden_size = config['h_hidden_size']
    h_output_size = h_size 
    x_input_size = h_size + h_size + d_size
    x_hidden_size = config['x_hidden_size']
    x_output_size = 1
    
    #egnn = EquivariantGNN(L,m_input_size,m_hidden_size,m_output_size,x_input_size,x_hidden_size,x_output_size,h_input_size,h_hidden_size,h_output_size)
    egnn = EquivariantGNNold(L,m_input_size,m_hidden_size,m_output_size,x_input_size,x_hidden_size,x_output_size,h_input_size,h_hidden_size,h_output_size)


    nn_dict = {'egnn': egnn}
    try:
        load_model_state(nn_dict, config['model_save_path'], config)
    except:
        egnn = EquivariantGNN(L,m_input_size,m_hidden_size,m_output_size,x_input_size,x_hidden_size,x_output_size,h_input_size,h_hidden_size,h_output_size)
        nn_dict = {'egnn': egnn}
        load_model_state(nn_dict, config['model_save_path'], config)
    nn_dict['egnn_ema'] = nn_dict['egnn']
    egnn.eval()

    config['classifier_free'] = {
        'use_classifier_free': True,
    }

    
    #cfg weight list
    cfg_weight_list = [2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
    cfg_weight_list = [0.0]

    for cfg_weight in cfg_weight_list:
        config['classifier_free']['classifier_free_weight'] = cfg_weight
        run_name = f"{args.run_name}_{cfg_weight}"

        #initialize wandb
        run = wandb.init(project=args.project_name, name=run_name, config=config)

        # Datasetのスペクトルのscaling
        if args.scaling == 'max':
            for data in test_data:
                data.spectrum = data.spectrum_raw / data.spectrum_raw.max()
        elif args.scaling == 'area':
            for data in test_data:
                data.spectrum = data.spectrum_raw / data.spectrum_raw.sum()
        elif args.scaling is None:
            for data in test_data:
                data.spectrum = data.spectrum_raw
        
        # generate
        with torch.no_grad():
            original_graph_list, generated_graph_list = generate(nn_dict,test_data,config,diffusion_process,ema_model=nn_dict['egnn_ema'],gen_num_per_spectrum=5)

        # save the generated and original graphs
        generated_graph_save_path = os.path.join(wandb.run.dir,'generated_graph.pt')
        original_graph_save_path = os.path.join(wandb.run.dir,'original_graph.pt')
        torch.save(generated_graph_list,generated_graph_save_path)
        torch.save(original_graph_list,original_graph_save_path)
        wandb.config.update({'original_graph_save_path': original_graph_save_path},allow_val_change=True)
        print('The original graph has been saved.')
        wandb.config.update({'generated_graph_save_path': generated_graph_save_path},allow_val_change=True)
        print('The generated graph has been saved.')

        # evaluate by atom type prediction
        original_graph_list = torch.load(original_graph_save_path)
        generated_graph_list = torch.load(generated_graph_save_path)
        Si_tensor = torch.tensor([0,1],dtype=torch.long)
        O_tensor = torch.tensor([1,0],dtype=torch.long)
        density_O_original = []
        density_O_generated = []
        for i in range(len(original_graph_list)):
            num_O = 0
            original_graph = original_graph_list[i]
            generated_graph = generated_graph_list[i][-1]
            for j in range(original_graph.x.shape[0]):
                if torch.equal(original_graph.x[j],O_tensor):
                    num_O += 1
                elif torch.equal(original_graph.x[j],Si_tensor):
                    pass
                else:
                    print('Error')
            density_O_original.append(num_O/original_graph.x.shape[0])
            num_O = 0
            for j in range(generated_graph.x.shape[0]):
                if torch.equal(generated_graph.x[j],O_tensor):
                    num_O += 1
                elif torch.equal(generated_graph.x[j],Si_tensor):
                    pass
                else:
                    print('Error')
            density_O_generated.append(num_O/generated_graph.x.shape[0])
        fig, ax = plt.subplots()
        ax.plot([0,1],[0,1],linestyle='-',color='red')
        ax.plot(density_O_original,density_O_generated,linestyle='None',marker='o')
        ax.set_xlabel('density of O in original')
        ax.set_ylabel('density of O in generated')
        ax.set_title('density of O in original and generated')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        plt.text(0.05, 0.95, f'accuracy: {sum([1 for i in range(len(density_O_original)) if abs(density_O_original[i]-density_O_generated[i])==0])/len(density_O_original)}', 
                transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        wandb.log({'atom_type_eval':wandb.Image(fig)})
        plt.close()

        wandb.finish()    
    """

    params['classifier_free']

    run_name = f"{args.run_name}"

    #initialize wandb
    run = wandb.init(project=args.project_name, name=run_name, config=config)

    # Datasetのスペクトルのscaling
    if args.scaling == 'max':
        for data in test_data:
            data.spectrum = data.spectrum_raw / data.spectrum_raw.max()
    elif args.scaling == 'area':
        for data in test_data:
            data.spectrum = data.spectrum_raw / data.spectrum_raw.sum()
    elif args.scaling is None:
        for data in test_data:
            data.spectrum = data.spectrum_raw
    
    # generate
    with torch.no_grad():
        original_graph_list, generated_graph_list = generate(nn_dict,test_data,config,diffusion_process,ema_model=nn_dict['egnn_ema'],gen_num_per_spectrum=5)

    # save the generated and original graphs
    generated_graph_save_path = os.path.join(wandb.run.dir,'generated_graph.pt')
    original_graph_save_path = os.path.join(wandb.run.dir,'original_graph.pt')
    torch.save(generated_graph_list,generated_graph_save_path)
    torch.save(original_graph_list,original_graph_save_path)
    wandb.config.update({'original_graph_save_path': original_graph_save_path},allow_val_change=True)
    print('The original graph has been saved.')
    wandb.config.update({'generated_graph_save_path': generated_graph_save_path},allow_val_change=True)
    print('The generated graph has been saved.')

    # evaluate by atom type prediction
    original_graph_list = torch.load(original_graph_save_path)
    generated_graph_list = torch.load(generated_graph_save_path)
    Si_tensor = torch.tensor([0,1],dtype=torch.long)
    O_tensor = torch.tensor([1,0],dtype=torch.long)
    density_O_original = []
    density_O_generated = []
    for i in range(len(original_graph_list)):
        num_O = 0
        original_graph = original_graph_list[i]
        generated_graph = generated_graph_list[i][-1]
        for j in range(original_graph.x.shape[0]):
            if torch.equal(original_graph.x[j],O_tensor):
                num_O += 1
            elif torch.equal(original_graph.x[j],Si_tensor):
                pass
            else:
                print('Error')
        density_O_original.append(num_O/original_graph.x.shape[0])
        num_O = 0
        for j in range(generated_graph.x.shape[0]):
            if torch.equal(generated_graph.x[j],O_tensor):
                num_O += 1
            elif torch.equal(generated_graph.x[j],Si_tensor):
                pass
            else:
                print('Error')
        density_O_generated.append(num_O/generated_graph.x.shape[0])
    fig, ax = plt.subplots()
    ax.plot([0,1],[0,1],linestyle='-',color='red')
    ax.plot(density_O_original,density_O_generated,linestyle='None',marker='o')
    ax.set_xlabel('density of O in original')
    ax.set_ylabel('density of O in generated')
    ax.set_title('density of O in original and generated')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    plt.text(0.05, 0.95, f'accuracy: {sum([1 for i in range(len(density_O_original)) if abs(density_O_original[i]-density_O_generated[i])==0])/len(density_O_original)}', 
            transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    wandb.log({'atom_type_eval':wandb.Image(fig)})
    plt.close()

    wandb.finish() 

    """

    