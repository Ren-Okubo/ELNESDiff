import wandb
import os, pdb
import torch
import argparse
import numpy as np
from evaluate_rmsd_for_pos_generate import kabsch_torch, kabsch_numpy

def write_xyz_for_abinitio(run_id,generated_abinitio_data_path):
    os.makedirs(os.path.join('/mnt/homenfsxx/rokubo/data/diffusion_model/xyz',run_id),exist_ok=True)
    datalist = torch.load(generated_abinitio_data_path)
    for i in range(len(datalist)):
        data = datalist[i][-1].to('cuda')
        with open(os.path.join('/mnt/homenfsxx/rokubo/data/diffusion_model/xyz',run_id,f'{i}.xyz'),'w') as f:
            f.write(str(data.pos.shape[0])+'\n')
            f.write('\n')
            for j in range(data.pos.shape[0]):
                if torch.equal(data.x[j],torch.tensor([0,1]).to('cuda')):
                    atom_type = 'Si'
                elif torch.equal(data.x[j],torch.tensor([1,0]).to('cuda')):
                    atom_type = 'O'
                f.write(f'{atom_type} {data.pos[j][0].item()} {data.pos[j][1].item()} {data.pos[j][2].item()}\n')

def write_xyz(save_path,original_graph,generated_graph,comment=None):
    original_graph = original_graph.to('cuda')
    generated_graph = generated_graph.to('cuda')
    num_atom = original_graph.pos.shape[0]
    Si_tensor = torch.tensor([0,1]).to('cuda')
    O_tensor = torch.tensor([1,0]).to('cuda')
    with open(save_path,'w') as f:
        f.write(str(num_atom*2)+'\n')
        f.write(f'{comment}\n')
        for i in range(num_atom):
            if torch.equal(original_graph.x[i],Si_tensor):
                atom_type = 'Al'
            elif torch.equal(original_graph.x[i],O_tensor):
                atom_type = 'F'
            else:
                print('error')
                exit()
            f.write(f'{atom_type} {original_graph.pos[i][0].item()} {original_graph.pos[i][1].item()} {original_graph.pos[i][2].item()}\n')
        for i in range(num_atom):
            if torch.equal(generated_graph.x[i],Si_tensor):
                atom_type = 'Si'
            elif torch.equal(generated_graph.x[i],O_tensor):
                atom_type = 'O'
            f.write(f'{atom_type} {generated_graph.pos[i][0].item()} {generated_graph.pos[i][1].item()} {generated_graph.pos[i][2].item()}\n')

def write_xyz_from_only_pos(save_path,original_pos,generated_pos,comment=None):
    num_atom = original_pos.shape[0]
    
    with open(save_path,'w') as f:
        f.write(str(num_atom*2)+'\n')
        f.write(f'{comment}\n')
        f.write(f'F {original_pos[0][0].item()} {original_pos[0][1].item()} {original_pos[0][2].item()}\n')
        for i in range(1,num_atom):
            f.write(f'Al {original_pos[i][0].item()} {original_pos[i][1].item()} {original_pos[i][2].item()}\n')
        f.write(f'O {generated_pos[0][0].item()} {generated_pos[0][1].item()} {generated_pos[0][2].item()}\n')
        for i in range(1,num_atom):
            f.write(f'Si {generated_pos[i][0].item()} {generated_pos[i][1].item()} {generated_pos[i][2].item()}\n')

def make_xyz_ignore_atom_type(save_path,original_graph,generated_graph,comment=None):
    num_atom = original_graph.pos.shape[0]
    with open(save_path,'w') as f:
        f.write(str(num_atom*2)+'\n')
        f.write(f'{comment}\n')
        for i in range(num_atom):
            f.write(f'F {original_graph.pos[i][0].item()} {original_graph.pos[i][1].item()} {original_graph.pos[i][2].item()}\n')
        for i in range(num_atom):
            f.write(f'O {generated_graph.pos[i][0].item()} {generated_graph.pos[i][1].item()} {generated_graph.pos[i][2].item()}\n')


def align_pos(original_pos,generated_pos):
    R, t, _ = kabsch_numpy(generated_pos.cpu().numpy(),original_pos.cpu().numpy())
    aligned_pos = np.dot(generated_pos.cpu().numpy() + t,R.T)
    return aligned_pos

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, required=True)
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--all', type=bool, default=False)
    args = parser.parse_args()

    run = wandb.init(project=args.project_name, id=args.run_id, resume='must')
    run_dir = run.dir
    run_id = run.id
    rmsd_path = run.config.rmsd_save_path
    data = torch.load(rmsd_path)

    if args.all:
        os.makedirs(os.path.join('/mnt/homenfsxx/rokubo/data/diffusion_model/xyz',run_id),exist_ok=True)
        save_path = os.path.join('/mnt/homenfsxx/rokubo/data/diffusion_model/xyz',run_id)
        for i in range(len(data)):
            data[i][3].pos = align_pos(data[i][2].pos,data[i][3].pos)
            


    #xyzファイルにする生成結果を指定
    first_min_rmsd_data = data[0]
    second_min_rmsd_data = data[1]
    third_min_rmsd_data = data[2]
    mid_rmsd_data = data[int(len(data)/2)]
    max_rmad_data = data[-1]


    first_min_rmsd_data[3].pos = align_pos(first_min_rmsd_data[2].pos,first_min_rmsd_data[3].pos)
    second_min_rmsd_data[3].pos = align_pos(second_min_rmsd_data[2].pos,second_min_rmsd_data[3].pos)
    third_min_rmsd_data[3].pos = align_pos(third_min_rmsd_data[2].pos,third_min_rmsd_data[3].pos)
    mid_rmsd_data[3].pos = align_pos(mid_rmsd_data[2].pos,mid_rmsd_data[3].pos)
    max_rmad_data[3].pos = align_pos(max_rmad_data[2].pos,max_rmad_data[3].pos)

    write_xyz(os.path.join(run_dir,'first_min_rmsd.xyz'),first_min_rmsd_data[2],first_min_rmsd_data[3],comment='first_min_rmsd ' + str(first_min_rmsd_data[0]) + ' rmsd : ' + str(first_min_rmsd_data[1]))
    write_xyz(os.path.join(run_dir,'second_min_rmsd.xyz'),second_min_rmsd_data[2],second_min_rmsd_data[3],comment='second_min_rmsd ' + str(second_min_rmsd_data[0]) + ' rmsd : ' + str(second_min_rmsd_data[1]))
    write_xyz(os.path.join(run_dir,'third_min_rmsd.xyz'),third_min_rmsd_data[2],third_min_rmsd_data[3],comment='third_min_rmsd ' + str(third_min_rmsd_data[0]) + ' rmsd : ' + str(third_min_rmsd_data[1]))
    write_xyz(os.path.join(run_dir,'mid_rmsd.xyz'),mid_rmsd_data[2],mid_rmsd_data[3],comment='mid_rmsd ' + str(mid_rmsd_data[0]) + ' rmsd : ' + str(mid_rmsd_data[1]))
    write_xyz(os.path.join(run_dir,'max_rmsd.xyz'),max_rmad_data[2],max_rmad_data[3],comment='max_rmsd ' + str(max_rmad_data[0]) + ' rmsd : ' + str(max_rmad_data[1]))
    run.config.update({'rmsd_xyz_path':run_dir})
    run.finish()


