import os
import numpy as np
import torch
import itertools
import argparse
import wandb
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment


def write_xyz_for_pos_generation(save_path,original_pos,generated_pos,comment=None):
    num_atom = original_pos.shape[0]
    
    with open(save_path,'w') as f:
        f.write(str(num_atom*2)+'\n')
        f.write(f'{comment}\n')
        for i in range(num_atom):
            f.write(f'F {original_pos[i][0].item()} {original_pos[i][1].item()} {original_pos[i][2].item()}\n')
        for i in range(num_atom):
            f.write(f'O {generated_pos[i][0].item()} {generated_pos[i][1].item()} {generated_pos[i][2].item()}\n')

def write_xyz_for_aligned_pos(save_path,position,comment=None):
    num_atom = position.shape[0]
    
    with open(save_path,'w') as f:
        f.write(str(num_atom)+'\n')
        f.write(f'{comment}\n')
        for i in range(num_atom):
            f.write(f'O {position[i][0].item()} {position[i][1].item()} {position[i][2].item()}\n')

def write_xyz(save_path,graph,comment=None):
    num_atom = graph.pos.shape[0]
    device = graph.x.device
    Si_tensor = torch.tensor([0,1], device=device)
    O_tensor = torch.tensor([1,0], device=device)
    with open(save_path,'w') as f:
        f.write(str(num_atom)+'\n')
        f.write(f'{comment}\n')
        for i in range(num_atom):
            if torch.equal(graph.x[i].to(device),Si_tensor):
                atom_type = 'Si'
            elif torch.equal(graph.x[i].to(device),O_tensor):
                atom_type = 'O'
            else:
                print('atom type error')
                exit()
            f.write(f'{atom_type} {graph.pos[i][0].item()} {graph.pos[i][1].item()} {graph.pos[i][2].item()}\n')
    

def kabsch_numpy(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.

    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Center the points
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Optimal translation
    t = centroid_Q - centroid_P

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = np.dot(p.T, q)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Validate right-handed coordinate system
    if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:
        Vt[-1, :] *= -1.0

    # Optimal rotation
    R = np.dot(Vt.T, U.T)

    # RMSD
    rmsd = np.sqrt(np.sum(np.square(np.dot(p, R.T) - q)) / P.shape[0])

    return R, rmsd, t

def hungarian_algorithm(P,Q):
    D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(D)
    return row_ind, col_ind

def return_near_from_exO(position):
    exO = position[0]
    length_list, index_list = [], []
    for i in range(1,len(position)):
        length_list.append(torch.norm(position[i]-exO).item())
        index_list.append(i)
    length_index_list = list(zip(length_list,index_list))
    sorted_legnth_index_list = sorted(length_index_list,key=lambda x:x[0])
    sorted_length_list, sorted_index_list = zip(*sorted_legnth_index_list)
    return sorted_index_list[:8]


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--project_name', type=str, required=True)
    args.add_argument('--run_id', type=str, required=True)
    args = args.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    api = wandb.Api()
    run = api.run(f'{args.project_name}/{args.run_id}')
    config = run.config
    generated_graph_save_path = config['generated_graph_save_path']
    original_graph_save_path = config['original_graph_save_path']

    os.makedirs(os.path.join('/jbod/rokubo/data/diffusion_model/xyz',args.run_id),exist_ok=True)
    
    if os.path.exists(generated_graph_save_path):
        original_graph_list = torch.load(original_graph_save_path, map_location=device, weights_only=False)
        generated_graph_list = torch.load(generated_graph_save_path, map_location=device, weights_only=False)
    else:
        original_graph_list = torch.load(original_graph_save_path.replace('/mnt',''), map_location=device, weights_only=False)
        generated_graph_list = torch.load(generated_graph_save_path.replace('/mnt',''), map_location=device, weights_only=False)
    rmsd_list = []
    id_list = []

    for i in tqdm(range(len(original_graph_list))):
        original_graph = copy.deepcopy(original_graph_list[i])
        generated_graph = copy.deepcopy(generated_graph_list[i][-1])

        id = f'{original_graph.id}_{i%5+1}'
        #id = f'{args.run_id}_{i}'
        #id = f'{i}'
        id_path = os.path.join('/jbod/rokubo/data/diffusion_model/xyz',args.run_id,id)
        """if os.path.exists(id_path):
            #continue
            pass"""
        os.makedirs(id_path,exist_ok=True)
        original_graph.id = id
        generated_graph.id = id

        min_rmsd = 1e+10
        if original_graph.pos.shape[0] < 9:
            check_original = original_graph.pos
            check_generated = torch.zeros_like(generated_graph.pos)
            check_generated[0] = generated_graph.pos[0]
            perms = list(itertools.permutations(range(1,original_graph.pos.shape[0])))
            for perm in perms:
                for j in range(len(perm)):
                    check_generated[j+1] = generated_graph.pos[perm[j]]
                R, rmsd, t = kabsch_numpy(check_generated.cpu().numpy(),check_original.cpu().numpy())
                if rmsd < min_rmsd:
                    min_rmsd = rmsd
                    min_R = R
                    min_perm = perm
                    min_t = t
            #generated_graph.pos = generated_graph.pos - generated_graph.pos[0]
            

            aligned_generated_pos = torch.zeros_like(generated_graph.pos)
            aligned_generated_pos[0] = generated_graph.pos[0]
            for j in range(len(min_perm)):
                aligned_generated_pos[j+1] = generated_graph.pos[min_perm[j]]
            aligned_generated_pos = np.dot(aligned_generated_pos.cpu().numpy(), min_R.T) + min_t
            generated_graph.pos = torch.from_numpy(aligned_generated_pos)
            aligned_generated_x = torch.zeros_like(generated_graph.x)
            aligned_generated_x[0] = generated_graph.x[0]
            for j in range(len(min_perm)):
                aligned_generated_x[j+1] = generated_graph.x[min_perm[j]]
            generated_graph.x = aligned_generated_x
            comment = f'{id} {min_rmsd}'
            original_save_path =os.path.join(id_path,'original.xyz')
            generated_save_path = os.path.join(id_path,'generated.xyz')
            write_xyz(original_save_path,original_graph,comment=comment)
            write_xyz(generated_save_path,generated_graph,comment=comment)
            rmsd_list.append(min_rmsd)
            id_list.append(id)
        else:
            generated_index_list = return_near_from_exO(generated_graph.pos)
            original_index_list = return_near_from_exO(original_graph.pos)
            original_near_exO = torch.zeros(8,3)
            generated_near_exO = torch.zeros(8,3)
            original_near_exO[0] = original_graph.pos[0]
            generated_near_exO[0] = generated_graph.pos[0]
            perms = list(itertools.permutations(range(7)))
            min_rmsd = 1e+10
            min_perm = None
            for j in range(7):
                original_near_exO[j+1] = original_graph.pos[original_index_list[j]]
            for perm in perms:
                for i in perm:
                    generated_near_exO[i+1] = generated_graph.pos[generated_index_list[perm[i]]]
                R, rmsd, t = kabsch_numpy(generated_near_exO.cpu().numpy(),original_near_exO.cpu().numpy())
                if rmsd < min_rmsd:
                    min_rmsd = rmsd
                    min_R = R
                    min_perm = perm
                    min_t = t
            generated_pos = generated_graph.pos.cpu().numpy()
            original_pos = original_graph.pos.cpu().numpy()
            #generated_pos = generated_pos - generated_pos[0]
            #original_pos = original_pos - original_pos[0]
            aligned_generated_pos = np.dot(generated_pos, min_R.T) + min_t
            row_ind, col_ind = hungarian_algorithm(original_pos,aligned_generated_pos)
            adjusted_generated_pos = aligned_generated_pos[col_ind]
            adjusted_original_pos = original_pos[row_ind]
            generated_graph.pos = torch.from_numpy(adjusted_generated_pos)      
            original_graph.pos = torch.from_numpy(adjusted_original_pos)     
            adjusted_generated_x = generated_graph.x[col_ind]
            adjusted_original_x = original_graph.x[row_ind]
            generated_graph.x = adjusted_generated_x
            original_graph.x = adjusted_original_x
            R, rmsd, t = kabsch_numpy(adjusted_generated_pos,adjusted_original_pos)
            adjusted_generated_pos = np.dot(adjusted_generated_pos,R.T) + t
            generated_graph.pos = torch.from_numpy(adjusted_generated_pos)
            comment = f'{id} {rmsd}'
            original_save_path =os.path.join(id_path,'original.xyz')
            generated_save_path = os.path.join(id_path,'generated.xyz')
            write_xyz(original_save_path,original_graph,comment=comment)
            write_xyz(generated_save_path,generated_graph,comment=comment)
            rmsd_list.append(rmsd)
            id_list.append(id)
    
    sorted_rmsd_list = sorted(rmsd_list)
    fig, ax = plt.subplots()
    ax.plot(sorted_rmsd_list,marker='o',linestyle='None',markersize=2)
    ax.set_xlabel('sorted_index')
    ax.set_ylabel('rmsd')
    ax.set_yscale('log')
    ax.set_title('rmsd')
    plt.savefig(f'/jbod/rokubo/data/diffusion_model/xyz/{args.run_id}/rmsd.png')
    plt.close()

    id_rmsd_list = list(zip(id_list,rmsd_list))
    sorted_id_rmsd_list = sorted(id_rmsd_list,key=lambda x:x[1])
    torch.save(sorted_id_rmsd_list,os.path.join('/jbod/rokubo/data/diffusion_model/xyz',args.run_id,'sorted_id_rmsd.pt'))
