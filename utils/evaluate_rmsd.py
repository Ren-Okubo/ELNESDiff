import os
import numpy as np
import torch
import itertools
import argparse
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm

def kabsch_numpy(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.

    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, and the RMSD.
    P, QはexOを原点とした座標系に移動させてから渡す
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Center the points
    p = P - P[0]
    q = Q - Q[0]

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

    return R, rmsd

def write_xyz(save_path,graph,comment=None):
    num_atom = graph.pos.shape[0]
    Si_tensor = torch.tensor([0,1]).to('cuda')
    O_tensor = torch.tensor([1,0]).to('cuda')
    with open(save_path,'w') as f:
        f.write(str(num_atom)+'\n')
        f.write(f'{comment}\n')
        for i in range(num_atom):
            if torch.equal(graph.x[i].to('cuda'),Si_tensor):
                atom_type = 'Si'
            elif torch.equal(graph.x[i].to('cuda'),O_tensor):
                atom_type = 'O'
            else:
                print('atom type error')
                exit()
            f.write(f'{atom_type} {graph.pos[i][0].item()} {graph.pos[i][1].item()} {graph.pos[i][2].item()}\n')

# xyzディレクトリ内にrun_idのディレクトリを作成し、その中にrmsd-sorted_indexの図,rmsdの値を保存した.ptファイル、xyzファイルを保存する

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--project_name', type=str, required=True)
    argparser.add_argument('--run_id', type=str, required=True)
    args = argparser.parse_args()

    os.makedirs('xyz/'+args.run_id,exist_ok=True)
    os.makedirs(f'xyz/{args.run_id}/rmsd',exist_ok=True)

    api = wandb.Api()
    run = api.run(path=f"{args.project_name}/{args.run_id}")
    config = run.config

    original_graph_list = torch.load(config['original_graph_save_path'])
    generated_graph_list = torch.load(config['generated_graph_save_path'])

    rmsd_list = []
    id_list = []
    id_dict = {}
    for i in tqdm(range(len(original_graph_list))):
        original_graph = original_graph_list[i]
        generated_graph = generated_graph_list[i][-1]
        if original_graph.pos.shape[0] >10:
            continue
        id = original_graph.id
        if not id in id_dict:
            id_dict[id] = 1
        else:
            id_dict[id] += 1
        id = f'{id}_{id_dict[id]}'
        original_graph_pos = original_graph.pos - original_graph.pos[0]
        generated_graph_pos = generated_graph.pos - generated_graph.pos[0]
        original_graph_pos = original_graph_pos.cpu().numpy()
        generated_graph_pos = generated_graph_pos.cpu().numpy()
        min_rmsd = 1e10
        perms = list(itertools.permutations(range(1,original_graph.pos.shape[0])))
        for perm in perms:
            order = [0]+list(perm)
            aligned_generated_pos = generated_graph_pos[order]
            R, rmsd = kabsch_numpy(aligned_generated_pos,original_graph_pos)
            if rmsd < min_rmsd:
                min_rmsd = rmsd
                min_pos = np.dot(aligned_generated_pos,R.T)
                min_order = order
                min_R = R
        generated_graph.pos = torch.from_numpy(min_pos)
        generated_graph.x = generated_graph.x[min_order]
        comment = f'{id} {min_rmsd}'
        id_list.append(id)
        rmsd_list.append(min_rmsd)
        os.makedirs(f'xyz/{args.run_id}/rmsd/{id}',exist_ok=True)
        write_xyz(f'xyz/{args.run_id}/rmsd/{id}/original.xyz',original_graph,comment=comment)
        write_xyz(f'xyz/{args.run_id}/rmsd/{id}/generated.xyz',generated_graph,comment=comment)
    id_rmsd_list = list(zip(id_list,rmsd_list))
    sorted_id_rmsd_list = sorted(id_rmsd_list,key=lambda x:x[1])
    sorted_id_list,sorted_rmsd_list = zip(*sorted_id_rmsd_list)
    sorted_rmsd_list = torch.tensor(sorted_rmsd_list).cpu().numpy()
    fig, ax = plt.subplots()
    ax.plot(sorted_rmsd_list,marker='o',linestyle='None')
    ax.set_xlabel('sorted_index')
    ax.set_ylabel('rmsd')
    ax.set_yscale('log')
    ax.set_title('rmsd')
    plt.savefig(f'xyz/{args.run_id}/rmsd/rmsd.png')
    plt.close()
    torch.save(sorted_id_rmsd_list,f'xyz/{args.run_id}/rmsd/sorted_id_rmsd.pt')
    print(f'first min rmsd : {sorted_id_rmsd_list[0]}')
    print(f'second min rmsd : {sorted_id_rmsd_list[1]}')
    print(f'third min rmsd : {sorted_id_rmsd_list[2]}')
    print(f'mid rmsd : {sorted_id_rmsd_list[len(sorted_id_rmsd_list)//2]}')
    print(f'worst min rmsd : {sorted_id_rmsd_list[-1]}')


 