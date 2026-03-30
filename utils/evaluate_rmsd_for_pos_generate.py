import torch
import argparse
import wandb
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np



def kabsch_torch(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.
    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = torch.mean(P, dim=0)
    centroid_Q = torch.mean(Q, dim=0)

    # Optimal translation
    t = centroid_Q - centroid_P

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(0, 1), q)

    # SVD
    U, S, Vt = torch.linalg.svd(H)

    # Validate right-handed coordinate system
    if torch.det(torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))) < 0.0:
        Vt = Vt.clone()
        #Vt[-1, :] *= -1.0
        Vt[:, -1] *= -1.0

    # Optimal rotation
    R = torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))

    # RMSD
    rmsd = torch.sqrt(torch.sum(torch.square(torch.matmul(p, R.transpose(0, 1)) - q)) / P.shape[0])

    return R, t, rmsd

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

    # Compute centroids
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

    return R, t, rmsd


def evaluate_by_rmsd(original_graph_list,generated_graph_list):
    id_list = []
    rmsd_value_list = []
    original_coords_list, generated_coords_list = [],[]
    for i in range(len(original_graph_list)):
        original_graph = original_graph_list[i]
        generated_graph = generated_graph_list[i][-1]
        if original_graph.pos.shape[0] == 1:
            continue
        _,_,rmsd_value = kabsch_numpy(original_graph.pos.cpu().numpy(),generated_graph.pos.cpu().numpy())
        rmsd_value_list.append(rmsd_value)
        id_list.append(original_graph.id)
        original_coords_list.append(original_graph)
        generated_coords_list.append(generated_graph)
    id_rmsd_original_generated_list = list(zip(id_list,rmsd_value_list,original_coords_list,generated_coords_list)) #rmsdの値でソート
    sorted_id_rmsd_original_generated_list = sorted(id_rmsd_original_generated_list,key=lambda x:x[1])
    return sorted_id_rmsd_original_generated_list

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--project_name', type=str, required=True)
    argparser.add_argument('--run_id', type=str, required=True)
    args = argparser.parse_args()

    run = wandb.init(project=args.project_name, id=args.run_id, resume='must')
    config = run.config
    generated_graph_save_path = config.generated_graph_save_path
    original_graph_save_path = config.original_graph_save_path
    generated_graph_list = torch.load(generated_graph_save_path)
    original_graph_list = torch.load(original_graph_save_path)
    sorted_id_rmsd_original_generated_list = evaluate_by_rmsd(original_graph_list,generated_graph_list)
    sorted_id_list,sorted_rmsd_list,sorted_original_coords_list,sorted_generated_coords_list = zip(*sorted_id_rmsd_original_generated_list)
    sorted_rmsd_list = torch.tensor(sorted_rmsd_list).cpu().numpy()
    rmsd_save_path = os.path.join(wandb.run.dir,'rmsd.pt')
    torch.save(sorted_id_rmsd_original_generated_list,rmsd_save_path)
    wandb.config.update({'rmsd_save_path':rmsd_save_path})
    print(f'rmsd saved at {rmsd_save_path}')
    fig, ax = plt.subplots()
    ax.plot(sorted_rmsd_list,marker='o',linestyle='None')
    ax.set_xlabel('sorted_index')
    ax.set_ylabel('rmsd')
    ax.set_yscale('log')
    ax.set_title('rmsd')
    #plt.savefig(os.path.join(run.dir,'rmsd.png'))
    wandb.log({'rmsd':wandb.Image(fig)})
    plt.close()
    run.finish()