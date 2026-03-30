import torch, os, pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import scipy.stats as stats
from split_to_train_and_test import SetUpData
import wandb
import argparse
from PIL import Image
from sklearn.metrics import r2_score

def calculate_angle_for_CN2(coords_tensor):
    v1 = coords_tensor[1] - coords_tensor[0]
    v2 = coords_tensor[2] - coords_tensor[0]
    cos = torch.dot(v1,v2) / (torch.norm(v1) * torch.norm(v2))
    return np.degrees(torch.acos(cos).item())

def calculate_bond_length_for_CN2(coords_tensor):
    v1 = coords_tensor[1] - coords_tensor[0]
    v2 = coords_tensor[2] - coords_tensor[0]
    return torch.norm(v1).item(),torch.norm(v2).item()

def r2score(a,b):
    arr_x = np.array(a)
    arr_y = np.array(b)
    n = len(arr_x)
    mean_x = sum(arr_x)/n
    mean_y = sum(arr_y)/n
    t_xx = sum((arr_x-mean_x)**2)
    t_yy = sum((arr_y-mean_y)**2)
    t_xy = sum((arr_x-mean_x)*(arr_y-mean_y))
    slope = t_xy/t_xx
    intercept = (1/n)*sum(arr_y)-(1/n)*slope*sum(arr_x)
    predict_x=intercept+slope*arr_x
    resudial_y=arr_y-predict_x
    r2 = 1-(sum(resudial_y**2))/t_yy
    return r2

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--project_name', type=str, required=True)
    argparser.add_argument('--run_id', type=str, required=True)
    argparser.add_argument('-e','--errorbar',type=bool,default=False)
    args = argparser.parse_args()

    run = wandb.init(project=args.project_name, id=args.run_id, resume='must')
    params = run.config
    
    gs = GridSpec(2,2,height_ratios=[1,4],width_ratios=[4,1])
    fig = plt.figure(figsize=(10,10))
    ax_scatter = fig.add_subplot(gs[1,0])
    #model_name = 'egnn_202411281617'
    #model_name = str(input('model_name:'))
    model_name = 'egnn_' + params['now']



    """
    project_name = str(input('project_name:'))
    run_id = str(input('run_id:'))
    api = wandb.Api()
    run = api.run(f'{project_name}/{run_id}')
    config = run.config
    generated_graph_save_path = config['generated_graph_save_path']
    original_graph_save_path = config['original_graph_save_path']
    generated_graph = torch.load(generated_graph_save_path)
    original_graph = torch.load(original_graph_save_path)

    theta_list, phi_list = [],[]
    for i in range(len(original_graph)):
        theta_list.append(calculate_angle_for_CN2(original_graph[i].pos))
        phi_list.append(calculate_angle_for_CN2(generated_graph[i][-1].pos))
    """
    """
    data = np.load('/mnt/homenfsxx/rokubo/data/diffusion_model/conditional_gen_by_dataset_only_CN2_including_180_' + model_name +'.npz')
    original_coords = data['original_coords_list']
    generated_coords = data['generated_coords_list']
    theta_list, phi_list = [],[]
    for i in range(len(original_coords)):
        theta_list.append(calculate_angle_for_CN2(torch.tensor(original_coords[i])))
        phi_list.append(calculate_angle_for_CN2(torch.tensor(generated_coords[i][-1])))
    """

    #original_data = torch.load("/mnt/homenfsxx/rokubo/data/diffusion_model/wandb/run-20250110_105421-y7oeqxx0/files/original_graph.pt")
    #generated_data = torch.load("/mnt/homenfsxx/rokubo/data/diffusion_model/wandb/run-20250110_105421-y7oeqxx0/files/generated_graph.pt")
    if os.path.exists(params['original_graph_save_path']):
        original_data = torch.load(params['original_graph_save_path'])
        generated_data = torch.load(params['generated_graph_save_path'])
    else:
        original_data = torch.load(params['original_graph_save_path'].replace('/mnt',''))
        generated_data = torch.load(params['generated_graph_save_path'].replace('/mnt',''))
    original_coords, generated_coords = [],[]
    for data in original_data:
        original_coords.append(data.pos)
    for data in generated_data:
        generated_coords.append(data[-1].pos)

    """
    #第一近接がCN2の3ang_datasetのときの処理
    original_CN2, generated_CN2 = [],[]
    for i in range(0,len(original_coords),5):
        check_per_graph_original, check_per_graph_generated = [],[]
        for coord in original_coords[i:i+5]:
            if coord.shape[0] < 3:
                continue
            length_check_original = []
            index_original = []
            for j in range(coord.shape[0]):
                length_check_original.append(torch.norm(coord[j]-coord[0]).item())
                index_original.append(j)
            sorted_index_length_original = sorted(list(zip(index_original,length_check_original)),key=lambda x:x[1])
            sorted_index_original, sorted_length_original = zip(*sorted_index_length_original)
            if sorted_length_original[1] > 1.8 or sorted_length_original[2] > 1.8:
                continue
            tensor_original = torch.zeros((3,3))
            tensor_original[0] = coord[sorted_index_original[0]]
            tensor_original[1] = coord[sorted_index_original[1]]
            tensor_original[2] = coord[sorted_index_original[2]]
            check_per_graph_original.append(tensor_original)
        for coord in generated_coords[i:i+5]:
            if coord.shape[0] < 3:
                continue
            length_check_generated = []
            index_generated = []
            for j in range(coord.shape[0]):
                length_check_generated.append(torch.norm(coord[j]-coord[0]).item())
                index_generated.append(j)
            sorted_index_length_generated = sorted(list(zip(index_generated,length_check_generated)),key=lambda x:x[1])
            sorted_index_generated, sorted_length_generated = zip(*sorted_index_length_generated)
            if sorted_length_generated[1] > 1.8 or sorted_length_generated[2] > 1.8:
                continue
            tensor_generated = torch.zeros((3,3))
            tensor_generated[0] = coord[sorted_index_generated[0]]
            tensor_generated[1] = coord[sorted_index_generated[1]]
            tensor_generated[2] = coord[sorted_index_generated[2]]
            check_per_graph_generated.append(tensor_generated)
        if len(check_per_graph_original) == 5 and len(check_per_graph_generated) == 5:
            original_CN2 += check_per_graph_original
            generated_CN2 += check_per_graph_generated
    original_coords = original_CN2
    generated_coords = generated_CN2
    """
    """
    original_only_CN2, generated_only_CN2 = [],[]
    for i in range(len(original_coords)):
        norm_list = []
        for j in range(original_coords[i].shape[0]):
            norm_list.append(torch.norm(original_coords[i][j]-original_coords[i][0]).item())
        norm_list = np.array(norm_list)
        if len(norm_list) < 3:
            continue
        small_index = np.argsort(norm_list)
        tensor = torch.zeros((3,3))
        tensor[0] = original_coords[i][small_index[0]]
        tensor[1] = original_coords[i][small_index[1]]
        tensor[2] = original_coords[i][small_index[2]]
        original_only_CN2.append(tensor)
    for i in range(len(generated_coords)):
        norm_list = []
        for j in range(generated_coords[i].shape[0]):
            norm_list.append(torch.norm(generated_coords[i][j]-generated_coords[i][0]).item())
        norm_list = np.array(norm_list)
        if len(norm_list) < 3:
            continue
        small_index = np.argsort(norm_list)
        tensor = torch.zeros((3,3))
        tensor[0] = generated_coords[i][small_index[0]]
        tensor[1] = generated_coords[i][small_index[1]]
        tensor[2] = generated_coords[i][small_index[2]]
        generated_only_CN2.append(tensor)

    original_coords = original_only_CN2
    generated_coords = generated_only_CN2
    """


    theta_list, phi_list = [],[]
    for i in range(len(original_coords)):
        theta_list.append(calculate_angle_for_CN2(original_coords[i]))
        phi_list.append(calculate_angle_for_CN2(generated_coords[i]))

    
    #theta_list = data['theta_list']
    #phi_list = data['phi_list']
    average_theta_per_graph =[]
    average_phi_per_graph =[]
    std_phi_per_graph = []

    how_many_gen = 5

    for i in list(range(0,len(theta_list),how_many_gen)):
        if np.isnan(np.mean(theta_list[i:i+how_many_gen])) or np.isnan(np.mean(phi_list[i:i+how_many_gen])):
            print("Error: Data contains NaN values")
        else:
            average_theta_per_graph.append(np.mean(theta_list[i:i+how_many_gen]))
            average_phi_per_graph.append(np.mean(phi_list[i:i+how_many_gen]))
            std_phi_per_graph.append(np.std(phi_list[i:i+how_many_gen]))
    


    hist_theta, bins_theta = np.histogram(average_theta_per_graph,bins=50,range=(70,180))
    hist_phi, bins_phi = np.histogram(average_phi_per_graph,bins=50,range=(70,180))
    #hist_theta, bins_theta = np.histogram(theta_list,bins=50,range=(70,180))
    #hist_phi, bins_phi = np.histogram(phi_list,bins=50,range=(70,180))
    bin_centers_theta = (bins_theta[:-1] + bins_theta[1:]) / 2
    bin_centers_phi = (bins_phi[:-1] + bins_phi[1:]) / 2

    #distance = stats.wasserstein_distance(bin_centers_theta,bin_centers_phi,u_weights=hist_theta,v_weights=hist_phi)
    #print('distance:',distance)
    
    
    list_theta = []
    list_phi = []
    for i in range(len(average_theta_per_graph)):
        if np.isnan(average_theta_per_graph[i]) or np.isnan(average_phi_per_graph[i]):
            print("Error: Data contains NaN values")
        else:
            list_theta.append(average_theta_per_graph[i])
            list_phi.append(average_phi_per_graph[i])
    
    average_theta_per_graph = np.array(list_theta)
    average_phi_per_graph = np.array(list_phi)
    if np.any(np.isnan(average_theta_per_graph)) or np.any(np.isnan(average_phi_per_graph)):
        print("Error: Data contains NaN values")

    
    ax_scatter.plot([0,180],[0,180],zorder=3,alpha=0.7)

    if args.errorbar:
        ax_scatter.errorbar(average_theta_per_graph,average_phi_per_graph,yerr=std_phi_per_graph,fmt='none',ecolor='red',capsize=3,capthick=1,alpha=0.5)
    else:
        ax_scatter.plot(theta_list,phi_list,'o',markersize=1.5)
    ax_scatter.plot(average_theta_per_graph,average_phi_per_graph,'o',markersize=3,color='blue',label='average per graph')
    #ax_scatter.errorbar(average_theta_per_graph,average_phi_per_graph,yerr=std_phi_per_graph,fmt='none',ecolor='red',capsize=3,capthick=1,alpha=0.5)
    ax_scatter.set_xlabel('theta')
    ax_scatter.set_ylabel('phi')
    ax_scatter.set_xlim(60,180)
    ax_scatter.set_ylim(60,180)
    ax_scatter.legend()

    ax_text = fig.add_subplot(gs[0,1])
    """
    num_of_test_data = len(average_theta_per_graph)
    num_within_range = 0
    for i in range(num_of_test_data):
        if average_theta_per_graph[i] >= average_phi_per_graph[i]:
            if average_theta_per_graph[i] < average_phi_per_graph[i]+std_phi_per_graph[i]:
                num_within_range += 1
        else:
            if average_theta_per_graph[i] > average_phi_per_graph[i]-std_phi_per_graph[i]:
                num_within_range += 1
    """

    r2 = r2score(average_theta_per_graph,average_phi_per_graph)
    theta_r2, phi_r2 = [],[]
    for i in range(len(theta_list)):
        if np.isnan(theta_list[i]) or np.isnan(phi_list[i]):
            pass
        else:
            theta_r2.append(theta_list[i])
            phi_r2.append(phi_list[i])
    r2_for_all = r2_score(theta_r2,phi_r2)
    ax_text.text(0.5, 0.6, 'r2_score:\n{:.2f}'.format(r2), fontsize=12, ha='center', va='center')
    ax_text.text(0.5, 0.4, 'r2_for_all:\n{:.2f}'.format(r2_for_all), fontsize=12, ha='center', va='center')
    #ax_text.text(0.5,0.5,'r2_score:\n{:.2f}'.format(r2),fontsize=12,ha='center',va='center')
    ax_text.axis('off')


    ax_hist_theta = fig.add_subplot(gs[0,0],sharex=ax_scatter)
    ax_hist_phi = fig.add_subplot(gs[1,1],sharey=ax_scatter)
    ax_hist_theta.hist(average_theta_per_graph,bins=50,orientation='vertical')
    ax_hist_phi.hist(average_phi_per_graph,bins=50,orientation='horizontal')
    #ax_hist_theta.hist(theta_list,bins=50,range=(70,180),orientation='vertical')
    #ax_hist_phi.hist(phi_list,bins=50,range=(70,180),orientation='horizontal')
    ax_hist_theta.get_xaxis().set_visible(False)
    ax_hist_phi.get_yaxis().set_visible(False)
    plt.subplots_adjust(wspace=0.05,hspace=0.05)
    if args.errorbar:
        plt.savefig(os.path.join(run.dir,"angle_CN2_eval_errorbar.png"))
        image = Image.open(os.path.join(run.dir,"angle_CN2_eval_errorbar.png"))
        run.log({'angle errorbar': wandb.Image(image)})
    else:
        plt.savefig(os.path.join(run.dir,"angle_CN2_eval.png"))
        image = Image.open(os.path.join(run.dir,"angle_CN2_eval.png"))
        run.log({'angle': wandb.Image(image)})
    plt.close()
    
    
    gs = GridSpec(2,2,height_ratios=[1,4],width_ratios=[4,1])
    fig = plt.figure(figsize=(10,10))
    ax_scatter = fig.add_subplot(gs[1,0])
    
    """
    #data = np.load('/mnt/homenfsxx/rokubo/data/diffusion_model/conditional_gen_by_dataset_only_CN2_including_180_egnn_202410291612.npz')
    original_coords = data['original_coords_list']
    generated_coords = data['generated_coords_list']
    theta_length, phi_length = [],[]
    for i in range(len(original_coords)):
        norm1, norm2 = calculate_bond_length_for_CN2(torch.tensor(original_coords[i]))
        theta_length.append((norm1+norm2)/2)
        norm1, norm2 = calculate_bond_length_for_CN2(torch.tensor(generated_coords[i][-1]))
        phi_length.append((norm1+norm2)/2)
    
    theta_length, phi_length = [],[]
    for i in range(len(original_graph)):
        norm1, norm2 = calculate_bond_length_for_CN2(original_graph[i].pos)
        theta_length.append((norm1+norm2)/2)
        norm1, norm2 = calculate_bond_length_for_CN2(generated_graph[i][-1].pos)
        phi_length.append((norm1+norm2)/2)
    """
    theta_length, phi_length = [],[]
    for i in range(len(original_coords)):
        norm1, norm2 = calculate_bond_length_for_CN2(original_coords[i])
        theta_length.append((norm1+norm2)/2)
        norm1, norm2 = calculate_bond_length_for_CN2(generated_coords[i])
        phi_length.append((norm1+norm2)/2)

    std_phi_per_graph = []
    average_length_of_original, average_length_of_generated = [],[]
    for i in list(range(0,len(theta_list),how_many_gen)):
        average_length_of_original.append(np.mean(theta_length[i:i+how_many_gen]))
        average_length_of_generated.append(np.mean(phi_length[i:i+how_many_gen]))
        std_phi_per_graph.append(np.std(phi_length[i:i+how_many_gen]))

    hist_theta, bins_theta = np.histogram(average_length_of_original,bins=50)
    hist_phi, bins_phi = np.histogram(average_length_of_generated,bins=50)
    bin_centers_theta = (bins_theta[:-1] + bins_theta[1:]) / 2
    bin_centers_phi = (bins_phi[:-1] + bins_phi[1:]) / 2

    distance = stats.wasserstein_distance(bin_centers_theta,bin_centers_phi,u_weights=hist_theta,v_weights=hist_phi)
    
    ax_scatter.plot([0,5],[0,5],zorder=3,alpha=0.7)
    if args.errorbar:
        ax_scatter.errorbar(average_length_of_original,average_length_of_generated,yerr=std_phi_per_graph,fmt='none',ecolor='red',capsize=3,capthick=1,alpha=0.5)
    else:
        ax_scatter.plot(theta_length,phi_length,'o',markersize=1.5)
    ax_scatter.plot(average_length_of_original,average_length_of_generated,'o',markersize=3,label='average per graph',color='blue')
    #ax_scatter.errorbar(average_length_of_original,average_length_of_generated,yerr=std_phi_per_graph,fmt='none',ecolor='red',capsize=3,capthick=1,alpha=0.5)
    ax_scatter.set_xlabel('mean of 2 lengths of original')
    ax_scatter.set_ylabel('mean of 2 lengths of generated')
    ax_scatter.set_xlim(1.2,2.2)
    ax_scatter.set_ylim(1.2,2.2)
    ax_scatter.legend()

    ax_text = fig.add_subplot(gs[0,1])
    r2 = r2score(average_length_of_original,average_length_of_generated)
    theta_r2, phi_r2 = [],[]
    for i in range(len(theta_length)):
        if np.isnan(theta_length[i]) or np.isnan(phi_length[i]):
            pass
        else:
            theta_r2.append(theta_length[i])
            phi_r2.append(phi_length[i])
    r2_for_all = r2_score(theta_length,phi_length)
    ax_text.text(0.5, 0.6, 'r2_score:\n{:.2f}'.format(r2), fontsize=12, ha='center', va='center')
    ax_text.text(0.5, 0.4, 'r2_for_all:\n{:.2f}'.format(r2_for_all), fontsize=12, ha='center', va='center')
    #ax_text.text(0.5,0.5,'r2_score:\n{:.2f}'.format(r2),fontsize=12,ha='center',va='center')
    ax_text.axis('off')


    ax_hist_theta = fig.add_subplot(gs[0,0],sharex=ax_scatter)
    ax_hist_phi = fig.add_subplot(gs[1,1],sharey=ax_scatter)
    #ax_text = fig.add_subplot(gs[0,1])
    #ax_text.text(0.5,0.5,'wasserstein_distance:\n{:.2f}'.format(distance),fontsize=12,ha='center',va='center')
    #ax_text.axis('off')
    ax_hist_theta.hist(average_length_of_original,bins=50,orientation='vertical')
    ax_hist_phi.hist(average_length_of_generated,bins=50,orientation='horizontal')
    ax_hist_theta.get_xaxis().set_visible(False)
    ax_hist_phi.get_yaxis().set_visible(False)
    plt.subplots_adjust(wspace=0.05,hspace=0.05)
    if args.errorbar:
        plt.savefig(os.path.join(run.dir,"length_CN2_eval_errorbar.png"))
        image = Image.open(os.path.join(run.dir,"length_CN2_eval_errorbar.png"))
        run.log({'bond length errorbar': wandb.Image(image)})
    else:
        plt.savefig(os.path.join(run.dir,"length_CN2_eval.png"))
        image = Image.open(os.path.join(run.dir,"length_CN2_eval.png"))
        run.log({'bond length': wandb.Image(image)})
    plt.close()
    run.finish()
    
    """
    data = np.load('generated_graph_on_1000_different_seeds_conditioned_by_angle_149_mp_557004_17_dataset_except_180.npz')
    generated_coords = data['generated_coords_list']
    original_coords = data['original_coords_list']
    angle_for_gen = []
    target_angle = calculate_angle_for_CN2(torch.tensor(original_coords[0]))
    for gen_coord in generated_coords:
        angle_for_gen.append(calculate_angle_for_CN2(torch.tensor(gen_coord)))
    fif, ax = plt.subplots()
    ax.hist(angle_for_gen,bins=75)
    ax.axvline(target_angle,color='red',label='target angle = {}'.format(target_angle))
    ax.set_xlabel('angle')
    ax.set_ylabel('frequency')
    ax.legend()
    plt.savefig('angle_distribution_of_generated_graph_on_1000_different_seeds_conditioned_by_angle_149_mp_557004_17_dataset_except_180.png')
    plt.close()
    
    
    #data_for_original_distribution = np.load('./comparison_between_original_and_gen/about_angle/angle_comparison_between_original_and_generated_except_180.npz')
    data = np.load("/mnt/homenfsxx/rokubo/data/diffusion_model/dataset/dataset.npy",allow_pickle=True)
    dataset = SetUpData().npy_to_graph(data)
    theta_list = []
    #for i in range(len(dataset)):
        #if dataset[i].num_nodes == 3:
            #theta_list.append(calculate_angle_for_CN2(dataset[i].pos))

    for i in range(len(dataset)):
        if dataset[i].pos.shape[0] == 3:
            distance1, distance2 = calculate_bond_length_for_CN2(dataset[i].pos)
            theta_list.append(0.5*(distance1+distance2))
    
    #data_for_original_distribution = np.load('./comparison_between_original_and_gen/about_angle/angle_comparison_between_original_and_generated.npz')
    #theta_list = data_for_original_distribution['theta_list']
    data_of_abinitio = np.load('/mnt/homenfsxx/rokubo/data/diffusion_model/result_or_unknown/abinitio_gen_by_dataset_only_CN2_including_180_10161249.npz')
    abinitio_coords = data_of_abinitio['generated_coords_list']
    #angle_for_abinitio = []
    #for abinitio_coord in abinitio_coords:
        #angle_for_abinitio.append(calculate_angle_for_CN2(torch.tensor(abinitio_coord)))
    distance_for_abinitio = []
    for abinitio_coord in abinitio_coords:
        distance1, distance2 = calculate_bond_length_for_CN2(torch.tensor(abinitio_coord))
        distance_for_abinitio.append(0.5*(distance1+distance2))

    #hist_original, bins_original = np.histogram(theta_list,bins=50,range=(70,180))
    #hist_abinitio, bins_abinitio = np.histogram(angle_for_abinitio,bins=50,range=(70,180))

    hist_original, bins_original = np.histogram(theta_list,bins=50,range=(0.5,2.0))
    hist_abinitio, bins_abinitio = np.histogram(distance_for_abinitio,bins=50,range=(0.5,2.0))

    bins_center_original = (bins_original[:-1] + bins_original[1:]) / 2
    bins_center_abinitio = (bins_abinitio[:-1] + bins_abinitio[1:]) / 2

    distance = stats.wasserstein_distance(bins_center_original,bins_center_abinitio,u_weights=hist_original,v_weights=hist_abinitio)
    print('distance:',distance)

    hist_original = hist_original/np.sum(hist_original)
    hist_abinitio = hist_abinitio/np.sum(hist_abinitio)



    plt.figure(figsize=(10,6))
    plt.bar(bins_original[:-1],hist_original,width=np.diff(bins_original),align='edge',label='original',alpha=0.5)
    plt.bar(bins_abinitio[:-1],hist_abinitio,width=np.diff(bins_abinitio),align='edge',label='abinitio',alpha=0.5)
    plt.xlabel('angle')
    plt.ylabel('Normalized frequency')
    plt.legend()
    plt.title('distance distribution of original and abinitio')
    plt.savefig('abinitio_distance_bad.png')
    plt.close()
    """