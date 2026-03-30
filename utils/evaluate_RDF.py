import torch
import argparse
import wandb
import os
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter1d
import numpy as np
from scipy.stats import wasserstein_distance
from matplotlib.gridspec import GridSpec


def calculate_wasserstein_distance(rdf1, rdf2):
    """
    2つのRDFのWasserstein距離を計算する関数

    Parameters:
    rdf1 (numpy.ndarray): 1つ目のRDF
    rdf2 (numpy.ndarray): 2つ目のRDF

    Returns:
    float: Wasserstein距離
    """
    return wasserstein_distance(rdf1, rdf2)

def mean_squared_error(rdf1, rdf2):
    """
    2つのRDFの平均二乗誤差（MSE）を計算する関数

    Parameters:
    rdf1 (numpy.ndarray): 1つ目のRDF
    rdf2 (numpy.ndarray): 2つ目のRDF

    Returns:
    float: 平均二乗誤差（MSE）
    """
    return np.mean((rdf1 - rdf2) ** 2)

def length_from_exO(position):
    exO = position[0]
    length_list = []
    for i in range(1,len(position)):
        length = torch.norm(position[i]-exO)
        length_list.append(length)
    return length_list


def RDF(position,sigma=5,R=5.0,dR=0.01,Normalize=False):
    length_list = length_from_exO(position)
    num_atom = position.shape[0]
    ro = num_atom/(4/3*np.pi*R**3)
    dR
    R = np.arange(0+dR,R+dR,dR)
    RDF = []
    for r in R:
        RDF.append(sum([1 for d in length_list if r < d < r+dR])/(4*np.pi*ro*r**2*dR))
    RDF_smoothed = gaussian_filter1d(RDF, sigma)
    if Normalize:
        RDF_smoothed = RDF_smoothed/np.max(RDF_smoothed)    
    return RDF_smoothed

def cos_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

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

    
def euclidean_distance(a,b):
    return np.linalg.norm(a-b)

def list_cos_similarity(original_graph_list,generated_graph_list):
    id_list = []
    cos_similarity_list = []
    original_coords_list, generated_coords_list = [],[]
    for i in range(len(original_graph_list)):
        original_graph = original_graph_list[i]
        generated_graph = generated_graph_list[i][-1]
        if original_graph.pos.shape[0] == 1:
            continue
        original_RDF = RDF(original_graph.pos.cpu())
        generated_RDF = RDF(generated_graph.pos.cpu())
        cos_similarity_value = cos_similarity(original_RDF,generated_RDF)
        cos_similarity_list.append(cos_similarity_value)
        id_list.append(0)
        #id_list.append(original_graph.id)
        original_coords_list.append(original_graph)
        generated_coords_list.append(generated_graph)
    id_cos_similarity_original_generated_list = list(zip(id_list,cos_similarity_list,original_coords_list,generated_coords_list)) #cos_similarityの値でソート
    return id_cos_similarity_original_generated_list

def list_euclidian_distance(original_graph_list,generated_graph_list):
    id_list = []
    euclidian_distance_list = []
    original_coords_list, generated_coords_list = [],[]
    for i in range(len(original_graph_list)):
        original_graph = original_graph_list[i]
        generated_graph = generated_graph_list[i][-1]
        if original_graph.pos.shape[0] == 1:
            continue
        original_RDF = RDF(original_graph.pos.cpu())
        generated_RDF = RDF(generated_graph.pos.cpu())
        euclidian_distance_value = euclidean_distance(original_RDF,generated_RDF)
        euclidian_distance_list.append(euclidian_distance_value)
        id_list.append(0)
        #id_list.append(original_graph.id)
        original_coords_list.append(original_graph)
        generated_coords_list.append(generated_graph)
    id_euclidian_distance_original_generated_list = list(zip(id_list,euclidian_distance_list,original_coords_list,generated_coords_list)) #euclidian_distanceの値でソート
    return id_euclidian_distance_original_generated_list


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--project_name', type=str, required=True)
    argparser.add_argument('--run_id', type=str, required=True)
    args = argparser.parse_args()

    run = wandb.init(project=args.project_name, id=args.run_id, resume='must')
    config = run.config
    generated_graph_save_path = config.generated_graph_save_path
    original_graph_save_path = config.original_graph_save_path
    #generated_graph_save_path = input('generated_graph_save_path:')
    #original_graph_save_path = input('original_graph_save_path:')
    
    if os.path.exists(generated_graph_save_path):
        generated_graph_list = torch.load(generated_graph_save_path)
        original_graph_list = torch.load(original_graph_save_path)
    else:
        generated_graph_list = torch.load(generated_graph_save_path.replace('/mnt',''))
        original_graph_list = torch.load(original_graph_save_path.replace('/mnt',''))
    gs = GridSpec(2,2,height_ratios=[1,1],width_ratios=[1,1])
    fig = plt.figure(figsize=(12,8))
    ax_cos =fig.add_subplot(gs[0,0])
    ax_euclidian = fig.add_subplot(gs[0,1])
    ax_mse = fig.add_subplot(gs[1,0])
    ax_wasserstein = fig.add_subplot(gs[1,1])

    original_RDF_list, generated_RDF_list = [],[]
    original_graph_list_for_eval, generated_graph_list_for_eval = [], []
    id_list = []
    cos_sim_list, euclidean_distance_list, mse_list, wasserstein_distance_list = [],[],[],[]
    eval_list = [] #[originalRDF,generatedRDF,cos_sim,euclidean_distance,mse,wasserstein_distance]を要素に持つリスト
    for i in range(len(original_graph_list)):
        original_graph = original_graph_list[i]
        generated_graph = generated_graph_list[i][-1]
        if original_graph.pos.shape[0] == 1:
            continue

        id = original_graph.id
        
        original_RDF = RDF(original_graph.pos.cpu())
        generated_RDF = RDF(generated_graph.pos.cpu())

        cos_sim = cos_similarity(original_RDF,generated_RDF)
        euclidean_distance_value = euclidean_distance(original_RDF,generated_RDF)
        mse = mean_squared_error(original_RDF,generated_RDF)
        wasserstein_distance_value = calculate_wasserstein_distance(original_RDF,generated_RDF)

        original_RDF_list.append(original_RDF)
        generated_RDF_list.append(generated_RDF)

        cos_sim_list.append(cos_sim)
        euclidean_distance_list.append(euclidean_distance_value)
        mse_list.append(mse)
        wasserstein_distance_list.append(wasserstein_distance_value)

        id_list.append(id)

        original_graph_list_for_eval.append(original_graph)
        generated_graph_list_for_eval.append(generated_graph)


    eval_list = list(zip(original_RDF_list,generated_RDF_list,cos_sim_list,euclidean_distance_list,mse_list,wasserstein_distance_list,original_graph_list_for_eval,generated_graph_list_for_eval,id_list))

    torch.save(eval_list,os.path.join(run.dir,'eval_list.pt'))
    wandb.config.update({'eval_list':os.path.join(run.dir,'eval_list.pt')},allow_val_change=True)

    original_RDF_array = np.array(original_RDF_list)
    generated_RDF_array = np.array(generated_RDF_list)

    ax_cos.hist(cos_sim_list,bins=40)
    ax_cos.set_xlabel('cos_similarity')
    ax_cos.set_ylabel('frequency')
    ax_cos.set_title('cos_similarity')
    
    ax_euclidian.hist(euclidean_distance_list,bins=40)
    ax_euclidian.set_xlabel('euclidean_distance')
    ax_euclidian.set_ylabel('frequency')
    ax_euclidian.set_title('euclidean_distance')

    ax_mse.hist(mse_list,bins=40)
    ax_mse.set_xlabel('mean_squared_error')
    ax_mse.set_ylabel('frequency')
    ax_mse.set_title('mean_squared_error')

    ax_wasserstein.hist(wasserstein_distance_list,bins=40)
    ax_wasserstein.set_xlabel('wasserstein_distance')
    ax_wasserstein.set_ylabel('frequency')
    ax_wasserstein.set_title('wasserstein_distance')

    wandb.log({'evaluate_RDF':wandb.Image(fig)})
    plt.close(fig)

    x_for_RDF = np.arange(0.01,5.01,0.01)

    gs_cos = GridSpec(2,3,height_ratios=[1,1],width_ratios=[1,1,1])
    fig_cos = plt.figure(figsize=(18,8))
    ax_cos_first = fig_cos.add_subplot(gs_cos[0,0])
    ax_cos_second = fig_cos.add_subplot(gs_cos[0,1])
    ax_cos_third = fig_cos.add_subplot(gs_cos[0,2])
    ax_cos_mid =fig_cos.add_subplot(gs_cos[1,0])
    ax_cos_worst = fig_cos.add_subplot(gs_cos[1,1])
    ax_cos_text = fig_cos.add_subplot(gs_cos[1,2])
    sorted_by_cos = sorted(eval_list,key=lambda x:x[2])
    ax_cos_first.plot(x_for_RDF,sorted_by_cos[-1][0],label='original')
    ax_cos_first.plot(x_for_RDF,sorted_by_cos[-1][1],label='generated')
    ax_cos_first.set_title('best_cos_similarity')
    ax_cos_second.plot(x_for_RDF,sorted_by_cos[-2][0],label='original')
    ax_cos_second.plot(x_for_RDF,sorted_by_cos[-2][1],label='generated')  
    ax_cos_second.set_title('second_best_cos_similarity')
    ax_cos_third.plot(x_for_RDF,sorted_by_cos[-3][0],label='original')
    ax_cos_third.plot(x_for_RDF,sorted_by_cos[-3][1],label='generated')
    ax_cos_third.set_title('third_best_cos_similarity')
    ax_cos_mid.plot(x_for_RDF,sorted_by_cos[len(sorted_by_cos)//2][0],label='original')
    ax_cos_mid.plot(x_for_RDF,sorted_by_cos[len(sorted_by_cos)//2][1],label='generated')
    ax_cos_mid.set_title('mid_cos_similarity')
    ax_cos_worst.plot(x_for_RDF,sorted_by_cos[0][0],label='original')
    ax_cos_worst.plot(x_for_RDF,sorted_by_cos[0][1],label='generated')
    ax_cos_worst.set_title('worst_cos_similarity')
    ax_cos_text.text(0,0.5,f'cos_similarity\nbest:{sorted_by_cos[0][-1]}\nsecond_best:{sorted_by_cos[1][-1]}\nthird_best:{sorted_by_cos[2][-1]}\nmid:{sorted_by_cos[len(sorted_by_cos)//2][-1]}\nworst:{sorted_by_cos[-1][-1]}',fontsize=12)
    ax_cos_text.axis('off')
    ax_cos_first.legend()
    ax_cos_second.legend()
    ax_cos_third.legend()
    ax_cos_mid.legend()
    ax_cos_worst.legend()
    
    gs_euclidean = GridSpec(2,3,height_ratios=[1,1],width_ratios=[1,1,1])
    fig_euclidean = plt.figure(figsize=(18,8))
    ax_euclidean_first = fig_euclidean.add_subplot(gs_euclidean[0,0])
    ax_euclidean_second = fig_euclidean.add_subplot(gs_euclidean[0,1])
    ax_euclidean_third = fig_euclidean.add_subplot(gs_euclidean[0,2])
    ax_euclidean_mid =fig_euclidean.add_subplot(gs_euclidean[1,0])
    ax_euclidean_worst = fig_euclidean.add_subplot(gs_euclidean[1,1])
    ax_euclidean_text = fig_euclidean.add_subplot(gs_euclidean[1,2])
    sorted_by_euclidean = sorted(eval_list,key=lambda x:x[3])
    ax_euclidean_first.plot(x_for_RDF,sorted_by_euclidean[0][0],label='original')
    ax_euclidean_first.plot(x_for_RDF,sorted_by_euclidean[0][1],label='generated')
    ax_euclidean_first.set_title('best_euclidean_distance')
    ax_euclidean_second.plot(x_for_RDF,sorted_by_euclidean[1][0],label='original')
    ax_euclidean_second.plot(x_for_RDF,sorted_by_euclidean[1][1],label='generated')
    ax_euclidean_second.set_title('second_best_euclidean_distance')
    ax_euclidean_third.plot(x_for_RDF,sorted_by_euclidean[2][0],label='original')
    ax_euclidean_third.plot(x_for_RDF,sorted_by_euclidean[2][1],label='generated')
    ax_euclidean_third.set_title('third_best_euclidean_distance')
    ax_euclidean_mid.plot(x_for_RDF,sorted_by_euclidean[len(sorted_by_euclidean)//2][0],label='original')
    ax_euclidean_mid.plot(x_for_RDF,sorted_by_euclidean[len(sorted_by_euclidean)//2][1],label='generated')
    ax_euclidean_mid.set_title('mid_euclidean_distance')
    ax_euclidean_worst.plot(x_for_RDF,sorted_by_euclidean[-1][0],label='original')
    ax_euclidean_worst.plot(x_for_RDF,sorted_by_euclidean[-1][1],label='generated')
    ax_euclidean_worst.set_title('worst_euclidean_distance')
    ax_euclidean_text.text(0,0.5,f'euclidean_distance\nbest:{sorted_by_euclidean[0][-1]}\nsecond_best:{sorted_by_euclidean[1][-1]}\nthird_best:{sorted_by_euclidean[2][-1]}\nmid:{sorted_by_euclidean[len(sorted_by_euclidean)//2][-1]}\nworst:{sorted_by_euclidean[-1][-1]}',fontsize=12)
    ax_euclidean_text.axis('off')
    ax_euclidean_first.legend()
    ax_euclidean_second.legend()
    ax_euclidean_third.legend()
    ax_euclidean_mid.legend()
    ax_euclidean_worst.legend()


    gs_mse = GridSpec(2,3,height_ratios=[1,1],width_ratios=[1,1,1])
    fig_mse = plt.figure(figsize=(18,8))
    ax_mse_first = fig_mse.add_subplot(gs_mse[0,0])
    ax_mse_second = fig_mse.add_subplot(gs_mse[0,1])
    ax_mse_third = fig_mse.add_subplot(gs_mse[0,2])
    ax_mse_mid =fig_mse.add_subplot(gs_mse[1,0])
    ax_mse_worst = fig_mse.add_subplot(gs_mse[1,1])
    ax_mse_text = fig_mse.add_subplot(gs_mse[1,2])
    sorted_by_mse = sorted(eval_list,key=lambda x:x[4])
    ax_mse_first.plot(x_for_RDF,sorted_by_mse[0][0],label='original')
    ax_mse_first.plot(x_for_RDF,sorted_by_mse[0][1],label='generated')
    ax_mse_first.set_title('best_mse')
    ax_mse_second.plot(x_for_RDF,sorted_by_mse[1][0],label='original')
    ax_mse_second.plot(x_for_RDF,sorted_by_mse[1][1],label='generated')
    ax_mse_second.set_title('second_best_mse')
    ax_mse_third.plot(x_for_RDF,sorted_by_mse[2][0],label='original')
    ax_mse_third.plot(x_for_RDF,sorted_by_mse[2][1],label='generated')
    ax_mse_third.set_title('third_best_mse')
    ax_mse_mid.plot(x_for_RDF,sorted_by_mse[len(sorted_by_mse)//2][0],label='original')
    ax_mse_mid.plot(x_for_RDF,sorted_by_mse[len(sorted_by_mse)//2][1],label='generated')
    ax_mse_mid.set_title('mid_mse')
    ax_mse_worst.plot(x_for_RDF,sorted_by_mse[-1][0],label='original')
    ax_mse_worst.plot(x_for_RDF,sorted_by_mse[-1][1],label='generated')
    ax_mse_worst.set_title('worst_mse')
    ax_mse_text.text(0,0.5,f'mse\nbest:{sorted_by_mse[0][-1]}\nsecond_best:{sorted_by_mse[1][-1]}\nthird_best:{sorted_by_mse[2][-1]}\nmid:{sorted_by_mse[len(sorted_by_mse)//2][-1]}\nworst:{sorted_by_mse[-1][-1]}',fontsize=12)
    ax_mse_text.axis('off')
    ax_mse_first.legend()
    ax_mse_second.legend()
    ax_mse_third.legend()
    ax_mse_mid.legend()
    ax_mse_worst.legend()
    
    gs_wasserstein = GridSpec(2,3,height_ratios=[1,1],width_ratios=[1,1,1])
    fig_wasserstein = plt.figure(figsize=(18,8))
    ax_wasserstein_first = fig_wasserstein.add_subplot(gs_wasserstein[0,0])
    ax_wasserstein_second = fig_wasserstein.add_subplot(gs_wasserstein[0,1])
    ax_wasserstein_third = fig_wasserstein.add_subplot(gs_wasserstein[0,2])
    ax_wasserstein_mid =fig_wasserstein.add_subplot(gs_wasserstein[1,0])
    ax_wasserstein_worst = fig_wasserstein.add_subplot(gs_wasserstein[1,1])
    ax_wasserstein_text = fig_wasserstein.add_subplot(gs_wasserstein[1,2])
    sorted_by_wasserstein = sorted(eval_list,key=lambda x:x[5])
    ax_wasserstein_first.plot(x_for_RDF,sorted_by_wasserstein[0][0],label='original')
    ax_wasserstein_first.plot(x_for_RDF,sorted_by_wasserstein[0][1],label='generated')
    ax_wasserstein_first.set_title('best_wasserstein_distance')
    ax_wasserstein_second.plot(x_for_RDF,sorted_by_wasserstein[1][0],label='original')
    ax_wasserstein_second.plot(x_for_RDF,sorted_by_wasserstein[1][1],label='generated')
    ax_wasserstein_second.set_title('second_best_wasserstein_distance')
    ax_wasserstein_third.plot(x_for_RDF,sorted_by_wasserstein[2][0],label='original')
    ax_wasserstein_third.plot(x_for_RDF,sorted_by_wasserstein[2][1],label='generated')
    ax_wasserstein_third.set_title('third_best_wasserstein_distance')
    ax_wasserstein_mid.plot(x_for_RDF,sorted_by_wasserstein[len(sorted_by_wasserstein)//2][0],label='original')
    ax_wasserstein_mid.plot(x_for_RDF,sorted_by_wasserstein[len(sorted_by_wasserstein)//2][1],label='generated')
    ax_wasserstein_mid.set_title('mid_wasserstein_distance')
    ax_wasserstein_worst.plot(x_for_RDF,sorted_by_wasserstein[-1][0],label='original')
    ax_wasserstein_worst.plot(x_for_RDF,sorted_by_wasserstein[-1][1],label='generated')
    ax_wasserstein_worst.set_title('worst_wasserstein_distance')
    ax_wasserstein_text.text(0,0.5,f'wasserstein_distance\nbest:{sorted_by_wasserstein[0][-1]}\nsecond_best:{sorted_by_wasserstein[1][-1]}\nthird_best:{sorted_by_wasserstein[2][-1]}\nmid:{sorted_by_wasserstein[len(sorted_by_wasserstein)//2][-1]}\nworst:{sorted_by_wasserstein[-1][-1]}',fontsize=12)
    ax_wasserstein_text.axis('off')
    ax_wasserstein_first.legend()
    ax_wasserstein_second.legend()
    ax_wasserstein_third.legend()
    ax_wasserstein_mid.legend()
    ax_wasserstein_worst.legend()

    wandb.log({'cos_similarity':wandb.Image(fig_cos)})
    wandb.log({'euclidean_distance':wandb.Image(fig_euclidean)})
    wandb.log({'mean_squared_error':wandb.Image(fig_mse)})
    wandb.log({'wasserstein_distance':wandb.Image(fig_wasserstein)})
    plt.close(fig_cos)
    plt.close(fig_euclidean)
    plt.close(fig_mse)
    plt.close(fig_wasserstein)


    run.finish()

