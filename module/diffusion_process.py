import numpy as np
import torch
import torch.nn as nn

def sigmoid(x):
        return 1 / (1 + np.exp(-x))
"""
def calculate_center_of_mass(local_env_list): 
    atom_mass_list, vector_list = [], []
    for atom in local_env_list:
        #print(local_env_list,atom,type(atom))
        if atom[0] == [1,0]:
            vector_list.append(atom[1])
            atom_mass_list.append(16)
        elif atom[0] == [0,1]:
            vector_list.append(atom[1])
            atom_mass_list.append(28.0855)
    sum_mass = sum(atom_mass_list)
    x_CoM = sum([atom_mass_list[i]*vector_list[i][0] for i in range(len(atom_mass_list))])/sum_mass
    y_CoM = sum([atom_mass_list[i]*vector_list[i][1] for i in range(len(atom_mass_list))])/sum_mass
    z_CoM = sum([atom_mass_list[i]*vector_list[i][2] for i in range(len(atom_mass_list))])/sum_mass
    return np.array([x_CoM, y_CoM, z_CoM])

def Kabsch_algorithm(local_env_list_at_zero, local_env_list_at_t):
    P,Q = [],[]
    for atom in local_env_list_at_zero:
        P.append(atom[1])
    for atom in local_env_list_at_t:
        Q.append(atom[1])
    P = np.array(P)
    Q = np.array(Q)

    CoMt = calculate_center_of_mass(local_env_list_at_t)

    
    #The Kabsch algorithm is a method for calculating the optimal rotation matrix that minimizes the RMSD (root mean squared deviation) between two paired sets of points.
    
    # Center the data
    P = P - CoMt


    # Compute the covariance matrix
    H = np.dot(np.transpose(P), Q)

    # Use the SVD algorithm to compute the optimal rotation matrix
    U, S, Vt = np.linalg.svd(H)
    V = np.transpose(Vt)
    D = np.diag([1, 1, np.linalg.det(np.dot(V, U))])

    # Compute the optimal rotation matrix
    R = np.dot(V, np.dot(D, U))

    return R
"""
"""
def calculate_center_of_mass_torch(nodes,tensor):
    atom_mass_list, vector_list = [], []
    num_node = len(nodes)
    #atom_type_dict = {torch.tensor([1,0],dtype=torch.float32):"O",torch.tensor([0,1],dtype=torch.float32):"Si"}
    for i in range(num_node):
        if int(nodes[i][0]) == 1 and int(nodes[i][1]) == 0:
            vector_list.append(tensor[i])
            atom_mass_list.append(16)
        elif int(nodes[i][0]) == 0 and int(nodes[i][1]) == 1:
            vector_list.append(tensor[i])
            atom_mass_list.append(28.0855)
    sum_mass = sum(atom_mass_list)
    if sum_mass == 0:
        print(atom_mass_list)
    r_CoM = sum([atom_mass_list[i]*vector_list[i] for i in range(len(atom_mass_list))])/sum_mass
    return r_CoM
"""
def Kabsch_algorithm_torch(tensor_at_zero,tensor_at_t):
    
    #CoMt = calculate_center_of_mass_torch(tensor_at_t)
    CoMt = 0
    for i in range(tensor_at_t.shape[0]):
        CoMt += tensor_at_t[i]
    CoMt /= tensor_at_t.shape[0]


    """
    The Kabsch algorithm is a method for calculating the optimal rotation matrix that minimizes the RMSD (root mean squared deviation) between two paired sets of points.
    """
    # Center the data
    tensor_at_zero = tensor_at_zero - CoMt
    tensor_at_t = tensor_at_t - CoMt

    P = tensor_at_zero
    Q = tensor_at_t
    
    if not torch.isfinite(P).all():
        print('tensor_at_t:',tensor_at_t)
        print('CoMt:',CoMt)
        print("P:",P)
        raise ValueError("P contains non-finite values (inf or NaN)")
    if not torch.isfinite(Q).all():
        print("Q:",Q)
        raise ValueError("Q contains non-finite values (inf or NaN)")
    
    # Compute the covariance matrix
    H = torch.matmul(P.T, Q)

    # Use the SVD algorithm to compute the optimal rotation matrix
    U, S, Vt = torch.linalg.svd(H)
    V = Vt.T
    D = torch.diag(torch.tensor([1, 1, torch.linalg.det(torch.matmul(V, U))]))

    # Compute the optimal rotation matrix
    R = torch.matmul(V, torch.matmul(D, U))

    return R

class DiffusionProcess():
    def __init__(self,initial_beta,final_beta,num_diffusion_timestep,schedule_func='sigmoid'):
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.num_diffusion_timestep = num_diffusion_timestep

        if schedule_func == 'sigmoid':
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            beta_schedule = np.linspace(-6,6,num_diffusion_timestep)
            beta_schedule = sigmoid(beta_schedule) * (final_beta - initial_beta) + initial_beta
            self.beta_schedule = beta_schedule
        elif schedule_func =='linear':
            beta_schedule = np.linspace(initial_beta,final_beta,num_diffusion_timestep)
            self.beta_schedule = beta_schedule

        self.alpha = [1-beta for beta in self.beta_schedule]

    def alpha_bar_t(self,t):
        alpha_bar = 1
        for i in range(t):
            alpha_bar *= self.alpha[i]
        return alpha_bar
    
    def diffuse_zero_to_t_torch(self,C_at_zero,t):
        alpha_bar = torch.tensor(self.alpha_bar_t(t),dtype=torch.float32)
        noise = torch.randn(C_at_zero.shape)
        C_at_t = torch.sqrt(alpha_bar) * C_at_zero + torch.sqrt(1-alpha_bar) * noise
        return C_at_t
    
    def diffuse_by_SNR_schedule(self,C_at_zero,t):
        squared_sigma_t = 1/(1+torch.exp(12+(-17/5000)*torch.tensor(t,dtype=torch.float32)))
        C_at_t = torch.normal(torch.sqrt(1-squared_sigma_t)*C_at_zero,torch.sqrt(squared_sigma_t))
        return C_at_t

    def equivariant_SNR(self,C_at_zero,C_at_t,t):
        CoM = 0
        for i in range(C_at_t.shape[0]):
            CoM += C_at_t[i]
        CoM /= C_at_t.shape[0]
        aligned_C_at_zero = (C_at_zero - CoM)
        aligned_C_at_t = C_at_t - CoM
        squared_sigma_t = 1/(1+torch.exp(12+(-17/5000)*torch.tensor(t,dtype=torch.float32)))
        R = Kabsch_algorithm_torch(aligned_C_at_zero,aligned_C_at_t)
        aligned_C_at_zero = aligned_C_at_zero @ R
        equivariant_epsilon = (aligned_C_at_t - torch.sqrt(1-squared_sigma_t)*aligned_C_at_zero) / torch.sqrt(squared_sigma_t)
        return equivariant_epsilon

    def equivariant_epsilon_torch(self,C_at_zero,C_at_t,t,aligned_standard='CoM'):
        """
        if abs(int(C_at_t[0,0])) > 100:
            print('t:',t)
            print('C_at_zero:',C_at_zero)
            print('C_at_t:',C_at_t)
        """
        if aligned_standard == 'CoM':
            CoM = 0
            for i in range(C_at_t.shape[0]):
                CoM += C_at_t[i]
            CoM /= C_at_t.shape[0]
            aligned_C_at_zero = (C_at_zero - CoM)
            aligned_C_at_t = C_at_t - CoM
        elif aligned_standard == 'exO':
            standard = torch.stack([C_at_t[0] for d in range(C_at_t.shape[0])])
            aligned_C_at_zero = C_at_zero - standard
            aligned_C_at_t = C_at_t - standard
        alpha_bar = torch.tensor(self.alpha_bar_t(t),dtype=torch.float32)
        R = Kabsch_algorithm_torch(aligned_C_at_zero,aligned_C_at_t)
        aligned_C_at_zero = aligned_C_at_zero @ R
        equivariant_epsilon = (aligned_C_at_t - torch.sqrt(alpha_bar)*aligned_C_at_zero) / torch.sqrt(1-alpha_bar)
        return equivariant_epsilon

    def calculate_mu(self,C_at_t,epsilon,time):
        alpha_bar = torch.tensor(self.alpha_bar_t(time))
        alpha_t = torch.tensor(self.alpha[time-1])
        beta_t = torch.tensor(self.beta_schedule[time-1],dtype=torch.float32)
        mu = (C_at_t - beta_t * epsilon / torch.sqrt(1-alpha_bar)) / torch.sqrt(alpha_t)
        return mu

    def calculate_onestep_before(self,mu,time):
        std = (1 - self.alpha_bar_t(int(time-1))) * self.beta_schedule[time-1] / (1 - self.alpha_bar_t(int(time)))
        return torch.normal(mean=mu,std=std)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def plt_SNR(initial_beta,final_beta,num_diffusion_timestep,schedule_func):
        diffusion_process = DiffusionProcess(initial_beta,final_beta,num_diffusion_timestep,schedule_func)
        SNR_x = np.linspace(1,num_diffusion_timestep,num_diffusion_timestep)
        SNR_y = []
        for t in SNR_x:
            alpha_bar = diffusion_process.alpha_bar_t(int(t))
            SNR_y.append(np.log(alpha_bar/(1-alpha_bar)))
        return plt.plot(SNR_x,SNR_y,label='schedule_func:{}, initial:{}, final:{}'.format(schedule_func,initial_beta,final_beta))
    
    def plt_beta(initial_beta,final_beta,num_diffusion_timestep,schedule_func):
        diffusion_process = DiffusionProcess(initial_beta,final_beta,num_diffusion_timestep,schedule_func)
        beta_x = np.linspace(1,num_diffusion_timestep,num_diffusion_timestep)
        beta_y = diffusion_process.beta_schedule
        return plt.plot(beta_x,beta_y,label='schedule_func:{}, initial:{}, final:{}'.format(schedule_func,initial_beta,final_beta))
    
    def plt_alpha(initial_beta,final_beta,num_diffusion_timestep,schedule_func):
        diffusion_process = DiffusionProcess(initial_beta,final_beta,num_diffusion_timestep,schedule_func)
        alpha_x = np.linspace(1,num_diffusion_timestep,num_diffusion_timestep)
        alpha_y = diffusion_process.alpha
        return plt.plot(alpha_x,alpha_y,label='schedule_func:{}, initial:{}, final:{}'.format(schedule_func,initial_beta,final_beta))

    def plt_alpha_bar(initial_beta,final_beta,num_diffusion_timestep,schedule_func):
        diffusion_process = DiffusionProcess(initial_beta,final_beta,num_diffusion_timestep,schedule_func)
        alpha_bar_x = np.linspace(1,num_diffusion_timestep,num_diffusion_timestep)
        alpha_bar_y = []
        for t in alpha_bar_x:
            alpha_bar_y.append(diffusion_process.alpha_bar_t(int(t)))
        return plt.plot(alpha_bar_x,alpha_bar_y,label='schedule_func:{}, initial:{}, final:{}'.format(schedule_func,initial_beta,final_beta))
    """
    fig, ax = plt.subplots()
    plt_SNR(1.0e-7,2.0e-3,3000,'sigmoid')
    plt_SNR(0.0001,0.01,3000,'sigmoid')
    ax.set_xlabel('t')
    ax.set_ylabel('SNR')
    plt.legend()
    plt.savefig('SNR.png')
    plt.close()
    
    fig, ax = plt.subplots()
    plt_beta(1.0e-5,0.002,5000,'sigmoid')
    plt_beta(0.0001,0.01,5000,'sigmoid')
    ax.set_xlabel('t')
    ax.set_ylabel('beta')
    plt.legend()
    plt.savefig('beta.png')
    plt.close()

    fig, ax = plt.subplots()
    plt_alpha(1.0e-5,0.002,5000,'sigmoid')
    plt_alpha(0.0001,0.01,5000,'sigmoid')
    ax.set_xlabel('t')
    ax.set_ylabel('alpha')
    plt.legend()
    plt.savefig('alpha.png')
    plt.close()
    """

    fig, ax = plt.subplots()
    plt_alpha_bar(1.0e-5,0.01,5000,'sigmoid')
    plt_alpha_bar(1.0e-5,0.009,5000,'sigmoid')
    plt_alpha_bar(1.0e-5,0.008,5000,'sigmoid')
    plt_alpha_bar(1.0e-5,0.007,5000,'sigmoid')
    plt_alpha_bar(1.0e-5,0.006,5000,'sigmoid')
    plt_alpha_bar(1.0e-5,0.005,5000,'sigmoid')

    plt_alpha_bar(0.0001,0.01,5000,'sigmoid')
    ax.set_xlabel('t')
    ax.set_ylabel('alpha_bar')
    plt.legend()
    plt.savefig('alpha_bar_3.png')
    plt.close()