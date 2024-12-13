from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from moments_engr import *
from agglo_model_clust import *
from cluster_acc_viz_funcs import *
from DNN_FT_funcs import *
from DNN_AMC_funcs import *
from model_classes import *


def full_comparison_run(data_splits, nested_clus_model_dict, clus_asgn_loader, config,
                        num_local_training_epochs=50, num_ft_epochs=50, ft_lr=0.001, cluster_merge_iter='Iter18'):
    train_pids = np.unique(data_splits['train']['participant_ids'])
    novel_participant_ft_data = data_splits['novel_trainFT']
    novel_participant_test_data = data_splits['cross_subject_test']
    novel_pids = np.unique(data_splits['novel_trainFT']['participant_ids'])
    novel_pid_clus_asgn_data = clus_asgn_loader['novel_trainFT']

    cluster_lst = list(nested_clus_model_dict[cluster_merge_iter].keys())

    novel_pid_res_dict = {}
    for pid_count, pid in enumerate(novel_pids):
        print(f"PID {pid}, {pid_count}/{len(novel_pids)}")
        novel_pid_res_dict[pid] = {}

        # Create the testloader by segmenting out this specific pid
        # Filter based on participant_id
        indices = [i for i, datasplit_pid in enumerate(novel_participant_ft_data['participant_ids']) if datasplit_pid == pid]
        ############## Novel Participant Finetuning Dataset ##############
        ft_dataset = GestureDataset([novel_participant_ft_data['feature'][i] for i in indices], [novel_participant_ft_data['labels'][i] for i in indices])
        ft_loader = DataLoader(ft_dataset, batch_size=config["batch_size"], shuffle=True)
        ############## Novel Participant Intra Testing Dataset ##############
        indices = [i for i, datasplit_pid in enumerate(novel_participant_test_data['participant_ids']) if datasplit_pid == pid]
        intra_test_dataset = GestureDataset([novel_participant_test_data['feature'][i] for i in indices], [novel_participant_test_data['labels'][i] for i in indices])
        intra_test_loader = DataLoader(intra_test_dataset, batch_size=config["batch_size"], shuffle=True)
        ############## Novel Participant Cross Testing Dataset ##############
        indices = [i for i, datasplit_pid in enumerate(novel_participant_test_data['participant_ids']) if datasplit_pid != pid]
        cross_test_dataset = GestureDataset([novel_participant_test_data['feature'][i] for i in indices], [novel_participant_test_data['labels'][i] for i in indices])
        cross_test_loader = DataLoader(intra_test_dataset, batch_size=config["batch_size"], shuffle=True)
        ############## One Trial Cluster Assignment Dataset ##############
        indices = [i for i, datasplit_pid in enumerate(novel_pid_clus_asgn_data['participant_ids']) if datasplit_pid == pid]
        clust_asgn_dataset = GestureDataset([novel_pid_clus_asgn_data['feature'][i] for i in indices], [novel_pid_clus_asgn_data['labels'][i] for i in indices])
        clust_asgn_loader = DataLoader(clust_asgn_dataset, batch_size=config["batch_size"], shuffle=True)

        # 1) Train a local CNN model
        local_model = CNNModel_3layer(config, input_dim=80, num_classes=10)

        local_results = main_training_pipeline(data_splits=None, train_intra_cross_loaders=[ft_loader, intra_test_loader, cross_test_loader],
                            all_participants=train_pids, test_participants=novel_pids, model_type=local_model,  
                            num_epochs=num_local_training_epochs, config=config)
        # This is kind of repeated but whatever
        local_clus_res = evaluate_model(local_model, intra_test_loader)
        novel_pid_res_dict[pid]["local_acc"] = local_clus_res["accuracy"]

        # 2) Have the pretrained CNN model from the best cluster do inference
        #   - Have all cluster models do inference and compare assign to whichever cluster gives best results
        #pretrained_generic_CNN_model

        # Apply all the cluster models at the chosen iteration on the given participant data
        ## Record that cluster's performance (all cluster's performances?... Ideally)

        clus_model_res_dict = {}
        for clus_id in cluster_lst:
            clus_model = nested_clus_model_dict['Iter18'][clus_id]
            clus_res = evaluate_model(clus_model, clust_asgn_loader) 
            clus_acc = clus_res["accuracy"]
            clus_model_res_dict[clus_id] = clus_acc
            
        # Assign participant to highest scoring cluster
        # Find the key with the highest accuracy
        max_key = max(clus_model_res_dict, key=clus_model_res_dict.get)
        max_value = clus_model_res_dict[max_key]
        print(f"The highest accuracy is {max_value} and it is associated with the key {max_key}.")
        print("Full cluster assignment results dict:")
        print(clus_model_res_dict)

        original_cluster_model = nested_clus_model_dict['Iter18'][max_key]

        # Have the pretrained CNN model from the best cluster do inference
        pretrained_clus_res = evaluate_model(original_cluster_model, intra_test_loader)
        novel_pid_res_dict[pid]["pretrained_acc"] = pretrained_clus_res["accuracy"]

        # 3) FT the above model (or all??) pretrained CNN model on the participant
        ft_model, original_cluster_model, train_loss_log, test_loss_log = fine_tune_model(
            original_cluster_model, ft_loader, intra_test_loader, num_epochs=num_ft_epochs, lr=ft_lr)
        ft_clus_res = evaluate_model(ft_model, intra_test_loader)
        novel_pid_res_dict[pid]["ft_acc"] = ft_clus_res["accuracy"]
    
    return novel_pid_res_dict


def plot_boxplot_comparisons(novel_pid_res_dict, save_fig=False, my_title='Final CNN Accuracy at 50 Epochs', colors=['blue', 'purple', 'green']):
    local_acc_data = [] 
    pretrained_acc_data = [] 
    ft_acc_data = [] 
    # Collecting data from the dictionary 
    for pid in novel_pid_res_dict: 
        local_acc_data.append(novel_pid_res_dict[pid]['local_acc']) 
        pretrained_acc_data.append(novel_pid_res_dict[pid]['pretrained_acc']) 
        ft_acc_data.append(novel_pid_res_dict[pid]['ft_acc']) 

    # Plotting box plots with improved aesthetics 
    fig, ax = plt.subplots(figsize=(10, 6)) 
    data = [local_acc_data, pretrained_acc_data, ft_acc_data] 
    # Colors for each boxplot 
    bp = ax.boxplot(data, patch_artist=True, labels=['Local Accuracy', 'Pretrained Accuracy', 'Fine-Tuned Accuracy']) 
    # Customize boxplot colors 
    for patch, color in zip(bp['boxes'], colors): 
        patch.set_facecolor(color) 
    # Remove top and right borders 
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False) 
    # Increase font sizes 
    ax.set_ylabel('Accuracy', fontsize=16) 
    ax.set_title(my_title, fontsize=20) 
    ax.tick_params(axis='both', labelsize=14) 
    if save_fig:
        plt.savefig(f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results\\Final_CNN_Acc.png", dpi=500) 
    plt.show()