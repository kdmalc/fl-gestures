from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from moments_engr import *
from agglo_model_clust import *
from cluster_acc_viz_funcs import *
from DNN_FT_funcs import *
from DNN_AMC_funcs import *
#from model_classes import *
from revamped_model_classes import *


def full_comparison_run(finetuning_datasplits, cluster_assgnmt_data_splits, config, pretrained_generic_model,
                        nested_clus_model_dict, model_str, cluster_iter_str='Iter18'):
    
    os.makedirs(config['results_save_dir'], exist_ok=True)

    local_user_dict = prepare_data_for_local_models(finetuning_datasplits, model_str, config)
    
    #train_pids = np.unique(finetuning_datasplits['train']['participant_ids'])
    novel_participant_ft_data = finetuning_datasplits['novel_trainFT']
    novel_participant_test_data = finetuning_datasplits['cross_subject_test']
    novel_pids = np.unique(finetuning_datasplits['novel_trainFT']['participant_ids'])
    novel_pid_clus_asgn_data = cluster_assgnmt_data_splits['novel_trainFT']

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
        #cross_test_loader = DataLoader(cross_test_dataset, batch_size=config["batch_size"], shuffle=True)
        ############## One Trial Cluster Assignment Dataset ##############
        indices = [i for i, datasplit_pid in enumerate(novel_pid_clus_asgn_data['participant_ids']) if datasplit_pid == pid]
        clust_asgn_dataset = GestureDataset([novel_pid_clus_asgn_data['feature'][i] for i in indices], [novel_pid_clus_asgn_data['labels'][i] for i in indices])
        clust_asgn_loader = DataLoader(clust_asgn_dataset, batch_size=config["batch_size"], shuffle=True)

        # 1) Train a local model
        ## MomonaNets do not need input_dim and num_classes, other models need to be updated to infer that or pull it from config
        #local_model = select_model(model_str, config)
        ## This passes in the model object! Not the model string...
        #local_results = main_training_pipeline(data_splits=None, all_participants=train_pids, test_participants=novel_pids, model_type=model_str, config=config, 
        #                   train_intra_cross_loaders=[ft_loader, intra_test_loader, cross_test_loader])
        ## This is kind of repeated but whatever
        #local_clus_res = evaluate_model(local_model, intra_test_loader)
        #novel_pid_res_dict[pid]["local_acc"] = local_clus_res["accuracy"]

        # Wait can I just pass finetuning datasplits in here? I dont think so since I'm using FT not train data?
        local_res = main_training_pipeline(data_splits=None, all_participants=list(set(novel_participant_test_data['participant_ids'])), 
                                                test_participants=finetuning_datasplits['cross_subject_test']['participant_ids'], 
                                                model_type=model_str, config=config, 
                                                train_intra_cross_loaders=local_user_dict[pid])
        local_clus_res = evaluate_model(local_res["model"], intra_test_loader)
        novel_pid_res_dict[pid]["local_acc"] = local_clus_res["accuracy"]

        # 2) Test the full pretrained model
        generic_clus_res = evaluate_model(pretrained_generic_model, intra_test_loader)
        novel_pid_res_dict[pid]["centralized_acc"] = generic_clus_res["accuracy"]

        # 3) Test finetuned pretrained model
        #def fine_tune_model(finetuned_model, fine_tune_loader, config, timestamp, test_loader=None, pid=None, use_earlystopping=None):
        ft_centralized_model, _, _, _ = fine_tune_model(
            pretrained_generic_model, ft_loader, config, config['timestamp'], test_loader=intra_test_loader, pid=pid)
        ft_centralized_res = evaluate_model(ft_centralized_model, intra_test_loader)
        novel_pid_res_dict[pid]["ft_centralized_acc"] = ft_centralized_res["accuracy"]

        # 4) CLUSTER MODEL: Have the pretrained model from the best cluster do inference
        #   - Have all cluster models do inference and compare assign to whichever cluster gives best results
        # Apply all the cluster models at the chosen iteration on the given participant data
        ## Record that cluster's performance (all cluster's performances?... Ideally)
        clus_model_res_dict = {}
        cluster_lst = list(nested_clus_model_dict[cluster_iter_str].keys())
        for clus_id in cluster_lst:
            clus_model = nested_clus_model_dict[cluster_iter_str][clus_id]
            clus_res = evaluate_model(clus_model, clust_asgn_loader) 
            clus_acc = clus_res["accuracy"]
            clus_model_res_dict[clus_id] = clus_acc
        # Assign participant to highest scoring cluster
        # Find the key with the highest accuracy
        max_key = max(clus_model_res_dict, key=clus_model_res_dict.get)
        max_value = clus_model_res_dict[max_key]
        print(f"Cluster {max_key} had the highest accuracy ({max_value})")
        if config['verbose']:
            print("Full cluster assignment results dict:")
            print(clus_model_res_dict)
        original_cluster_model = nested_clus_model_dict[cluster_iter_str][max_key]
        # Have the pretrained model from the best cluster do inference
        pretrained_clus_res = evaluate_model(original_cluster_model, intra_test_loader)
        novel_pid_res_dict[pid]["pretrained_cluster_acc"] = pretrained_clus_res["accuracy"]

        # 5) FT the pretrained cluster model on the participant
        # ft_model, original_cluster_model, train_loss_log, test_loss_log = fine_tune_model()
        #def fine_tune_model(finetuned_model, fine_tune_loader, config, timestamp, test_loader=None, pid=None, use_earlystopping=None):
        ft_cluster_model, original_cluster_model, _, _ = fine_tune_model(
            original_cluster_model, ft_loader, config, config['timestamp'], test_loader=intra_test_loader, pid=pid)
        ft_clus_res = evaluate_model(ft_cluster_model, intra_test_loader)
        novel_pid_res_dict[pid]["ft_cluster_acc"] = ft_clus_res["accuracy"]

    local_acc_data = [] 
    centralized_acc_data = [] 
    ft_centralized_acc_data = []
    pretrained_cluster_acc_data = []
    ft_cluster_acc_data = [] 
    # Collecting data from the dictionary 
    for pid in novel_pid_res_dict: 
        local_acc_data.append(novel_pid_res_dict[pid]['local_acc']) 
        centralized_acc_data.append(novel_pid_res_dict[pid]['centralized_acc']) 
        ft_centralized_acc_data.append(novel_pid_res_dict[pid]['ft_centralized_acc']) 
        pretrained_cluster_acc_data.append(novel_pid_res_dict[pid]['pretrained_cluster_acc']) 
        ft_cluster_acc_data.append(novel_pid_res_dict[pid]['ft_cluster_acc']) 

    data_dict = {'local_acc_data':local_acc_data, 'centralized_acc_data':centralized_acc_data, 'ft_centralized_acc_data':ft_centralized_acc_data, 'pretrained_cluster_acc_data':pretrained_cluster_acc_data, 'ft_cluster_acc_data':ft_cluster_acc_data}
    return data_dict


def plot_model_acc_boxplots(data_dict, model_str=None, colors_lst=None, my_title=None, save_fig=False, plot_save_name=None, save_dir="C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results"):
    # Default order
    #data = [data_dict['local_acc_data'], data_dict['centralized_acc_data'], data_dict['ft_centralized_acc_data'], data_dict['pretrained_cluster_acc_data'], data_dict['ft_cluster_acc_data']] 
    # Ordering according to performance:
    data = [data_dict['centralized_acc_data'], data_dict['pretrained_cluster_acc_data'], data_dict['ft_centralized_acc_data'], data_dict['local_acc_data'], data_dict['ft_cluster_acc_data']] 

    if colors_lst is None:
        colors_lst = ['blue', 'purple', 'orange', 'green', 'yellow']
    if my_title is None and model_str is not None:
        my_title = f"{model_str} Accuracies Averaged Across Participants"

    fig, ax = plt.subplots(figsize=(6, 6)) 
    bp = ax.boxplot(data, patch_artist=True, labels=['Generic Centralized', 'Pretrained Cluster', 'Fine-Tuned Centralized', 'Local', 'Fine-Tuned Cluster']) 
    # Customize boxplot colors 
    for patch, color in zip(bp['boxes'], colors_lst): 
        patch.set_facecolor(color) 
    # Remove top and right borders 
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False) 
    # Increase font sizes 
    ax.set_ylabel('Accuracy', fontsize=14) 
    ax.set_title(my_title, fontsize=20) 
    ax.tick_params(axis='both', labelsize=14) 
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{save_dir}\\{plot_save_name}.png", dpi=500, bbox_inches='tight') 
    plt.show()

    # FOR LOCAL
def group_data_by_pid(features, labels, pids):
    """Group features and labels by unique participant IDs."""
    pids_npy = np.array(pids)
    unique_pids = np.unique(pids)
    pid_data = {}
    for pid in unique_pids:
        mask = (pids_npy == pid)
        pid_features = features[mask]
        pid_labels = labels[mask]
        pid_data[pid] = (pid_features, pid_labels)
    return pid_data

def prepare_data_for_local_models(data_splits, model_str, config):

    bs = config["batch_size"]
    sequence_length = config["sequence_length"]
    time_steps = config["time_steps"]

    # Group data from each split by participant ID
    train_groups = group_data_by_pid(
        data_splits['train']['feature'],
        data_splits['train']['labels'],
        data_splits['train']['participant_ids']
    )
    intra_groups = group_data_by_pid(
        data_splits['intra_subject_test']['feature'],
        data_splits['intra_subject_test']['labels'],
        data_splits['intra_subject_test']['participant_ids']
    )

    # Get all unique participant IDs across all splits
    all_pids = set(train_groups.keys()).union(intra_groups.keys())#.union(cross_groups.keys())
    my_gesture_dataset = select_dataset_class(model_str)

    #features, labels = cross_groups[pid]
    cross_dataset = my_gesture_dataset(
        data_splits['cross_subject_test']['feature'], data_splits['cross_subject_test']['labels'], sl=sequence_length, ts=time_steps
    )
    cross_loader = DataLoader(cross_dataset, batch_size=bs, shuffle=False)

    user_dict = {}
    for pid in all_pids:
        train_loader, intra_loader = None, None
        
        # Create train loader if participant exists in training split
        if pid in train_groups:
            features, labels = train_groups[pid]
            train_dataset = my_gesture_dataset(
                features, labels, sl=sequence_length, ts=time_steps
            )
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        # Create intra-subject test loader
        if pid in intra_groups:
            features, labels = intra_groups[pid]
            intra_dataset = my_gesture_dataset(
                features, labels, sl=sequence_length, ts=time_steps
            )
            intra_loader = DataLoader(intra_dataset, batch_size=bs, shuffle=False)
        # USE THE SAME CROSS LOADER FOR ALL USERS. NO SUBJECT SPECIFC CROSS SUBJECT (BY DEFINITION)
        user_dict[pid] = (train_loader, intra_loader, cross_loader)
    return user_dict
