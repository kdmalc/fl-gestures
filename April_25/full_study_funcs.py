from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

from moments_engr import *
from agglo_model_clust import *
from cluster_acc_viz_funcs import *
from DNN_FT_funcs import *
from DNN_AMC_funcs import *
#from model_classes import *
from revamped_model_classes import *


def full_comparison_run(finetuning_datasplits, cluster_assgnmt_data_splits, config, pretrained_generic_model,
                        nested_clus_model_dict, model_str, cluster_iter_str='Iter18'):
    
    if finetuning_datasplits != cluster_assgnmt_data_splits:
        print("Unique cluster_assgnmt_data_splits passed in. Not currently used in the code!")
    
    os.makedirs(config['results_save_dir'], exist_ok=True)

    # Wait who is this... are these the pretraining users?
    train_pids = np.unique(finetuning_datasplits['train']['participant_ids'])

    novel_participant_ft_data = finetuning_datasplits['novel_trainFT']
    # novel "cross subject" is the same as novel intra (but needs to be separated according to PID first...)
    novel_participant_test_data = finetuning_datasplits['cross_subject_test']
    novel_pids = np.unique(finetuning_datasplits['novel_trainFT']['participant_ids'])
    #novel_pid_clus_asgn_data = cluster_assgnmt_data_splits['novel_trainFT']

    novel_pid_res_dict = {}

    for pid_count, pid in enumerate(novel_pids):
        print(f"PID {pid}, {pid_count+1}/{len(novel_pids)}")
        novel_pid_res_dict[pid] = {}

        # Create the testloader by segmenting out this specific pid
        # Filter based on CURRENT participant ID: 
        indices = [i for i, datasplit_pid in enumerate(novel_participant_ft_data['participant_ids']) if datasplit_pid == pid]
        #indices = [i for i, datasplit_pid in enumerate(novel_participant_test_data['participant_ids']) if datasplit_pid == pid]
        # ^ These indices should be the same as the above indices I think...
        
        ############## Novel Participant Finetuning Dataset ##############
        ft_dataset = GestureDataset([novel_participant_ft_data['feature'][i] for i in indices], [novel_participant_ft_data['labels'][i] for i in indices])
        ft_loader = DataLoader(ft_dataset, batch_size=config["batch_size"], shuffle=True)
        ############## Novel Participant Intra Testing Dataset ##############
        intra_test_dataset = GestureDataset([novel_participant_test_data['feature'][i] for i in indices], [novel_participant_test_data['labels'][i] for i in indices])
        intra_test_loader = DataLoader(intra_test_dataset, batch_size=config["batch_size"], shuffle=True)
        ############## Cluster Assignment Dataset ##############
        ## This will just use ft_loader. No reason to have them separated really. 
        ## Removing this will remove functionality where num cluster assgnt != num ft trials
        #indices = [i for i, datasplit_pid in enumerate(novel_pid_clus_asgn_data['participant_ids']) if datasplit_pid == pid]
        #clust_asgn_dataset = GestureDataset([novel_pid_clus_asgn_data['feature'][i] for i in indices], [novel_pid_clus_asgn_data['labels'][i] for i in indices])
        #clust_asgn_loader = DataLoader(clust_asgn_dataset, batch_size=config["batch_size"], shuffle=True)

        ############## Novel Participant Cross Testing Dataset ##############
        # This code is testing on all the other novel participants... I don't think we care about that right now
        ## Idc but this will allow us to check cross perf. No real reason to remove...
        indices = [i for i, datasplit_pid in enumerate(novel_participant_test_data['participant_ids']) if datasplit_pid != pid]
        cross_test_dataset = GestureDataset([novel_participant_test_data['feature'][i] for i in indices], [novel_participant_test_data['labels'][i] for i in indices])
        cross_test_loader = DataLoader(cross_test_dataset, batch_size=config["batch_size"], shuffle=True)

        # 1) Train a local model
        ## MomonaNets do not need input_dim and num_classes, other models need to be updated to infer that or pull it from config
        local_res = main_training_pipeline(data_splits=None, all_participants=train_pids, test_participants=novel_pids, model_type=model_str, config=config, 
                           train_intra_cross_loaders=[ft_loader, intra_test_loader, cross_test_loader])
        # ^ This stuff was subject specific... so I'm not sure why performance was the same (RC) unless all models are severely over/under-trained
        local_clus_res = evaluate_model(local_res["model"], intra_test_loader)
        novel_pid_res_dict[pid]["local_acc"] = local_clus_res["accuracy"]
        # I do have the Local train/test curves in local_res! I will just save those as well!
        novel_pid_res_dict[pid]["local_train_loss_log"] = local_res["train_loss_log"]
        novel_pid_res_dict[pid]["local_intra_test_loss_log"] = local_res["intra_test_loss_log"]
        #novel_pid_res_dict[pid]["local_cross_test_loss_log"] = local_res["cross_test_loss_log"]

        # 2) Test the full pretrained (centralized) model
        generic_clus_res = evaluate_model(pretrained_generic_model, intra_test_loader)
        novel_pid_res_dict[pid]["centralized_acc"] = generic_clus_res["accuracy"]

        # 3) Test finetuned pretrained (centralized) model
        ft_centralized_model, _, ft_centralized_train_loss_log, ft_centralized_test_loss_log = fine_tune_model(
            pretrained_generic_model, ft_loader, config, config['timestamp'], test_loader=intra_test_loader, pid=pid)
        ft_centralized_res = evaluate_model(ft_centralized_model, intra_test_loader)
        novel_pid_res_dict[pid]["ft_centralized_acc"] = ft_centralized_res["accuracy"]
        novel_pid_res_dict[pid]["ft_centralized_train_loss_log"] = ft_centralized_train_loss_log
        novel_pid_res_dict[pid]["ft_centralized_test_loss_log"] = ft_centralized_test_loss_log

        # 4) CLUSTER MODEL: Have the pretrained model from the best cluster do inference
        #   - Have all cluster models do inference and compare assign to whichever cluster gives best results
        # Apply all the cluster models at the chosen iteration on the given participant data
        ## Record that cluster's performance (all cluster's performances?... Ideally)
        clus_model_res_dict = {}
        cluster_lst = list(nested_clus_model_dict[cluster_iter_str].keys())
        for clus_id in cluster_lst:
            clus_model = nested_clus_model_dict[cluster_iter_str][clus_id]
            clus_res = evaluate_model(clus_model, ft_loader)  # clust_asgn_loader) 
            clus_acc = clus_res["accuracy"]
            clus_model_res_dict[clus_id] = clus_acc
        # Assign participant to highest scoring cluster
        # Find the key with the highest accuracy
        max_key = max(clus_model_res_dict, key=clus_model_res_dict.get)
        max_value = clus_model_res_dict[max_key]
        # Is this saved anywhere or just printed? Presumably just printed...
        print(f"Cluster {max_key} had the highest accuracy ({max_value})")
        if config['verbose']:
            print("Full cluster assignment results dict:")
            print(clus_model_res_dict)
        original_cluster_model = nested_clus_model_dict[cluster_iter_str][max_key]
        # Have the pretrained model from the best cluster do inference
        pretrained_clus_res = evaluate_model(original_cluster_model, intra_test_loader)
        novel_pid_res_dict[pid]["pretrained_cluster_acc"] = pretrained_clus_res["accuracy"]

        # 5) FT the pretrained cluster model on the participant
        ft_cluster_model, original_cluster_model, ft_cluster_train_loss_log, ft_cluster_test_loss_log = fine_tune_model(
            original_cluster_model, ft_loader, config, config['timestamp'], test_loader=intra_test_loader, pid=pid)
        ft_clus_res = evaluate_model(ft_cluster_model, intra_test_loader)
        novel_pid_res_dict[pid]["ft_cluster_acc"] = ft_clus_res["accuracy"]
        novel_pid_res_dict[pid]["ft_cluster_train_loss_log"] = ft_cluster_train_loss_log
        novel_pid_res_dict[pid]["ft_cluster_test_loss_log"] = ft_cluster_test_loss_log

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


def plot_model_acc_boxplots(data_dict, model_str=None, colors_lst=None, my_title=None, save_fig=False, plot_save_name=None, 
                            save_dir="C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results",
                            data_keys=['centralized_acc_data', 'pretrained_cluster_acc_data', 'ft_centralized_acc_data', 'local_acc_data', 'ft_cluster_acc_data'],
                            labels=['Generic Centralized', 'Pretrained Cluster', 'Fine-Tuned Centralized', 'Local', 'Fine-Tuned Cluster']):
    
    data = [data_dict[key] for key in data_keys]

    # Generate default colors if not provided
    if colors_lst is None:
        colors_lst = ['lightgray'] * len(data_keys)  # Subtle box colors

    # Assign unique colors and markers per user (assuming user index)
    num_users = max(len(d) for d in data)
    user_colors = plt.cm.get_cmap("tab10", num_users)  # Distinct colors
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', '<', '>']  # Unique markers

    if my_title is None and model_str is not None:
        my_title = f"{model_str} Per-User Accuracies"

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create boxplots
    bp = ax.boxplot(data, patch_artist=True, labels=labels)

    # Customize box colors
    for patch, color in zip(bp['boxes'], colors_lst):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)  # Slight transparency

    # Overlay scatter points per user with consistent colors and markers
    for i, acc_list in enumerate(data):
        for user_idx, acc in enumerate(acc_list):
            marker = markers[user_idx % len(markers)]  # Cycle through markers
            ax.scatter(i + 1, acc, color=user_colors(user_idx), edgecolor='black', s=50, marker=marker, alpha=0.8)

    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Increase font sizes
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title(my_title, fontsize=20)
    ax.tick_params(axis='both', labelsize=12)

    plt.xticks(rotation=30, ha="right")
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


def prepare_data_for_local_models(data_splits, config):

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
    my_gesture_dataset = select_dataset_class(config)

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


def finetuning_comparison_run(finetuning_datasplits, config, pretrained_generic_model,
                        nested_clus_model_dict, cluster_iter_str='Iter18'):
    
    #os.makedirs(config['results_save_dir'], exist_ok=True)

    novel_participant_ft_data = finetuning_datasplits['novel_trainFT']
    # novel "cross subject" is the same as novel intra (but needs to be separated according to PID first...)
    novel_participant_test_data = finetuning_datasplits['cross_subject_test']
    novel_pids = np.unique(finetuning_datasplits['novel_trainFT']['participant_ids'])
    #novel_pid_clus_asgn_data = cluster_assgnmt_data_splits['novel_trainFT']

    novel_pid_res_dict = {}

    for pid_count, pid in enumerate(novel_pids):
        print(f"PID {pid}, {pid_count+1}/{len(novel_pids)}")
        novel_pid_res_dict[pid] = {}

        # Create the testloader by segmenting out this specific pid
        # Filter based on CURRENT participant ID: 
        indices = [i for i, datasplit_pid in enumerate(novel_participant_ft_data['participant_ids']) if datasplit_pid == pid]
        #indices = [i for i, datasplit_pid in enumerate(novel_participant_test_data['participant_ids']) if datasplit_pid == pid]
        # ^ These indices should be the same as the above indices I think...
        
        ############## Novel Participant Finetuning Dataset ##############
        ft_dataset = GestureDataset([novel_participant_ft_data['feature'][i] for i in indices], [novel_participant_ft_data['labels'][i] for i in indices])
        ft_loader = DataLoader(ft_dataset, batch_size=config["batch_size"], shuffle=True)
        ############## Novel Participant Intra Testing Dataset ##############
        intra_test_dataset = GestureDataset([novel_participant_test_data['feature'][i] for i in indices], [novel_participant_test_data['labels'][i] for i in indices])
        intra_test_loader = DataLoader(intra_test_dataset, batch_size=config["batch_size"], shuffle=True)
        ############## Cluster Assignment Dataset ##############
        ## This will just use ft_loader. No reason to have them separated really. 
        ## Removing this will remove functionality where num cluster assgnt != num ft trials
        #indices = [i for i, datasplit_pid in enumerate(novel_pid_clus_asgn_data['participant_ids']) if datasplit_pid == pid]
        #clust_asgn_dataset = GestureDataset([novel_pid_clus_asgn_data['feature'][i] for i in indices], [novel_pid_clus_asgn_data['labels'][i] for i in indices])
        #clust_asgn_loader = DataLoader(clust_asgn_dataset, batch_size=config["batch_size"], shuffle=True)

        ############## Novel Participant Cross Testing Dataset ##############
        # This code is testing on all the other novel participants... I don't think we care about that right now
        ## Idc but this will allow us to check cross perf. No real reason to remove...
        #indices = [i for i, datasplit_pid in enumerate(novel_participant_test_data['participant_ids']) if datasplit_pid != pid]
        #cross_test_dataset = GestureDataset([novel_participant_test_data['feature'][i] for i in indices], [novel_participant_test_data['labels'][i] for i in indices])
        #cross_test_loader = DataLoader(cross_test_dataset, batch_size=config["batch_size"], shuffle=True)

        # 2) Test the full pretrained (centralized) model
        generic_clus_res = evaluate_model(pretrained_generic_model, intra_test_loader)
        novel_pid_res_dict[pid]["centralized_acc"] = generic_clus_res["accuracy"]

        # 3) Test finetuned pretrained (centralized) model
        ft_centralized_model, _, ft_centralized_train_loss_log, ft_centralized_test_loss_log = fine_tune_model(
            pretrained_generic_model, ft_loader, config, config['timestamp'], test_loader=intra_test_loader, pid=pid)
        ft_centralized_res = evaluate_model(ft_centralized_model, intra_test_loader)
        novel_pid_res_dict[pid]["ft_centralized_acc"] = ft_centralized_res["accuracy"]
        novel_pid_res_dict[pid]["ft_centralized_train_loss_log"] = ft_centralized_train_loss_log
        novel_pid_res_dict[pid]["ft_centralized_test_loss_log"] = ft_centralized_test_loss_log

        # 4) CLUSTER MODEL: Have the pretrained model from the best cluster do inference
        #   - Have all cluster models do inference and compare assign to whichever cluster gives best results
        # Apply all the cluster models at the chosen iteration on the given participant data
        ## Record that cluster's performance (all cluster's performances?... Ideally)
        clus_model_res_dict = {}
        cluster_lst = list(nested_clus_model_dict[cluster_iter_str].keys())
        for clus_id in cluster_lst:
            clus_model = nested_clus_model_dict[cluster_iter_str][clus_id]
            clus_res = evaluate_model(clus_model, ft_loader)  # clust_asgn_loader) 
            clus_acc = clus_res["accuracy"]
            clus_model_res_dict[clus_id] = clus_acc
        # Assign participant to highest scoring cluster
        # Find the key with the highest accuracy
        max_key = max(clus_model_res_dict, key=clus_model_res_dict.get)
        max_value = clus_model_res_dict[max_key]
        # Is this saved anywhere or just printed? Presumably just printed...
        print(f"Cluster {max_key} had the highest accuracy ({max_value})")
        if config['verbose']:
            print("Full cluster assignment results dict:")
            print(clus_model_res_dict)
        original_cluster_model = nested_clus_model_dict[cluster_iter_str][max_key]
        # Have the pretrained model from the best cluster do inference
        pretrained_clus_res = evaluate_model(original_cluster_model, intra_test_loader)
        novel_pid_res_dict[pid]["pretrained_cluster_acc"] = pretrained_clus_res["accuracy"]

        # 5) FT the pretrained cluster model on the participant
        ft_cluster_model, original_cluster_model, ft_cluster_train_loss_log, ft_cluster_test_loss_log = fine_tune_model(
            original_cluster_model, ft_loader, config, config['timestamp'], test_loader=intra_test_loader, pid=pid)
        ft_clus_res = evaluate_model(ft_cluster_model, intra_test_loader)
        novel_pid_res_dict[pid]["ft_cluster_acc"] = ft_clus_res["accuracy"]
        novel_pid_res_dict[pid]["ft_cluster_train_loss_log"] = ft_cluster_train_loss_log
        novel_pid_res_dict[pid]["ft_cluster_test_loss_log"] = ft_cluster_test_loss_log
    full_data_output_dict = novel_pid_res_dict

    centralized_acc_res = [] 
    ft_centralized_acc_res = []
    pretrained_cluster_acc_res = []
    ft_cluster_acc_res = [] 
    # Collecting data from the dictionary 
    for pid in novel_pid_res_dict: 
        centralized_acc_res.append(novel_pid_res_dict[pid]['centralized_acc']) 
        ft_centralized_acc_res.append(novel_pid_res_dict[pid]['ft_centralized_acc']) 
        pretrained_cluster_acc_res.append(novel_pid_res_dict[pid]['pretrained_cluster_acc']) 
        ft_cluster_acc_res.append(novel_pid_res_dict[pid]['ft_cluster_acc']) 

    final_user_res_dict = {'centralized_acc_data':centralized_acc_res, 'ft_centralized_acc_data':ft_centralized_acc_res, 'pretrained_cluster_acc_data':pretrained_cluster_acc_res, 'ft_cluster_acc_data':ft_cluster_acc_res}
    return final_user_res_dict, full_data_output_dict


def create_shared_trial_data_splits(expdef_df, all_participants, test_participants, num_monte_carlo_runs=5, num_train_gesture_trials=8, num_ft_gesture_trials=1):
    trial_data_splits_lst = [0]*num_monte_carlo_runs
    for i in range(num_monte_carlo_runs):
        # Prepare data
        trial_data_splits = prepare_data(
            expdef_df, 'feature', 'Gesture_Encoded', 
            all_participants, test_participants, 
            training_trials_per_gesture=num_train_gesture_trials, finetuning_trials_per_gesture=num_ft_gesture_trials,
        )
        trial_data_splits_lst[i] = trial_data_splits
    return trial_data_splits_lst


def finetuning_run(trial_data_splits_lst, config, pretrained_generic_model, nested_clus_model_dict):
    """This function de facto does MonteCarlo averaging, and the number of elements (separate datasplits) in trial_data_splits_lst determines how many repetitions there are. """
    num_monte_carlo_runs = len(trial_data_splits_lst)
    num_ft_users = config["num_testft_users"]
    
    lst_of_res_dicts = [0] * num_monte_carlo_runs
    train_test_logs_list = []  # Store train/test logs separately
    
    for i in range(num_monte_carlo_runs):
        res, logs = finetuning_comparison_run(trial_data_splits_lst[i], config, pretrained_generic_model,
                                              nested_clus_model_dict, cluster_iter_str='Iter18')
        lst_of_res_dicts[i] = res
        train_test_logs_list.append(logs)  # Save logs without averaging
    
    data_dict = {}    
    data_dict['centralized_acc_data'] = np.zeros(num_ft_users, dtype=np.float32)
    data_dict['ft_centralized_acc_data'] = np.zeros(num_ft_users, dtype=np.float32)
    data_dict['pretrained_cluster_acc_data'] = np.zeros(num_ft_users, dtype=np.float32)
    data_dict['ft_cluster_acc_data'] = np.zeros(num_ft_users, dtype=np.float32)

    for idx, res_dict in enumerate(lst_of_res_dicts):
        data_dict['centralized_acc_data'] += np.array(res_dict['centralized_acc_data'])
        data_dict['ft_centralized_acc_data'] += np.array(res_dict['ft_centralized_acc_data'])
        data_dict['pretrained_cluster_acc_data'] += np.array(res_dict['pretrained_cluster_acc_data'])
        data_dict['ft_cluster_acc_data'] += np.array(res_dict['ft_cluster_acc_data'])

    data_dict['centralized_acc_data'] /= num_monte_carlo_runs
    data_dict['ft_centralized_acc_data'] /= num_monte_carlo_runs
    data_dict['pretrained_cluster_acc_data'] /= num_monte_carlo_runs
    data_dict['ft_cluster_acc_data'] /= num_monte_carlo_runs

    # Return both averaged results and raw logs
    data_dict['train_test_logs_list'] = train_test_logs_list
    return data_dict
