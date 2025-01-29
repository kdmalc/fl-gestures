# OLD VERSION - NOT UP TO DATE. ARCHIVE THIS

import torch
import copy
import pandas as pd
import pickle
import numpy as np
from collections import defaultdict
from datetime import datetime
from sklearn.model_selection import ParameterGrid
import os
cwd = os.getcwd()
print("Current Working Directory: ", cwd)

from DNN_FT_funcs import *
from revamped_model_classes import *


NUM_CONFIGS = 20
MODEL_STR = "GenMomonaNet"
expdef_df = load_expdef_gestures(apply_hc_feateng=False)


# Define the search space
GenMomonaNet_architecture_space = {
    "num_channels": [NUM_CHANNELS],  #(int): Number of input channels.
    "sequence_length": [64],  #(int): Length of the input sequence.
    # ^ 32 wasn't working, it doesn't support time_steps to do multiple sequence batches for each slice
    "time_steps": [None],  # Required so that reused code doesn't break. Not used in GenMomonanet
    "conv_layers": [ #(list of tuples): List of tuples specifying convolutional layers. Each tuple should contain (out_channels, kernel_size, stride).
        # Configuration 1: Moderate Complexity
        [(32, 5, 1), (64, 3, 1), (128, 2, 1)],
        # Configuration 2: Increased Depth --> I think this would throw things off...
        [(16, 3, 1), (32, 3, 1), (64, 3, 1), (128, 2, 1)],
        # Configuration 3: Larger Filters for Broader Features
        [(32, 7, 1), (64, 5, 1), (128, 3, 1)],
        # Configuration 4: Smaller Stride for More Detailed Features
        [(32, 5, 1), (64, 3, 1), (128, 3, 1)],
        # Configuration 5: Balanced Approach
        [(32, 3, 1), (64, 3, 1), (128, 2, 1)]], 
    "pooling_layers": [[True, False, False, False], [True, True, True, True], [False, False, False, False], [True, True, False, False]],  # Max pooling only after the first conv layer
    "use_dense_cnn_lstm": [True, False],  # Use dense layer between CNN and LSTM
    "lstm_hidden_size": [8, 12, 16, 24],  #(int): Hidden size for LSTM layers.
    "lstm_num_layers": [1, 2, 3],  #(int): Number of LSTM layers.
    "fc_layers": [[64, 32], [128, 64], [32, 16]],  #(list of int): List of integers specifying the sizes of fully connected layers.
    "num_classes": [10] #(int): Number of output classes.
    #"use_batchnorm": [True, False]  # This is not added yet
}

GenMomonaNet_hyperparameter_space = {
    "batch_size": [SHARED_BS],  #(int): Batch size.
    "lstm_dropout": [0.8],  #(float): Dropout probability for LSTM layers.
    "learning_rate": [0.0001, 0.001, 0.01, 0.1],
    "num_epochs": [500], #[30, 50, 70],
    "optimizer": ["adam", "sgd"],
    "weight_decay": [0.0, 1e-4],
    "dropout_rate": [0.0, 0.3, 0.5],
    "ft_learning_rate": [0.0001, 0.001, 0.01, 0.1],
    "num_ft_epochs": [500], #[10, 30, 50],
    "ft_weight_decay": [0.0, 1e-4], 
    "use_earlystopping": [True]  # Always use this to save time, in ft and earlier training
}

# Hyperparameter tuning function
def hyperparam_tuning_for_ft(model_str, expdef_df, hyperparameter_space, architecture_space, 
                             num_configs_to_test=20, save_path="results/hyperparam_tuning",
                             num_datasplits_to_test=2, num_train_trials=8, num_ft_trials=3):
    # Generate all possible configurations
    ## Does this like shuffle or is this deterministic?
    configs = list(ParameterGrid({**hyperparameter_space, **architecture_space}))
    configs = configs[:num_configs_to_test]  # Limit the number of configurations to test

    # This creates the (multiple) train/test splits
    data_splits_lst = []
    for datasplit in range(num_datasplits_to_test):
        all_participants = expdef_df['Participant'].unique()
        # Shuffle the participants for train/test user split --> UNIQUE
        np.random.shuffle(all_participants)
        test_participants = all_participants[24:]  # 24 train / 8 test
        data_splits_lst.append(prepare_data(
            expdef_df, 'feature', 'Gesture_Encoded', 
            all_participants, test_participants, 
            training_trials_per_gesture=num_train_trials, 
            finetuning_trials_per_gesture=num_ft_trials,
        ))

    results = []
    for config_idx, config in enumerate(configs):
        print(f"Testing config {config_idx + 1}/{len(configs)}: {config}")

        for datasplit in data_splits_lst:
            # Train to-be-pretrained model
            results = main_training_pipeline(
                datasplit, 
                all_participants=all_participants, 
                test_participants=test_participants,
                model_type=model_str, 
                config=config)
            pretrained_model = results["model"]
            # Save now-pretrained model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            full_path = os.path.join(cwd, 'ELEC573_Proj', 'models', f'{timestamp}_pretrained_{model_str}_model.pth')
            print("Full Path:", full_path)
            torch.save(pretrained_model.state_dict(), full_path)

            # Evaluate the configuration across all users
            user_accuracies = []
            # Extract data for novel_trainFT and cross_subject_test
            ft_features = datasplit['novel_trainFT']['feature']
            ft_labels = datasplit['novel_trainFT']['labels']
            ft_pids = datasplit['novel_trainFT']['participant_ids']
            cross_features = datasplit['cross_subject_test']['feature']
            cross_labels = datasplit['cross_subject_test']['labels']
            cross_pids = datasplit['cross_subject_test']['participant_ids']
            # Group data by participant_ID for both datasets
            ft_user_data = defaultdict(lambda: ([], []))  # Tuple of lists: (features, labels)
            for feature, label, participant_id in zip(ft_features, ft_labels, ft_pids):
                ft_user_data[participant_id][0].append(feature)
                ft_user_data[participant_id][1].append(label)
            cross_user_data = defaultdict(lambda: ([], []))  # Tuple of lists: (features, labels)
            for feature, label, participant_id in zip(cross_features, cross_labels, cross_pids):
                cross_user_data[participant_id][0].append(feature)
                cross_user_data[participant_id][1].append(label)
            # Iterate through each unique participant_ID for both datasets simultaneously
            for pid in ft_user_data.keys() & cross_user_data.keys():  # Ensure we only iterate over common participant_IDs
                print(f"Fine-tuning on user {pid}")
                ft_features_agg, ft_labels_agg = ft_user_data[pid]
                cross_features_agg, cross_labels_agg = cross_user_data[pid]
                # Need loader_dim or model_str to know which dataset to use...
                my_gesture_dataset = select_dataset_class(model_str)
                # WE HAVE 30 TOTAL GESTURES IN THE FINETUNING DATASET (3 samples from 10 classe)
                ft_train_dataset = my_gesture_dataset(
                    ft_features_agg, ft_labels_agg, 
                    sl=config['sequence_length'], ts=config['time_steps'])
                cross_test_dataset = my_gesture_dataset(
                    cross_features_agg, cross_labels_agg, 
                    sl=config['sequence_length'], ts=config['time_steps'])

                # Conver the above into datasets and then into dataloaders
                fine_tune_loader = DataLoader(ft_train_dataset, batch_size=config['batch_size'], shuffle=True)
                cross_test_loader = DataLoader(cross_test_dataset, batch_size=config['batch_size'], shuffle=False)

                print("STOP POINT!")

                # Fine-tune the model on the current user's data
                finetuned_model, _, _, _ = fine_tune_model(
                    pretrained_model, fine_tune_loader, config, timestamp,
                    test_loader=cross_test_loader, pid=pid
                )

                # Evaluate the fine-tuned model on the user's test set
                metrics = evaluate_model(finetuned_model, cross_test_loader)
                # This should be novel user intra-subject accuracy ig
                user_accuracies.append(metrics["accuracy"])

            # Aggregate results across users
            ## TODO: Also need to aggregate results across the different datasplits...
            avg_accuracy = sum(user_accuracies) / len(user_accuracies)
            results.append({"config": config, "avg_accuracy": avg_accuracy, "user_accuracies": user_accuracies})

    # Sort results by average accuracy
    results = sorted(results, key=lambda x: x["avg_accuracy"], reverse=True)

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(save_path, exist_ok=True)
    df = pd.DataFrame([{"avg_accuracy": r["avg_accuracy"], **r["config"]} for r in results])
    df.to_csv(os.path.join(save_path, f"{timestamp}_hyperparameter_results.csv"), index=False)

    return results

# Example usage
#ft_user_train_test_tuples
results = hyperparam_tuning_for_ft(MODEL_STR, expdef_df, GenMomonaNet_hyperparameter_space, GenMomonaNet_architecture_space, num_configs_to_test=NUM_CONFIGS)
print("Top configurations:", results[:5])