import torch
from torch.utils.data import DataLoader
#import copy
#import pandas as pd
#import pickle
import random
import json
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
timestamp = datetime.now().strftime("%Y%m%d_%H%M")


# Define the search space
## Is anything about these configs specific to GenMomonaNet? Or can I use the same one for each model?
GenMomonaNet_architecture_space = {
    "num_channels": [NUM_CHANNELS],  #(int): Number of input channels.
    "sequence_length": [64],  #(int): Length of the input sequence.
    # ^ 32 wasn't working, it doesn't support time_steps to do multiple sequence batches for each slice
    "time_steps": [None],  # Required so that reused code doesn't break. Not used in GenMomonanet
    "conv_layers": [ #(list of tuples): List of tuples specifying convolutional layers. Each tuple should contain (out_channels, kernel_size, stride).
        # Configuration 1: Moderate Complexity
        [(32, 5, 1), (64, 3, 1), (128, 2, 1)],
        # Configuration 2: Increased Depth --> I think this would throw things off...
        #[(16, 3, 1), (32, 3, 1), (64, 3, 1), (128, 2, 1)],
        # Configuration 3: Larger Filters for Broader Features
        [(32, 7, 1), (64, 5, 1), (128, 3, 1)],
        # Configuration 4: Smaller Stride for More Detailed Features
        #[(32, 5, 1), (64, 3, 1), (128, 3, 1)],
        # Configuration 5: Balanced Approach
        #[(32, 3, 1), (64, 3, 1), (128, 2, 1)]
        ], 
    "pooling_layers": [[True, False, False, False], 
                       #[True, True, True, True], 
                       [False, False, False, False], 
                       [True, True, False, False]
                       ],  # Max pooling only after the first conv layer
    "use_dense_cnn_lstm": [True, False],  # Use dense layer between CNN and LSTM
    "lstm_hidden_size": [8, 16, 24],  #(int): Hidden size for LSTM layers.
    "lstm_num_layers": [1, 2],  #(int): Number of LSTM layers.
    "fc_layers": [[64, 32], [128, 64]],  #(list of int): List of integers specifying the sizes of fully connected layers.
    "num_classes": [10] #(int): Number of output classes.
    #"use_batchnorm": [True, False]  # This is not added yet
}

GenMomonaNet_hyperparameter_space = {
    "batch_size": [16, 32, 64, 128],  #SHARED_BS #(int): Batch size.
    "lstm_dropout": [0.0, 0.5, 0.8],  #(float): Dropout probability for LSTM layers.
    "learning_rate": [0.001, 0.01, 0.1],
    "num_epochs": [500], #[30, 50, 70],
    "optimizer": ["adam", "sgd"],
    "weight_decay": [0.0, 1e-4],
    "dropout_rate": [0.0, 0.3, 0.5],
    "ft_learning_rate": [0.001, 0.01, 0.1],
    "num_ft_epochs": [500], #[10, 30, 50],
    "ft_weight_decay": [0.0, 1e-4], 
    "ft_batch_size": [1, 5, 10 ], 
    "use_earlystopping": [True],  # Always use this to save time, in ft and earlier training
    "results_save_dir": [f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results\\hyperparam_tuning\\{timestamp}"],  # TODO: Implement this
    "timestamp": [timestamp],  # TODO: Implement this
    "verbose": [True]  # TODO: Implement this
    #"models_save_dir": ["C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\models"]  # I might not use this since it is already working fine
}

def save_results(results, save_dir, timestamp):
    """Save the results to a JSON file, sorted by overall average accuracy."""
    # Sort results by overall average accuracy in descending order
    sorted_results = sorted(results, key=lambda x: x["overall_avg_accuracy"], reverse=True)

    # Add a note about the best configuration
    if sorted_results:
        best_config = sorted_results[0]["config"]
        best_accuracy = sorted_results[0]["overall_avg_accuracy"]
        sorted_results.insert(0, {
            "note": f"Best configuration: {best_config} with overall average accuracy: {best_accuracy:.4f}"
        })

    # Save to JSON file
    results_path = os.path.join(save_dir, f'{timestamp}_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)  # Ensure the directory exists
    with open(results_path, 'w') as f:
        json.dump(sorted_results, f, indent=4)

    print(f"Results saved to: {results_path}")

def group_data_by_participant(features, labels, pids):
    """Helper function to group features and labels by participant ID."""
    user_data = defaultdict(lambda: ([], []))  # Tuple of lists: (features, labels)
    for feature, label, pid in zip(features, labels, pids):
        user_data[pid][0].append(feature)
        user_data[pid][1].append(label)
    return user_data

def evaluate_configuration(datasplit, pretrained_model, config, model_str, timestamp):
    """Evaluate a configuration on a given data split."""
    user_accuracies = []

    # Group data by participant ID
    ft_user_data = group_data_by_participant(
        datasplit['novel_trainFT']['feature'],
        datasplit['novel_trainFT']['labels'],
        datasplit['novel_trainFT']['participant_ids']
    )
    cross_user_data = group_data_by_participant(
        datasplit['cross_subject_test']['feature'],
        datasplit['cross_subject_test']['labels'],
        datasplit['cross_subject_test']['participant_ids']
    )

    # Iterate through each unique participant ID
    for pid in ft_user_data.keys() & cross_user_data.keys():  # Only common participant IDs
        print(f"Fine-tuning on user {pid}")

        # Prepare datasets
        my_gesture_dataset = select_dataset_class(model_str)
        ft_train_dataset = my_gesture_dataset(
            *ft_user_data[pid], sl=config['sequence_length'], ts=config['time_steps']
        )
        cross_test_dataset = my_gesture_dataset(
            *cross_user_data[pid], sl=config['sequence_length'], ts=config['time_steps']
        )

        # Create dataloaders
        fine_tune_loader = DataLoader(ft_train_dataset, batch_size=config['batch_size'], shuffle=True)
        cross_test_loader = DataLoader(cross_test_dataset, batch_size=config['batch_size'], shuffle=False)

        # Fine-tune and evaluate the model
        finetuned_model, _, _, _ = fine_tune_model(
            pretrained_model, fine_tune_loader, config, timestamp, 
            test_loader=cross_test_loader,
            pid=pid, use_earlystopping=config["use_earlystopping"]
        )
        metrics = evaluate_model(finetuned_model, cross_test_loader)
        user_accuracies.append(metrics["accuracy"])

    return user_accuracies

def save_model(model, model_str, save_dir, model_scenario_str, verbose=True, timestamp=None):
    """Save the model with a timestamp."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    full_path = os.path.join(save_dir, f'{timestamp}_{model_scenario_str}_{model_str}_model.pth')
    if verbose:
        print("Full Path:", full_path)
    torch.save(model.state_dict(), full_path)

# main function
def hyperparam_tuning_for_ft(model_str, expdef_df, hyperparameter_space, architecture_space, 
                             num_configs_to_test=20, 
                             save_path="C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results\\hyperparam_tuning",
                             num_datasplits_to_test=2, num_train_trials=8, num_ft_trials=3):
    
    # Generate all possible configurations
    ## Does this like shuffle or is this deterministic?
    print("Combining configs")
    configs = list(ParameterGrid({**hyperparameter_space, **architecture_space}))
    # Shuffle the configurations
    random.shuffle(configs)
    configs = configs[:num_configs_to_test]  # Limit the number of configurations to test
    # Random search variant
    #from sklearn.model_selection import ParameterSampler
    #configs = list(ParameterSampler({**hyperparameter_space, **architecture_space}, n_iter=num_configs_to_test))

    # This creates the (multiple) train/test splits
    print("Creating datasplits")
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

        split_results = []
        for datasplit in data_splits_lst:
            # Here, each config+datasplit pair gets its own timestamp... not sure what is optimal
            ## Dont want anything to overwrite mainly
            ## TODO: Augment saving to create and save to folders (folders are timestamps?)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")

            # Train the model
            training_results = main_training_pipeline(
                datasplit, all_participants=all_participants, test_participants=test_participants,
                model_type=model_str, config=config
            )
            pretrained_model = training_results["model"]

            # Save the pretrained model
            save_model(pretrained_model, model_str, save_path, "pretrained", verbose=True)

            # Evaluate the configuration on the current data split
            user_accuracies = evaluate_configuration(datasplit, pretrained_model, config, model_str, timestamp)
            avg_accuracy = sum(user_accuracies) / len(user_accuracies)
            split_results.append({"avg_accuracy": avg_accuracy, "user_accuracies": user_accuracies})

        # Aggregate results across data splits
        overall_avg_accuracy = sum(split_result["avg_accuracy"] for split_result in split_results) / len(split_results)
        overall_user_accuracies = [acc for split_result in split_results for acc in split_result["user_accuracies"]]
        results.append({
            "config": config,
            "overall_avg_accuracy": overall_avg_accuracy,
            "overall_user_accuracies": overall_user_accuracies,
            "split_results": split_results
        })

    # Save the results
    save_results(results, save_path, timestamp)

    return results

# Run the main function
#results = main()
results = hyperparam_tuning_for_ft(MODEL_STR, expdef_df, GenMomonaNet_hyperparameter_space, GenMomonaNet_architecture_space, 
                             num_configs_to_test=2,
                             num_datasplits_to_test=2, num_train_trials=8, num_ft_trials=3)