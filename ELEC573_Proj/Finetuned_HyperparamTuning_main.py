import torch
import copy
import pandas as pd
import pickle
from datetime import datetime
from sklearn.model_selection import ParameterGrid
import os
cwd = os.getcwd()
print("Current Working Directory: ", cwd)

from DNN_FT_funcs import *
from revamped_model_classes import *


NUM_CONFIGS = 20
MODEL_STR = "GenMomonaNet"


# Define the search space
## Are these up to date with MomonaNet?
## If so, how does num_conv_layers work if it can be 2 or 3 but all the conv layer sizes are 3? 
## Just doesnt use the last one?
## TODO: I think I need to splice these together for GenMomonaNet?
### configs = list(ParameterGrid({**hyperparameter_space, **architecture_space}))
### ^ Creates a list of config? Merges the two... or??
### LSTM not even present
## Add maxpool options? Kernel(?) 1 and 2 (mostly 2)
GenMomonaNet_architecture_space = {
    "num_channels": NUM_CHANNELS,  #(int): Number of input channels.
    "sequence_length": 64,  #(int): Length of the input sequence.
    "conv_layers": [(32, 5, 1), (64, 3, 1), (128, 2, 1)],  #(list of tuples): List of tuples specifying convolutional layers. Each tuple should contain (out_channels, kernel_size, stride).
    "pooling_layers": [True, False, False],  # Max pooling only after the first conv layer
    "use_dense_cnn_lstm": True,  # Use dense layer between CNN and LSTM
    "lstm_hidden_size": 12,  #(int): Hidden size for LSTM layers.
    "lstm_num_layers": 2,  #(int): Number of LSTM layers.
    "fc_layers": [64, 32],  #(list of int): List of integers specifying the sizes of fully connected layers.
    "num_classes": 10 #(int): Number of output classes.
    #"use_batchnorm": [True, False]  # This is not added yet
}

GenMomonaNet_hyperparameter_space = {
    "batch_size": SHARED_BS,  #(int): Batch size.
    "lstm_dropout": 0.8,  #(float): Dropout probability for LSTM layers.
    # NONE OF THESE ARE INPUTS TO GENMOMONANETCONFIG
    ## Should they be? Or should config only hold architecture? I think it should hold both
    "learning_rate": [0.0001, 0.001, 0.01, 0.1],
    "batch_size": [32, 64, 128],
    "num_epochs": [30, 50, 70],
    "optimizer": ["adam", "sgd"],
    "weight_decay": [0.0, 1e-4],
    "dropout_rate": [0.0, 0.3, 0.5]
}

##########################################################################
##########################################################################
##########################################################################

expdef_df = load_expdef_gestures(apply_hc_feateng=False)

all_participants = expdef_df['Participant'].unique()
# Shuffle the participants for train/test user split --> UNIQUE
np.random.shuffle(all_participants)
# Split into two groups
#train_participants = all_participants[:24]  # First 24 participants
test_participants = all_participants[24:]  # Remaining 8 participants

# Prepare data
data_splits = prepare_data(
    expdef_df, 'feature', 'Gesture_Encoded', 
    all_participants, test_participants, 
    training_trials_per_gesture=8, finetuning_trials_per_gesture=3,
)

# TODO: Need to create train and test loader somewhere...
## Guess that should happen... inside the hyperparam tuning? Depends what cross val it is doing ig

##########################################################################
##########################################################################
##########################################################################


# Hyperparameter tuning function
def hyperparam_tuning_for_ft(model_str, users_data, hyperparameter_space, architecture_space, num_configs_to_test=20, save_path="results/hyperparam_tuning"):
    # Generate all possible configurations
    ## Does this like shuffle or is this deterministic?
    configs = list(ParameterGrid({**hyperparameter_space, **architecture_space}))
    configs = configs[:num_configs_to_test]  # Limit the number of configurations to test

    results = []

    for config_idx, config in enumerate(configs):
        print(f"Testing config {config_idx + 1}/{len(configs)}: {config}")

        # Do dataset/novel split
        ## DO NOT LOAD THE DATASET IN OVER AND OVER AGAIN
        ## All users should be passed in...
        ## Then do intra/cross/novel tts here...
        # ...
        # Initialize the model with the current configuration
        model, loader_dim = model_selection(model_str)
        # Train generic model
        results = main_training_pipeline(
        data_splits, 
        all_participants=all_participants, 
        test_participants=test_participants,
        model_type=model_str, 
        config=config)
        # Save generic model
        ## TODO: Add a timestamp to the save so it doesn't overwrite every time...
        full_path = os.path.join(cwd, 'ELEC573_Proj', 'models', f'generic_{model_str}_model.pth')
        print("Full Path:", full_path)
        torch.save(results["model"].state_dict(), full_path)

        # Evaluate the configuration across all users
        user_accuracies = []
        for user_id, (train_dataset, test_loader) in users_data.items():
            print(f"Fine-tuning on user {user_id}...")

            # Fine-tune the model on the current user's data
            fine_tune_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
            finetuned_model, _, _, _ = fine_tune_model(
                model, fine_tune_loader, test_loader, num_epochs=config["num_epochs"], lr=config["learning_rate"],
                use_weight_decay=config["weight_decay"] > 0, weight_decay=config["weight_decay"]
            )

            # Evaluate the fine-tuned model on the user's test set
            metrics = evaluate_model(finetuned_model, test_loader)
            # This should be novel user intra-subject accuracy ig
            user_accuracies.append(metrics["accuracy"])

        # Aggregate results across users
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
results = hyperparam_tuning_for_ft(MODEL_STR, ft_user_train_test_tuples, GenMomonaNet_hyperparameter_space, GenMomonaNet_architecture_space, num_configs_to_test=NUM_CONFIGS)
print("Top configurations:", results[:5])