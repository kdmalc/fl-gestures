import random
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from moments_engr import *
from DNN_FT_funcs import *
np.random.seed(42) 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

import pandas as pd
import seaborn as sns
import os
import sys
from datetime import datetime
import copy


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        torch.save(model.state_dict(), 'checkpoint.pt')
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')

# Usage example during training loop
'''
early_stopping = EarlyStopping(patience=10, verbose=True)

for epoch in range(1, epochs+1):
    # Train your model
    train(...)
    # Validate your model
    val_loss = validate(...)
    
    # Check early stopping condition
    early_stopping(val_loss, model)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

# Load the last checkpoint with the best model
model.load_state_dict(torch.load('checkpoint.pt'))
'''



# Generate random configurations
def sample_config(hp_space, arch_space, num_samples=10):
    configs = []
    for _ in range(num_samples):
        config = {}
        config.update({k: random.choice(v) for k, v in hp_space.items()})
        config.update({k: random.choice(v) for k, v in arch_space.items()})
        configs.append(config)
    return configs


# Evaluate a model given a specific configuration
def evaluate_config(config, train_dataset, intra_test_loader, cross_test_loader):
    
    # Create model
    ## THIS ONE ISNT WORKING, THE ACTUAL SIZE IS ALWAYS SMALLER THAN THE CALCULATED (input*seq) SIZE
    ## Because input_dims was set to 100 instead of 80? Test later...
    class DynamicCNNModel(nn.Module):
        def __init__(self, config, input_dim, num_classes):
            super(DynamicCNNModel, self).__init__()
            # This is redundant somewhat, but saving so I can access while debugging
            self.input_dim = input_dim
            self.num_classes = num_classes
            self.config = config

            layers = []
            in_channels = 1
            seq_len = input_dim

            for i in range(config["num_conv_layers"]):
                out_channels = config["conv_layer_sizes"][i]
                layers.append(nn.Conv1d(in_channels, out_channels, 
                                        kernel_size=config["kernel_size"], 
                                        stride=config["stride"], 
                                        padding=config["padding"]))
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool1d(config["maxpool"]))
                torch_seq_len = torch.floor(torch.tensor((torch.floor(torch.tensor((seq_len + 2*config["padding"] - config["kernel_size"]) / config["stride"])) + 1) / config["maxpool"]))
                seq_len = (seq_len + 2 * config["padding"] - config["kernel_size"]) // config["stride"] + 1
                seq_len = seq_len // config["maxpool"]
                print(f"Conv layer {i}, in_channels: {in_channels}, out_channels: {out_channels}, seq_len: {seq_len}, torch_seq_len: {torch_seq_len}")
                in_channels = out_channels

            # Also saving for access during debugging:
            #self.seq_len = int(seq_len.item())
            self.seq_len = seq_len
            self.in_channels = in_channels

            # NOTE: These only create, not apply, the following funcs!
            ## They get applied in forward
            self.conv = nn.Sequential(*layers)
            self.fc1 = nn.Linear(self.in_channels * self.seq_len, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.dropout = nn.Dropout(config["dropout_rate"])
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)
        
        def forward(self, x):
            #print(f"Input shape: {x.shape}")
            x = x.unsqueeze(1)
            #print(f"After unsqueeze: {x.shape}")
            x = self.conv(x)
            #print(f"After conv: {x.shape}")
            x = x.view(x.size(0), -1)  # Flatten for FC layers
            #print(f"After flatten: {x.shape}")
            x = self.fc1(x)
            #print(f"After fc1: {x.shape}")
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            #print(f"After fc2: {x.shape}")
            x = self.softmax(x)
            return x
        
    class CNNModel_2layer(nn.Module):
        def __init__(self, config, input_dim, num_classes):
            super(CNNModel_2layer, self).__init__()
            self.input_dim = input_dim
            self.num_classes = num_classes
            self.config = config

            # Activation and Pooling
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool1d(config["maxpool"])
            self.softmax = nn.Softmax(dim=1)
            
            # Convolutional Layers
            self.conv1 = nn.Conv1d(1, config["conv_layer_sizes"][0], kernel_size=config["kernel_size"], stride=config["stride"], padding=config["padding"])
            self.bn1 = nn.BatchNorm1d(config["conv_layer_sizes"][0]) if config["use_batchnorm"] else nn.Identity()
            self.conv2 = nn.Conv1d(config["conv_layer_sizes"][0], config["conv_layer_sizes"][1], kernel_size=config["kernel_size"], stride=config["stride"], padding=config["padding"])
            self.bn2 = nn.BatchNorm1d(config["conv_layer_sizes"][1]) if config["use_batchnorm"] else nn.Identity()
            
            # Dynamically calculate flattened size
            test_input = torch.randn(1, 1, input_dim)
            # Run through conv layers to calculate final size
            with torch.no_grad():
                test_x = self.conv1(test_input)
                test_x = self.maxpool(test_x)
                test_x = self.conv2(test_x)
                #test_x = self.maxpool(test_x)
                flattened_size = test_x.view(1, -1).size(1)
                #print(f"flattened_size of test input: {flattened_size}")
            
            # Fully Connected Layers
            self.fc1 = nn.Linear(flattened_size, 128)
            self.dropout = nn.Dropout(config["dropout_rate"])
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            #print(f"Input x shape: {x.shape}")
            x = x.unsqueeze(1)  # Reshape input to (batch_size, 1, sequence_length)
            #print(f"After unsqueeze: {x.shape}")
            # Conv Block 1
            x = self.conv1(x)
            #print(f"After conv1: {x.shape}")
            x = self.bn1(x)
            #print(f"After bn1: {x.shape}")
            x = self.relu(x)
            #print(f"After relu: {x.shape}")
            x = self.maxpool(x)
            #print(f"After maxpool: {x.shape}")
            # Conv Block 2
            x = self.conv2(x)
            #print(f"After conv2: {x.shape}")
            x = self.bn2(x)
            #print(f"After bn2: {x.shape}")
            x = self.relu(x)
            #print(f"After relu: {x.shape}")
            #x = self.maxpool(x)
            # Flatten and Fully Connected Layers
            x = x.view(x.size(0), -1)
            #print(f"After flattening for FC: {x.shape}")
            x = self.fc1(x)
            #print(f"After fc1: {x.shape}")
            x = self.relu(x)
            #print(f"After relu: {x.shape}")
            x = self.dropout(x)
            #print(f"After dropout: {x.shape}")
            x = self.fc2(x)
            #print(f"After fc2: {x.shape}")
            x = self.softmax(x)
            #print(f"After softmax: {x.shape}")
            return x
    
    class CNNModel_3layer(nn.Module):
        def __init__(self, config, input_dim, num_classes):
            super(CNNModel_3layer, self).__init__()
            self.input_dim = input_dim
            self.num_classes = num_classes
            self.config = config

            # Activation and Pooling
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool1d(config["maxpool"])
            self.softmax = nn.Softmax(dim=1)
            
            # Convolutional Layers
            self.conv1 = nn.Conv1d(1, config["conv_layer_sizes"][0], 
                                kernel_size=config["kernel_size"], 
                                stride=config["stride"], 
                                padding=config["padding"])
            self.bn1 = nn.BatchNorm1d(config["conv_layer_sizes"][0]) if config["use_batchnorm"] else nn.Identity()
            
            self.conv2 = nn.Conv1d(config["conv_layer_sizes"][0], config["conv_layer_sizes"][1], 
                                kernel_size=config["kernel_size"], 
                                stride=config["stride"], 
                                padding=config["padding"])
            self.bn2 = nn.BatchNorm1d(config["conv_layer_sizes"][1]) if config["use_batchnorm"] else nn.Identity()
            
            self.conv3 = nn.Conv1d(config["conv_layer_sizes"][1], config["conv_layer_sizes"][2], 
                                kernel_size=config["kernel_size"], 
                                stride=config["stride"], 
                                padding=config["padding"])
            self.bn3 = nn.BatchNorm1d(config["conv_layer_sizes"][2]) if config["use_batchnorm"] else nn.Identity()
            
            # Dynamically calculate flattened size
            test_input = torch.randn(1, 1, input_dim)
            # Run through conv layers to calculate final size
            with torch.no_grad():
                test_x = self.conv1(test_input)
                test_x = self.maxpool(test_x)
                test_x = self.conv2(test_x)
                test_x = self.maxpool(test_x)
                test_x = self.conv3(test_x)
                if test_x.shape[-1]>1:
                    test_x = self.maxpool(test_x)
                flattened_size = test_x.view(1, -1).size(1)
                #print(f"flattened_size of test input: {flattened_size}")
            
            # Fully Connected Layers
            self.fc1 = nn.Linear(flattened_size, 128)
            self.dropout = nn.Dropout(config["dropout_rate"])
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            #print(f"Input x shape: {x.shape}")
            # Ensure input is the right shape
            x = x.unsqueeze(1)  # Reshape input to (batch_size, 1, sequence_length)
            #print(f"After unsqueeze: {x.shape}")
            
            # Conv Block 1
            x = self.conv1(x)
            #print(f"After conv1: {x.shape}")
            x = self.bn1(x)
            #print(f"After bn1: {x.shape}")
            x = self.relu(x)
            #print(f"After relu: {x.shape}")
            x = self.maxpool(x)
            #print(f"After maxpool: {x.shape}")
            
            # Conv Block 2
            x = self.conv2(x)
            #print(f"After conv2: {x.shape}")
            x = self.bn2(x)
            #print(f"After bn2: {x.shape}")
            x = self.relu(x)
            #print(f"After relu: {x.shape}")
            x = self.maxpool(x)
            #print(f"After maxpool: {x.shape}")
            
            # Conv Block 3
            x = self.conv3(x)
            #print(f"After conv3: {x.shape}")
            x = self.bn3(x)
            #print(f"After bn3: {x.shape}")
            x = self.relu(x)
            #print(f"After relu: {x.shape}")
            if x.shape[-1]>1:
                x = self.maxpool(x)
            #print(f"After maxpool: {x.shape}")
            
            # Flatten and Fully Connected Layers
            x = x.view(x.size(0), -1)  # Flatten while preserving batch size
            #print(f"After flattening for FC: {x.shape}")
            x = self.fc1(x)
            #print(f"After fc1: {x.shape}")
            x = self.relu(x)
            #print(f"After relu: {x.shape}")
            x = self.dropout(x)
            #print(f"After dropout: {x.shape}")
            x = self.fc2(x)
            #print(f"After fc2: {x.shape}")
            x = self.softmax(x)
            #print(f"After softmax: {x.shape}")
            return x
        

    # Initialize model
    if config["num_conv_layers"]==2:
        model = CNNModel_2layer(config, input_dim=80, num_classes=10).to('cpu')
    elif config["num_conv_layers"]==3:
        model = CNNModel_3layer(config, input_dim=80, num_classes=10).to('cpu')
    # Throwing so error, no idea why
    else:
        raise ValueError(f"{config['num_conv_layers']} not supported, only 2 and 3 are supported")
    #model = CNNModel(input_dim=80, num_classes=10)
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    # Optimizer
    optimizer = None
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    # Training loop
    #train_loss_log = []
    for epoch in range(config["num_epochs"]):
        #train_loss_log.append(train_model(model, train_loader, optimizer))
        train_model(model, train_loader, optimizer)

    # Evaluation
    #return {
    #    'loss': total_loss / len(dataloader),
    #    'accuracy': np.mean(np.array(all_preds) == np.array(all_labels)),
    #    'predictions': all_preds,
    #    'true_labels': all_labels}
    train_acc = evaluate_model(model, train_loader)['accuracy']
    intra_test_acc = evaluate_model(model, intra_test_loader)['accuracy']
    cross_test_acc = evaluate_model(model, cross_test_loader)['accuracy']

    # Return combined performance metric (example: average accuracy)
    return {
        "config": config,
        #"train_loss_log": train_results,
        #"train_loss_log": train_results,
        #"train_loss_log": train_results,
        "train_acc": train_acc,
        "intra_test_acc": intra_test_acc,
        "cross_test_acc": cross_test_acc
    }


def make_dataloaders(data_splits, bs=32):
    train_dataset = TensorDataset(
        torch.tensor(data_splits['train']['feature'], dtype=torch.float32),
        torch.tensor(data_splits['train']['labels'], dtype=torch.long)
    )

    intra_test_loader = DataLoader(TensorDataset(
        torch.tensor(data_splits['intra_subject_test']['feature'], dtype=torch.float32),
        torch.tensor(data_splits['intra_subject_test']['labels'], dtype=torch.long)
    ), batch_size=bs, shuffle=False)

    cross_test_loader = DataLoader(TensorDataset(
        torch.tensor(data_splits['cross_subject_test']['feature'], dtype=torch.float32),
        torch.tensor(data_splits['cross_subject_test']['labels'], dtype=torch.long)
    ), batch_size=bs, shuffle=False)

    return train_dataset, intra_test_loader, cross_test_loader


def load_and_prepare_data(num_train_pids=24, num_total_pids=32, training_trials_per_gesture=8, finetuning_trials_per_gesture=3,
                          pickle_dataset_fullpath='C:\\Users\\kdmen\\Box\\Meta_Gesture_2024\\saved_datasets\\filtered_datasets\\$BStand_EMG_df.pkl'):
    num_test_pids = num_total_pids - num_train_pids
    with open(pickle_dataset_fullpath, 'rb') as file:
        raw_userdef_data_df = pickle.load(file)  # (204800, 19)

    userdef_df = raw_userdef_data_df.groupby(['Participant', 'Gesture_ID', 'Gesture_Num']).apply(create_feature_vectors)
    userdef_df = userdef_df.reset_index(drop=True)

    all_participants = userdef_df['Participant'].unique()
    # Shuffle the participants
    np.random.shuffle(all_participants)
    # Split into two groups
    #train_participants = all_participants[:num_test_pids]  # First 24 participants
    test_participants = all_participants[num_test_pids:]  # Remaining 8 participants

    #convert Gesture_ID to numerical with new Gesture_Encoded column
    label_encoder = LabelEncoder()
    userdef_df['Gesture_Encoded'] = label_encoder.fit_transform(userdef_df['Gesture_ID'])
    label_encoder2 = LabelEncoder()
    userdef_df['Cluster_ID'] = label_encoder2.fit_transform(userdef_df['Participant'])

    # Prepare data
    data_splits = prepare_data(
        userdef_df, 'feature', 'Gesture_Encoded', 
        all_participants, test_participants, 
        training_trials_per_gesture=training_trials_per_gesture, finetuning_trials_per_gesture=finetuning_trials_per_gesture,
    )

    return make_dataloaders(data_splits)