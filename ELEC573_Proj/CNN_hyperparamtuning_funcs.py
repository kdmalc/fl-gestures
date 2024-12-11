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
    class DynamicCNNModel(nn.Module):
        def __init__(self, config, input_dim, num_classes):
            super(DynamicCNNModel, self).__init__()
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
                layers.append(nn.MaxPool1d(2))
                in_channels = out_channels
                seq_len = (seq_len - config["kernel_size"] + 2 * config["padding"]) // config["stride"] + 1
                seq_len = seq_len // 2

            self.conv = nn.Sequential(*layers)
            self.fc1 = nn.Linear(in_channels * seq_len, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.dropout = nn.Dropout(config["dropout_rate"])

        def forward(self, x):
            x = x.unsqueeze(1)
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    # Initialize model
    model = DynamicCNNModel(config, input_dim=100, num_classes=10).to('cpu')
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    # Optimizer
    optimizer = None
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    # Training loop
    train_loss_log = []
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