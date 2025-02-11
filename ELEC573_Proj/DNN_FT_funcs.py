import torch
import torch.nn as nn
#import torch.optim as optim
from torch.utils.data import DataLoader
#import numpy as np
import json
from collections import defaultdict
#from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime
import time
import copy
import pickle
import random
from sklearn.preprocessing import LabelEncoder

from moments_engr import *
#from model_classes import *
from revamped_model_classes import *


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

    #print(f"Results saved to: {results_path}")

def group_data_by_participant(features, labels, pids):
    """Helper function to group features and labels by participant ID."""
    user_data = defaultdict(lambda: ([], []))  # Tuple of lists: (features, labels)
    for feature, label, pid in zip(features, labels, pids):
        user_data[pid][0].append(feature)
        user_data[pid][1].append(label)
    return user_data

def evaluate_configuration_on_ft(datasplit, pretrained_model, config, model_str, timestamp):
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
    ## ft and cross are paired, like train and intra are paired 
    ## The same users are in ft and cross datasets
    for pid in ft_user_data.keys() & cross_user_data.keys():  # Only common participant IDs
        if config["verbose"]:
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
        if config["save_ft_models"]:
            save_model(finetuned_model, model_str, config["models_save_dir"], f"{pid}_pretrainedFT", verbose=config["verbose"])


    return user_accuracies


def make_data_split(expdef_df, num_gesture_training_trials=8, num_gesture_ft_trials=3, num_train_users=24):
    """This is just a wrapper function around prepare_data(), downside is you can't access all_participants and test_participants"""

    all_participants = expdef_df['Participant'].unique()
    # Shuffle the participants
    random.shuffle(all_participants)
    # Split into two groups
    #train_participants = all_participants[:num_train_users]  # First num_train_users participants
    test_participants = all_participants[num_train_users:]  # Remaining 32-num_train_users participants

    # Prepare data
    data_splits = prepare_data(
        expdef_df, 'feature', 'Gesture_Encoded', 
        all_participants, test_participants, 
        training_trials_per_gesture=num_gesture_training_trials, finetuning_trials_per_gesture=num_gesture_ft_trials,
    )

    return data_splits


def save_model(model, model_str, save_dir, model_scenario_str, verbose=True, timestamp=None):
    # TODO: save_model verbose should pull from config... or at least make sure verbose is passed in through config
    """Save the model with a timestamp."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    full_path = os.path.join(save_dir, f'{timestamp}_{model_scenario_str}_{model_str}_model.pth')
    if verbose:
        print("Full Path:", full_path)
    torch.save(model.state_dict(), full_path)

# main function for hyperparameter tuning the finetuned models
def hyperparam_tuning_for_ft(model_str, expdef_df, hyperparameter_space, architecture_space, metadata_config,
                             num_configs_to_test=20, num_datasplits_to_test=2, num_train_trials=8, num_ft_trials=3, 
                             guaranteed_configs_to_test_lst=None, seed=100):
    
    print("Creating directories")
    # Results
    os.makedirs(metadata_config["results_save_dir"][0])
    print(f'Directory {metadata_config["results_save_dir"][0]} created successfully!')
    # Models
    os.makedirs(metadata_config["models_save_dir"][0]) 
    print(f'Directory {metadata_config["models_save_dir"][0]} created successfully!') 

    # Generate all possible configurations
    ## Does this like shuffle or is this deterministic?
    print("Combining configs")
    # Grid Search (generates full space, so O(N))
    #configs = list(ParameterGrid({**hyperparameter_space, **architecture_space, **metadata_config}))
    # Random search variant (samples so O(1))
    #configs = list(ParameterSampler({**hyperparameter_space, **architecture_space, **metadata_config}, n_iter=num_configs_to_test))
    configs = list(ParameterSampler(
        {**hyperparameter_space, **architecture_space, **metadata_config},
        n_iter=num_configs_to_test,
        random_state=seed
    ))
    # Shuffle the configurations
    random.shuffle(configs)
    configs = configs[:num_configs_to_test]  # Limit the number of configurations to test
    if guaranteed_configs_to_test_lst is not None:
        configs.extend(guaranteed_configs_to_test_lst)

    # This creates the (multiple) train/test splits
    print("Creating datasplits")
    data_splits_lst = []
    for datasplit in range(num_datasplits_to_test):
        all_participants = expdef_df['Participant'].unique()
        # Shuffle the participants for train/test user split --> UNIQUE
        random.shuffle(all_participants)
        test_participants = all_participants[24:]  # 24 train / 8 test
        data_splits_lst.append(prepare_data(
            expdef_df, 'feature', 'Gesture_Encoded', 
            all_participants, test_participants, 
            training_trials_per_gesture=num_train_trials, 
            finetuning_trials_per_gesture=num_ft_trials,
        ))

    results = []
    for config_idx, config in enumerate(configs):
        print(f"Testing config {config_idx + 1}/{len(configs)}:\n{config}")
        start_time = time.time()

        split_results = []
        for datasplit in data_splits_lst:
            # Here, each config+datasplit pair gets its own timestamp... not sure what is optimal
            ## Dont want anything to overwrite mainly
            ## TODO: Augment saving to create and save to folders (folders are timestamps?)
            ## I did that for results, but maybe should make the folder include the model name and scenario...
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")

            # Train the model
            training_results = main_training_pipeline(
                datasplit, all_participants=all_participants, test_participants=test_participants,
                model_type=model_str, config=config
            )
            pretrained_model = training_results["model"]

            # Save the pretrained model --> Don't actually this is super slow...
            #save_model(pretrained_model, model_str, metadata_config["models_save_dir"][0], "pretrained", verbose=metadata_config["verbose"][0])

            # Evaluate the configuration on the current data split
            user_accuracies = evaluate_configuration_on_ft(datasplit, pretrained_model, config, model_str, timestamp)
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
        print(f"Overall accuracies: {overall_avg_accuracy:.4f}")
        total_time = time.time() - start_time
        print(f"Completed in {total_time:.2f}s\n")

    # Save the results
    ## This is the aggregated and sorted JSON file. This always needs to be saved
    save_results(results, metadata_config["results_save_dir"][0], metadata_config["timestamp"][0]) 

    return results


def load_expdef_gestures(feateng_method, filepath_pkl='C:\\Users\\kdmen\\Box\\Meta_Gesture_2024\\saved_datasets\\filtered_datasets\\$BStand_EMG_df.pkl'):
    with open(filepath_pkl, 'rb') as file:
        raw_expdef_data_df = pickle.load(file)  # (204800, 19)

    if feateng_method is None:
        # Group by metadata columns and combine data into a matrix
        condensed_df = (
            raw_expdef_data_df.groupby(['Participant', 'Gesture_ID', 'Gesture_Num'], as_index=False)
            .apply(lambda group: pd.Series({
                'feature': group[raw_expdef_data_df.columns[3:]].to_numpy()
            }))
            .reset_index(drop=True)
        )
        # Combine metadata columns with the new data column
        expdef_df = pd.concat([raw_expdef_data_df[['Participant', 'Gesture_ID', 'Gesture_Num']].drop_duplicates().reset_index(drop=True), condensed_df['feature']], axis=1)
    elif feateng_method=="moments":
        expdef_df = raw_expdef_data_df.groupby(['Participant', 'Gesture_ID', 'Gesture_Num']).apply(create_khushaba_spectralmomentsFE_vectors)
        expdef_df = expdef_df.reset_index(drop=True)
    elif feateng_method=="FS":
        expdef_df = raw_expdef_data_df.groupby(['Participant', 'Gesture_ID', 'Gesture_Num']).apply(create_abbaspour_FS_vectors)
        expdef_df = expdef_df.reset_index(drop=True)
    else:
        raise ValueError(f"feateng_method {feateng_method} not recognized")
    
    #convert Gesture_ID to numerical with new Gesture_Encoded column
    label_encoder = LabelEncoder()
    expdef_df['Gesture_Encoded'] = label_encoder.fit_transform(expdef_df['Gesture_ID'])
    label_encoder2 = LabelEncoder()
    expdef_df['Cluster_ID'] = label_encoder2.fit_transform(expdef_df['Participant'])

    return expdef_df


def set_optimizer(model, lr=0.001, use_weight_decay=False, weight_decay=0.01, optimizer_name="ADAM"):
    """Configure optimizer with optional weight decay."""

    if optimizer_name!="ADAM":
        raise ValueError("Only ADAM is supported right now")
    
    if use_weight_decay:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer


def prepare_data(df, feature_column, target_column, participants, test_participants, 
                 training_trials_per_gesture=8, finetuning_trials_per_gesture=3):
    """
    Prepare data for training and testing across participants.
    
    Args:
    - df: DataFrame containing features and labels
    - feature_column: Name of column with feature vectors
    - target_column: Name of column with encoded gesture labels
    - participants: List of all participants
    - test_participants: List of participants to use as test set
    - trials_per_gesture: Number of trials to use for training per gesture
    
    Returns:
    - Training and testing data dictionaries
    """
    train_participants = [p for p in participants if p not in test_participants]
    
    # train_data and intra_subject_test_data are train/test pairs!
    train_data = {
        'feature': [],
        'labels': [],
        'participant_ids': []
    }
    # Gesture trials from the training subject withheld for intra-subject testing
    intra_subject_test_data = {
        'feature': [],
        'labels': [],
        'participant_ids': []
    }
    # Dataset of novel users for finetuning 
    ## novel_trainFT_data and cross_subject_test_data are train/test pairs!
    novel_trainFT_data = {
        'feature': [],
        'labels': [],
        'participant_ids': []
    }
    # I think this could equivalently be called "novel test data"
    cross_subject_test_data = {
        'feature': [],
        'labels': [],
        'participant_ids': []
    }
    
    for participant in participants:
        participant_data = df[df['Participant'] == participant]
        # Group by gesture
        gesture_groups = participant_data.groupby(target_column)
        
        # Separate data
        if participant in train_participants:
            training_dict = train_data
            testing_dict = intra_subject_test_data
            max_trials = training_trials_per_gesture
        else:
            training_dict = novel_trainFT_data
            testing_dict = cross_subject_test_data
            max_trials = finetuning_trials_per_gesture
            
        for gesture, group in gesture_groups:
            # NOTE: This is by gesture, so by def all thelabels will be the same here

            # Randomize and select trials
            group_features = np.array([x.flatten() for x in group[feature_column]])
            group_labels = group[target_column].values
            
            # Randomly select specified number of trials
            indices = np.random.choice(
                len(group_features), 
                #min(max_trials, len(group_features)), # This should always be max_trials right...
                max_trials,
                replace=False)

            train_features = group_features[indices]
            train_labels = group_labels[indices]
            training_dict['feature'].extend(train_features)
            training_dict['labels'].extend(train_labels)
            training_dict['participant_ids'].extend([participant] * len(train_labels))

            # Get complementary indices for the test set 
            all_indices = np.arange(len(group_features)) 
            test_indices = np.setdiff1d(all_indices, indices)

            test_features = group_features[test_indices]
            test_labels = group_labels[test_indices]
            testing_dict['feature'].extend(test_features)
            testing_dict['labels'].extend(test_labels)
            testing_dict['participant_ids'].extend([participant] * len(test_labels))
    
    return {
        'train': {
            'feature': np.array(train_data['feature']),
            'labels': np.array(train_data['labels']),
            'participant_ids': train_data['participant_ids']
        },
        'intra_subject_test': {
            'feature': np.array(intra_subject_test_data['feature']),
            'labels': np.array(intra_subject_test_data['labels']),
            'participant_ids': intra_subject_test_data['participant_ids']
        },
        'novel_trainFT': {
            'feature': np.array(novel_trainFT_data['feature']),
            'labels': np.array(novel_trainFT_data['labels']),
            'participant_ids': novel_trainFT_data['participant_ids']
        },
        'cross_subject_test': {
            'feature': np.array(cross_subject_test_data['feature']),
            'labels': np.array(cross_subject_test_data['labels']),
            'participant_ids': cross_subject_test_data['participant_ids']
        }
    }


def train_model(model, train_loader, optimizer, criterion=nn.CrossEntropyLoss(), device=None):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate_model(model, dataloader, criterion=nn.CrossEntropyLoss(), device=None):
    """Evaluate the model and return detailed performance metrics"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': np.mean(np.array(all_preds) == np.array(all_labels)),
        'predictions': all_preds,
        'true_labels': all_labels
    }


def gesture_performance_by_participant(predictions, true_labels, all_unique_participants, 
                                       all_shuffled_participant_ids, unique_gestures):
    """
    Calculate performance metrics for each participant and gesture
    
    Returns:
    - Dictionary with performance for each participant and gesture
    """
    performance = {}
    
    for participant in all_unique_participants:
        participant_mask = np.array(all_shuffled_participant_ids) == participant
        participant_preds = np.array(predictions)[participant_mask]
        participant_true = np.array(true_labels)[participant_mask]
        
        participant_performance = {}
        for gesture in unique_gestures:
            gesture_mask = participant_true == gesture
            if np.sum(gesture_mask) > 0:
                gesture_preds = participant_preds[gesture_mask]
                gesture_true = participant_true[gesture_mask]
                participant_performance[gesture] = np.mean(gesture_preds == gesture_true)
        
        performance[participant] = participant_performance
    
    return performance


def main_training_pipeline(data_splits, all_participants, test_participants, model_type, config, 
                           train_intra_cross_loaders=None, save_loss_here=False):
    """
    Main training pipeline with comprehensive performance tracking
    
    Args:
    - data_splits: dictionary (NOT DATALOADERS) with keys according to scenario and train/test
    - all_participants: List of all (unique) participants
    - test_participants: List of participants to hold out
    - model_type: string of model name
    - config: dictionary of configuration values for model architecture and hyperparams
    - train_intra_cross_loaders: ... not sure what this is, it seems to be left to the default None
    Returns:
    - Trained model, performance metrics
    """

    bs = config["batch_size"]
    #criterion = config["criterion"]  # This doesnt do anything rn
    lr = config["learning_rate"]
    weight_decay = config["weight_decay"]
    sequence_length = config["sequence_length"]
    time_steps = config["time_steps"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Unique gestures and number of classes
    ## TODO: Why is train_pids an unshuffled list of 1920 entries? Shouldn't it at least be shuffled?
    ## I am not sure if train_pids is aligned with the dataloaders (since the training dataloader shuffles)
    ## The testing dataloaders don't shuffle tho
    ## intra_test_results are in order [0,0,1,1,2,2, etc]
    ## So I'm pretty sure train_pids is wrong then, it should be shuffled...

    if train_intra_cross_loaders is not None:  # This is used in the NB full study...
        # Hardcoding cause I don't wanna deal with dataloader extraction
        #num_classes = 10
        #unique_gestures = list(range(num_classes))
        #input_dim = 80

        train_loader = train_intra_cross_loaders[0]
        intra_test_loader = train_intra_cross_loaders[1]
        cross_test_loader = train_intra_cross_loaders[2]
        #if len(train_intra_cross_loaders)==4:
        #    ft_loader = train_intra_cross_loaders[3]  # Not used at all...

        # Here we can assume that data_splits is None
        #unique_gestures = np.unique(data_splits['train']['labels'])
        num_classes = 10
        input_dim = next(iter(train_loader))[0].shape

        train_pids = [pid for pid in all_participants if pid not in test_participants]
        intra_pids = train_pids
        cross_pids = test_participants
    else:
        unique_gestures = np.unique(data_splits['train']['labels'])
        num_classes = len(unique_gestures)
        input_dim = data_splits['train']['feature'].shape[1]

        train_pids = data_splits['train']['participant_ids']  # THIS IS IN ORDER! NOT SHUFFLED!!
        intra_pids = data_splits['intra_subject_test']['participant_ids']
        cross_pids = data_splits['cross_subject_test']['participant_ids']
    
        my_gesture_dataset = select_dataset_class(model_type)
        
        train_dataset = my_gesture_dataset(
            data_splits['train']['feature'], 
            data_splits['train']['labels'], 
            sl=sequence_length, 
            ts=time_steps)
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True) #, drop_last=True)
        
        # INTRA SUBJECT (480)
        intra_test_dataset = my_gesture_dataset(
            data_splits['intra_subject_test']['feature'], 
            data_splits['intra_subject_test']['labels'], 
            sl=sequence_length, 
            ts=time_steps)
        # Shuffle doesn't matter for testing
        intra_test_loader = DataLoader(intra_test_dataset, batch_size=bs, shuffle=False) #, drop_last=True)

        # CROSS SUBJECT (560)
        ## Wait shouldn't I have 2 cross datasets? Are these the withheld users??
        ## I should have finetuning and withheld right? Presumably cross_test is withheld?
        cross_test_dataset = my_gesture_dataset(
            data_splits['cross_subject_test']['feature'], 
            data_splits['cross_subject_test']['labels'], 
            sl=sequence_length, 
            ts=time_steps)
        # Shuffle doesn't matter for testing
        cross_test_loader = DataLoader(cross_test_dataset, batch_size=bs, shuffle=False) #, drop_last=True)

    # Select model
    model = select_model(model_type, config) #, device=device, input_dim=input_dim, num_classes=num_classes)
    # Loss and optimizer
    optimizer = set_optimizer(model, lr=lr, use_weight_decay=weight_decay>0, weight_decay=weight_decay)
    # Annealing the learning rate if applicable
    if config["lr_scheduler_gamma"]<1.0:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["lr_scheduler_gamma"])
    
    # Training
    train_loss_log = []
    intra_test_loss_log = []
    cross_test_loss_log = []

    max_epochs = config["num_epochs"]
    epoch = 0
    done = False
    if config['use_earlystopping']:
        earlystopping = EarlyStopping()
    
    # Is this by participant? Or just once per config?
    if save_loss_here:
        # Open a text file for logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        log_file = open(f"{config['results_save_dir']}\\{timestamp}_{model_type}_pretrained_training_log.txt", "w")
    while not done and epoch < max_epochs:
        epoch += 1
        train_loss = train_model(model, train_loader, optimizer)
        train_loss_log.append(train_loss)

        # Validation
        intra_test_loss = evaluate_model(model, intra_test_loader)['loss']
        intra_test_loss_log.append(intra_test_loss)
        cross_test_loss = evaluate_model(model, cross_test_loader)['loss']
        cross_test_loss_log.append(cross_test_loss)

        # Anneal the learning rate (advance scheduler) if applicable
        if config["lr_scheduler_gamma"]<1.0:
            scheduler.step()

        # Early stopping check
        if config['use_earlystopping'] and earlystopping(model, intra_test_loss):
            done = True
            earlystopping_status = earlystopping.status
        else:
            earlystopping_status = ""
        # Log metrics to the console and the text file
        log_message = (
            f"Epoch {epoch}/{max_epochs}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Intra Testing Loss: {intra_test_loss:.4f}, "
            f"Cross Testing Loss: {cross_test_loss:.4f}, "
            f"{earlystopping_status}\n")
        if config["verbose"]:
            print(log_message, end="")  # Print to console
        if save_loss_here:
            log_file.write(log_message)  # Write to file
    if save_loss_here:
        # Close the log file
        log_file.close()
    
    # Evaluation --> One-off final results! Gets used in gesture_performance_by_participant
    train_results = evaluate_model(model, train_loader)
    intra_test_results = evaluate_model(model, intra_test_loader)
    cross_test_results = evaluate_model(model, cross_test_loader)

    if train_intra_cross_loaders is None:
        # This isn't working and I don't have time to fix it
        ## Is this running right now? I think so?
        
        # Performance by participant and gesture
        train_performance = gesture_performance_by_participant(
            train_results['predictions'], 
            train_results['true_labels'], 
            all_participants, train_pids, 
            unique_gestures
        )
        
        intra_test_performance = gesture_performance_by_participant(
            intra_test_results['predictions'], 
            intra_test_results['true_labels'], 
            all_participants, intra_pids, 
            unique_gestures
        )

        cross_test_performance = gesture_performance_by_participant(
            cross_test_results['predictions'], 
            cross_test_results['true_labels'], 
            all_participants, cross_pids, 
            unique_gestures
        )
    else:
        train_performance = None
        intra_test_performance = None
        cross_test_performance = None

    return {
        'model': model,
        'train_performance': train_performance,
        'intra_test_performance': intra_test_performance,
        'cross_test_performance': cross_test_performance,
        # These are the final accuracies on the respective datasets
        'train_accuracy': train_results['accuracy'],
        'intra_test_accuracy': intra_test_results['accuracy'],
        'cross_test_accuracy': cross_test_results['accuracy'], 
        'train_loss_log': train_loss_log,
        'intra_test_loss_log': intra_test_loss_log,
        'cross_test_loss_log': cross_test_loss_log
    }


def fine_tune_model(finetuned_model, fine_tune_loader, config, timestamp, test_loader=None, pid=None, use_earlystopping=None, num_epochs=50):
    """
    Fine-tune the base model on a small subset of data
    
    Args:
    - base_model: Pretrained model to fine-tune
    - fine_tune_data: Dictionary with features and labels for fine-tuning
    - num_epochs: Number of fine-tuning epochs
    
    Returns:
    - Fine-tuned model
    """

    finetune_strategy = config["finetune_strategy"]
    '''
    - finetune_strategy: The fine-tuning method to use. Options:
        - "full": Train the entire model.
        - "freeze_cnn": Freeze CNN, train LSTM and dense layers.
        - "freeze_cnn_lstm": Freeze CNN + LSTM, train only dense layers.
        - "freeze_all_add_dense": Freeze entire model, add a new dense layer.
        - "progressive_unfreeze": Start with frozen CNN/LSTM, progressively unfreeze.
    '''

    if pid is None:
        pid = ""
    else:
        pid = pid + "_"

    if use_earlystopping is None:
        use_earlystopping = config["use_earlystopping"]
        # Assume num_epochs is an integer (ideally 50) and use that
        max_epochs = num_epochs
    else:
        max_epochs = config["num_ft_epochs"]
    
    # Make a copy of the original (pretrained) model
    frozen_original_model = copy.deepcopy(finetuned_model)

    finetuned_model.train()  # Ensure the model is in training mode

    # Apply fine-tuning strategy
    if finetune_strategy == "full":
        # We don't need to do any layer freezing
        pass
    elif finetune_strategy == "freeze_cnn":
        for param in finetuned_model.conv_layers.parameters():
            param.requires_grad = False
    elif finetune_strategy == "freeze_cnn_lstm":
        for param in finetuned_model.conv_layers.parameters():
            param.requires_grad = False
        for param in finetuned_model.lstm.parameters():
            param.requires_grad = False
    elif finetune_strategy == "freeze_all_add_dense":
        for param in finetuned_model.parameters():
            param.requires_grad = False  # Freeze entire model
        # Add a new dense layer
        ## Actually assumed the last output has 10 (num_classes) nodes 
        #finetuned_model.fc.in_features --> Attempting to programatically get the final size
        ## I think the last layer is not named fc tho?
        num_features = config["num_classes"] 
        ## Add a new dense layer and train it
        #new_dense = nn.Linear(128, 64)  # Adjust size based on your architecture
        #finetuned_model.fc_layers.add_module("new_dense", new_dense)
        # How to verify that this actually got added and is in the right place...
        finetuned_model.fc = nn.Sequential(
            nn.Linear(num_features, config["added_dense_ft_hidden_size"]),
            nn.ReLU(),
            nn.Linear(config["added_dense_ft_hidden_size"], config["num_classes"])
        )
    elif finetune_strategy == "progressive_unfreeze":
        # Freeze everything first
        for param in finetuned_model.parameters():
            param.requires_grad = False
        
        # Get all named layers in reverse order (last layer first)
        layers = list(finetuned_model.named_parameters())[::-1]  
        # Unfreeze only the last dense layer at the beginning
        last_layer_name, last_layer_param = layers[0]
        last_layer_param.requires_grad = True
        #print(f"Initially unfrozen: {last_layer_name}")
        # Track how many layers are currently unfrozen
        num_unfrozen = 1  # Since last layer is already unfrozen
        total_layers = len(layers)  # Total number of layers
        # Define unfreeze step
        unfreeze_step = 1  # Adjust this as needed
    else:
        raise ValueError(f"finetune_strategy ({finetune_strategy}) not recognized!")

    # Loss and optimizer (with different learning rate for fine-tuning)
    optimizer = set_optimizer(finetuned_model, lr=config["ft_learning_rate"], use_weight_decay=config["ft_weight_decay"] > 0, weight_decay=config["ft_weight_decay"])
    # Set up learning rate scheduler if applicable
    if config["lr_scheduler_gamma"]<1.0:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["lr_scheduler_gamma"])

    # Fine-tuning training loop
    train_loss_log = []
    test_loss_log = []
    epoch = 0
    done = False
    if use_earlystopping:
        earlystopping = EarlyStopping()
    # Open a text file for logging
    if config["log_each_pid_results"]:
        log_file = open(f"{timestamp}_{pid}ft_log.txt", "w")
    while not done and epoch < max_epochs:
        epoch += 1

        # Progressive unfreezing logic
        if (finetune_strategy == "progressive_unfreeze" and 
            num_unfrozen < total_layers and 
            epoch % config["progressive_unfreezing_schedule"] == 0):
            # Unfreeze one additional layer
            num_unfreeze = min(num_unfrozen + unfreeze_step, total_layers)
            for name, param in layers[num_unfrozen:num_unfreeze]:
                param.requires_grad = True
                #print(f"Unfroze layer: {name}")
            num_unfrozen = num_unfreeze  # Update unfrozen layer count

        train_loss = train_model(finetuned_model, fine_tune_loader, optimizer)
        train_loss_log.append(train_loss)
        novel_intra_test_loss = evaluate_model(finetuned_model, test_loader)['loss']
        test_loss_log.append(novel_intra_test_loss)

        # Anneal the learning rate if applicable (advance the scheduler)
        if config["lr_scheduler_gamma"]<1.0:
            scheduler.step()

        # Early stopping check
        if use_earlystopping==True and earlystopping(finetuned_model, novel_intra_test_loss):
            done = True
            earlystopping_status = earlystopping.status
        else:
            earlystopping_status = ""

    # Log metrics to the console and the text file, AFTER the while loop has finished
    log_message = (
        f"Participant ID {pid[:-1]}, "  # [] in order to drop the "_" 
        f"Epoch {epoch}/{max_epochs}, "
        # TODO: Why are these losses and not accuracies...
        f"FT Train Loss: {train_loss:.4f}, "
        f"Novel Intra Subject Testing Loss: {novel_intra_test_loss:.4f}, "
        f"{earlystopping_status}\n")
    if config["verbose"]:
            print(log_message, end="")  # Print to console
    if config["log_each_pid_results"]:
        log_file.write(log_message)  # Write to file
        # Close the log file
        log_file.close()

    assert(not finetuned_model == frozen_original_model)

    # Could refactor this func to return a dict like the main training pipline...
    #return {
        #    'model': model,
        #    'train_performance': train_performance,
        #    'intra_test_performance': intra_test_performance,
        #    'cross_test_performance': cross_test_performance,
        #    # These are the final accuracies on the respective datasets
        #    'train_accuracy': train_results['accuracy'],
        #    'intra_test_accuracy': intra_test_results['accuracy'],
        #    'cross_test_accuracy': cross_test_results['accuracy'], 
        #    'train_loss_log': train_loss_log,
        #    'intra_test_loss_log': intra_test_loss_log,
        #    'cross_test_loss_log': cross_test_loss_log
        #}
    return finetuned_model, frozen_original_model, train_loss_log, test_loss_log


def plot_gesture_performance(performance_dict, title, save_filename, save_path="ELEC573_Proj\\results\\heatmaps"):
    """
    Create a heatmap of gesture performance for each participant
    
    Args:
    - performance_dict: Dictionary of performance metrics
    - title: Title for the plot
    - save_path: Optional path to save the plot
    """
    # Convert performance dict to a DataFrame
    data = []
    for participant, gesture_performance in performance_dict.items():
        for gesture, accuracy in gesture_performance.items():
            data.append({
                'Participant': participant, 
                'Gesture': gesture, 
                'Accuracy': accuracy
            })
    
    performance_df = pd.DataFrame(data)
    
    # Create a pivot table for heatmap
    performance_pivot = performance_df.pivot(
        index='Participant', 
        columns='Gesture', 
        values='Accuracy'
    )
    
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 8))
    
    # Create heatmap using seaborn
    sns.heatmap(
        performance_pivot, 
        annot=True, 
        cmap='YlGnBu', 
        center=0.5, 
        vmin=0, 
        vmax=1,
        fmt='.2f'
    )
    
    plt.title(title)
    plt.tight_layout()
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    fig_filename = f"{save_filename}.png"
    fig_dir = os.path.join(save_path, f"{timestamp}")
    os.makedirs(fig_dir, exist_ok=True)
    fig_save_path = os.path.join(fig_dir, fig_filename)
    plt.savefig(fig_save_path)
    return performance_pivot


def print_detailed_performance(performance_dict, set_type):
    """
    Print detailed performance metrics for each participant and gesture
    
    Args:
    - performance_dict: Dictionary of performance metrics
    - set_type: String describing the dataset (e.g., 'Training', 'Testing')
    """
    print(f"\n--- {set_type} Set Performance ---")
    
    # Overall statistics
    all_accuracies = []
    
    for participant, gesture_performance in performance_dict.items():
        print(f"\nParticipant {participant}:")
        participant_accuracies = []
        
        for gesture, accuracy in sorted(gesture_performance.items()):
            print(f"  Gesture {gesture}: {accuracy:.2%}")
            participant_accuracies.append(accuracy)
            all_accuracies.append(accuracy)
        
        # Participant-level summary
        print(f"  Average Accuracy: {np.mean(participant_accuracies):.2%}")
    
    # Overall summary
    print(f"\nOverall {set_type} Set Summary:")
    print(f"Mean Accuracy: {np.mean(all_accuracies):.2%}")
    print(f"Accuracy Standard Deviation: {np.std(all_accuracies):.2%}")
    print(f"Minimum Accuracy: {np.min(all_accuracies):.2%}")
    print(f"Maximum Accuracy: {np.max(all_accuracies):.2%}")


def visualize_model_performance(results, print_results=False):
    """
    Comprehensive visualization and printing of model performance
    
    Args:
    - results: Dictionary containing training and testing performance
    """
    # Visualize Training Set Performance
    train_performance_heatmap = plot_gesture_performance(
        results['train_performance'], 
        'Training Set - Gesture Performance by Participant',
        'training_performance_heatmap'
    )
    
    # Visualize Testing Set Performance
    intra_test_performance_heatmap = plot_gesture_performance(
        results['intra_test_performance'], 
        'Intra Testing Set - Gesture Performance by Participant',
        'intra_testing_performance_heatmap'
    )

    # Visualize Testing Set Performance
    cross_test_performance_heatmap = plot_gesture_performance(
        results['cross_test_performance'], 
        'Cross Testing Set - Gesture Performance by Participant',
        'cross_testing_performance_heatmap'
    )
    
    if print_results:
        # Print detailed performance
        print_detailed_performance(results['train_performance'], 'Training')
        print_detailed_performance(results['intra_test_performance'], 'Intra Testing')
        print_detailed_performance(results['cross_test_performance'], 'Cross Testing')
        
        # Additional overall metrics
        print("\nOverall Model Performance:")
        print(f"Training Accuracy: {results['train_accuracy']:.2%}")
        print(f"Intra Testing Accuracy: {results['intra_test_accuracy']:.2%}")
        print(f"Cross Testing Accuracy: {results['cross_test_accuracy']:.2%}")


def log_performance(results, config, base_filename='model_performance'):
    # THIS IS NOT GETTING USED IN MY CURRENT CODE???
    print("LOG_PERFORMANCE CALLED! FIGURE OUT WHERE!")

    """
    Comprehensive logging of model performance
    
    Args:
    - results: Dictionary containing training and testing performance
    - log_dir: Directory to save log files
    - base_filename: Base name for log files
    
    Returns:
    - Path to the created log file
    """
    # Create log directory if it doesn't exist
    os.makedirs(config["perf_log_dir"], exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_filename = f"{timestamp}_{base_filename}.txt"
    log_path = os.path.join(config["perf_log_dir"], log_filename)
    
    # Capture console output and log to file
    class Logger:
        def __init__(self, file_path):
            self.terminal = sys.stdout
            self.log = open(file_path, "w")
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    # Redirect stdout to both console and file
    sys.stdout = Logger(log_path)
    
    try:
        if config is not None:
            for key, value in config.items():
                # Handle list values separately for better formatting
                if isinstance(value, list):
                    print(f"{key}: {', '.join(map(str, value))}\n")
                else:
                    print(f"{key}: {value}\n")

        # Perform visualization and logging
        print("Model Performance Analysis")
        print("=" * 30)
        
        # Training Set Performance
        print("\n--- Training Set Performance ---")
        for participant, gesture_performance in results['train_performance'].items():
            if participant in results['train_performance'].keys():
                continue
            print(f"\nParticipant {participant}:")
            participant_accuracies = []
            for gesture, accuracy in sorted(gesture_performance.items()):
                print(f"  Gesture {gesture}: {accuracy:.2%}")
                participant_accuracies.append(accuracy)
            print(f"  Average Accuracy: {np.mean(participant_accuracies):.2%}")
        # Training Set Summary
        train_accuracies = [acc for participant in results['train_performance'].values() for acc in participant.values()]
        print("\nTraining Set Summary:")
        print(f"Mean Accuracy: {np.mean(train_accuracies):.2%}")
        print(f"Accuracy Standard Deviation: {np.std(train_accuracies):.2%}")
        print(f"Minimum Accuracy: {np.min(train_accuracies):.2%}")
        print(f"Maximum Accuracy: {np.max(train_accuracies):.2%}")

        ######################################################################
        
        # Repeat for Testing Set
        print("\n--- INTRA Testing Set Performance ---")
        for participant, gesture_performance in results['intra_test_performance'].items():
            print(f"\nParticipant {participant}:")
            participant_accuracies = []
            for gesture, accuracy in sorted(gesture_performance.items()):
                print(f"  Gesture {gesture}: {accuracy:.2%}")
                participant_accuracies.append(accuracy)
            print(f"  Average Accuracy: {np.mean(participant_accuracies):.2%}")
        # Testing Set Summary
        test_accuracies = [acc for participant in results['intra_test_performance'].values() for acc in participant.values()]
        print("\nIntra Testing Set Summary:")
        print(f"Mean Accuracy: {np.mean(test_accuracies):.2%}")
        print(f"Accuracy Standard Deviation: {np.std(test_accuracies):.2%}")
        print(f"Minimum Accuracy: {np.min(test_accuracies):.2%}")
        print(f"Maximum Accuracy: {np.max(test_accuracies):.2%}")

        ######################################################################

        # Repeat for Testing Set
        print("\n--- CROSS Testing Set Performance ---")
        for participant, gesture_performance in results['cross_test_performance'].items():
            print(f"\nParticipant {participant}:")
            participant_accuracies = []
            for gesture, accuracy in sorted(gesture_performance.items()):
                print(f"  Gesture {gesture}: {accuracy:.2%}")
                participant_accuracies.append(accuracy)
            print(f"  Average Accuracy: {np.mean(participant_accuracies):.2%}")
        # Testing Set Summary
        test_accuracies = [acc for participant in results['cross_test_performance'].values() for acc in participant.values()]
        print("\nCross Testing Set Summary:")
        print(f"Mean Accuracy: {np.mean(test_accuracies):.2%}")
        print(f"Accuracy Standard Deviation: {np.std(test_accuracies):.2%}")
        print(f"Minimum Accuracy: {np.min(test_accuracies):.2%}")
        print(f"Maximum Accuracy: {np.max(test_accuracies):.2%}")

        ######################################################################
        
        # Overall Model Performance
        print("\nOverall Model Performance:")
        print(f"Training Accuracy: {results['train_accuracy']:.2%}")
        print(f"Intra Testing Accuracy: {results['intra_test_accuracy']:.2%}")
        print(f"Cross Testing Accuracy: {results['cross_test_accuracy']:.2%}")
    
    finally:
        # Restore stdout
        sys.stdout = sys.stdout.terminal
    
    return log_path

