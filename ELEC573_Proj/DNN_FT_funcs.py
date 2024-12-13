import torch
import torch.nn as nn
#import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime
import copy

from model_classes import *


class GestureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def get_optimizer(model, lr=0.001, use_weight_decay=False, weight_decay=0.01, optimizer_name="ADAM"):
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
    intra_subject_test_data = {
        'feature': [],
        'labels': [],
        'participant_ids': []
    }
    # novel_trainFT_data and cross_subject_test_data are train/test pairs!
    novel_trainFT_data = {
        'feature': [],
        'labels': [],
        'participant_ids': []
    }
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


def gesture_performance_by_participant(predictions, true_labels, participant_ids, 
                                        unique_participants, unique_gestures):
    """
    Calculate performance metrics for each participant and gesture
    
    Returns:
    - Dictionary with performance for each participant and gesture
    """
    performance = {}
    
    for participant in unique_participants:
        participant_mask = np.array(participant_ids) == participant
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


def main_training_pipeline(data_splits, 
                           all_participants, test_participants, 
                           model_type='CNN', bs=32,  
                           num_epochs=30, criterion=nn.CrossEntropyLoss(), lr=0.001, 
                           use_weight_decay=True, weight_decay=0.01, config=None,
                           train_intra_cross_loaders=None):
    """
    Main training pipeline with comprehensive performance tracking
    
    Args:
    - df: Input DataFrame
    - all_participants: List of all participants
    - test_participants: List of participants to hold out
    - model_type: 'CNN' or 'RNN' (or the already created model object)
    - training_trials_per_gesture: Trials to use per gesture for training
    - num_epochs: Number of training epochs
    
    Returns:
    - Trained model, performance metrics
    """

    if config is not None:
        bs = config["batch_size"]
        #criterion = config["criterion"]  # This doesnt exist rn
        lr = config["learning_rate"]
        weight_decay = config["weight_decay"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Unique gestures and number of classes
    if train_intra_cross_loaders is not None:
        # Hardcoding cause I don't wanna deal with dataloader extraction
        unique_gestures = list(range(10))
        num_classes = 10
        input_dim = 80

        train_pids = [pid for pid in all_participants if pid not in test_participants]
        intra_pids = train_pids
        cross_pids = test_participants
    else:
        unique_gestures = np.unique(data_splits['train']['labels'])
        num_classes = len(unique_gestures)
        input_dim = data_splits['train']['feature'].shape[1]

        train_pids = data_splits['train']['participant_ids']
        intra_pids = data_splits['intra_subject_test']['participant_ids']
        cross_pids = data_splits['cross_subject_test']['participant_ids']
    
    # Select model
    if isinstance(model_type, str):
        if model_type == 'CNN':
            if config is None:
                model = CNNModel(input_dim, num_classes).to(device)
            else:
                model = CNNModel_3layer(config, input_dim=80, num_classes=10)
        elif model_type == 'RNN':
            model = RNNModel(input_dim, num_classes).to(device)
    else:
        model = model_type

    if train_intra_cross_loaders is None:
        # Datasets and loaders
        train_dataset = GestureDataset(
            data_splits['train']['feature'], 
            data_splits['train']['labels']
        )
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        
        # INTRA SUBJECT
        intra_test_dataset = GestureDataset(
            data_splits['intra_subject_test']['feature'], 
            data_splits['intra_subject_test']['labels'])
        intra_test_loader = DataLoader(intra_test_dataset, batch_size=bs, shuffle=False)

        # CROSS SUBJECT
        cross_test_dataset = GestureDataset(
            data_splits['cross_subject_test']['feature'], 
            data_splits['cross_subject_test']['labels'])
        cross_test_loader = DataLoader(cross_test_dataset, batch_size=bs, shuffle=False)
    else:
        train_loader = train_intra_cross_loaders[0]
        intra_test_loader = train_intra_cross_loaders[1]
        cross_test_loader = train_intra_cross_loaders[2]
    
    # Loss and optimizer
    optimizer = get_optimizer(model, lr=lr, use_weight_decay=use_weight_decay, weight_decay=weight_decay)
    
    # Training
    train_loss_log = []
    intra_test_loss_log = []
    cross_test_loss_log = []
    for epoch in range(num_epochs):
        train_loss_log.append(train_model(model, train_loader, optimizer))
        intra_test_loss_log.append(evaluate_model(model, intra_test_loader)['loss'])
        cross_test_loss_log.append(evaluate_model(model, cross_test_loader)['loss'])
    
    # Evaluation --> One-off final results! Gets used in gesture_performance_by_participant
    train_results = evaluate_model(model, train_loader)
    intra_test_results = evaluate_model(model, intra_test_loader)
    cross_test_results = evaluate_model(model, cross_test_loader)

    if train_intra_cross_loaders is None:
        # This isn't working and I don't have time to fix it
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
        'train_accuracy': train_results['accuracy'],
        'intra_test_accuracy': intra_test_results['accuracy'],
        'cross_test_accuracy': cross_test_results['accuracy'], 
        'train_loss_log': train_loss_log,
        'intra_test_loss_log': intra_test_loss_log,
        'cross_test_loss_log': cross_test_loss_log
    }


def fine_tune_model(finetuned_model, fine_tune_loader, test_loader=None, num_epochs=20, lr=0.00001, criterion=nn.CrossEntropyLoss(), 
                    use_weight_decay=True, weight_decay=0.05):
    """
    Fine-tune the base model on a small subset of data
    
    Args:
    - base_model: Pretrained model to fine-tune
    - fine_tune_data: Dictionary with features and labels for fine-tuning
    - num_epochs: Number of fine-tuning epochs
    
    Returns:
    - Fine-tuned model
    """

    # Extract the original pretrained model weights since finetuning happens in place
    frozen_base_model_state = copy.deepcopy(finetuned_model.state_dict())

    finetuned_model.train()  # Ensure the model is in training mode
    # Loss and optimizer (with lower learning rate for fine-tuning)
    optimizer = get_optimizer(finetuned_model, lr=lr, use_weight_decay=use_weight_decay, weight_decay=weight_decay)
    # Fine-tuning
    train_loss_log = []
    test_loss_log = []
    for epoch in range(num_epochs):
        train_loss_log.append(train_model(finetuned_model, fine_tune_loader, optimizer))
        if test_loader is not None:
            test_loss_log.append(evaluate_model(finetuned_model, test_loader)['loss'])

    original_model = finetuned_model.__class__(input_dim=finetuned_model.input_dim, num_classes=finetuned_model.num_classes)  
    original_model.load_state_dict(frozen_base_model_state)  # Load pretrained weights into the new model
    assert(not finetuned_model == original_model)

    return finetuned_model, original_model, train_loss_log, test_loss_log


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


def log_performance(results, log_dir='ELEC573_Proj\\results\\performance_logs', base_filename='model_performance'):
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
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_filename = f"{timestamp}_{base_filename}.txt"
    log_path = os.path.join(log_dir, log_filename)
    
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

