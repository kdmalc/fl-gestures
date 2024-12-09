import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime


class GestureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class CNNModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * (input_dim // 4), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Reshape input to (batch_size, 1, sequence_length)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


class RNNModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Reshape input to (batch_size, sequence_length, 1)
        x = x.unsqueeze(-1)
        _, (hidden, _) = self.rnn(x)
        x = hidden[-1]  # Take the last layer's hidden state
        return self.fc(x)


def prepare_data(df, feature_column, target_column, participants, test_participants, 
                 trials_per_gesture=8):
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
    
    train_data = {
        'features': [],
        'labels': [],
        'participant_ids': []
    }
    test_data = {
        'features': [],
        'labels': [],
        'participant_ids': []
    }
    
    for participant in participants:
        participant_data = df[df['Participant'] == participant]
        
        # Group by gesture
        gesture_groups = participant_data.groupby(target_column)
        
        # Separate data
        if participant in train_participants:
            target_dict = train_data
            max_trials = trials_per_gesture
        else:
            target_dict = test_data
            max_trials = None
        
        for gesture, group in gesture_groups:
            # Randomize and select trials
            group_features = np.array([x.flatten() for x in group[feature_column]])
            group_labels = group[target_column].values
            
            if max_trials is not None:
                # Randomly select specified number of trials
                indices = np.random.choice(
                    len(group_features), 
                    min(max_trials, len(group_features)), 
                    replace=False
                )
                group_features = group_features[indices]
                group_labels = group_labels[indices]
            
            target_dict['features'].extend(group_features)
            target_dict['labels'].extend(group_labels)
            target_dict['participant_ids'].extend([participant] * len(group_labels))
    
    return {
        'train': {
            'features': np.array(train_data['features']),
            'labels': np.array(train_data['labels']),
            'participant_ids': train_data['participant_ids']
        },
        'test': {
            'features': np.array(test_data['features']),
            'labels': np.array(test_data['labels']),
            'participant_ids': test_data['participant_ids']
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


def main_training_pipeline(df, feature_column, target_column, 
                           all_participants, test_participants, 
                           model_type='CNN', 
                           training_trials_per_gesture=8, 
                           num_epochs=50, criterion = nn.CrossEntropyLoss(), lr=0.001):
    """
    Main training pipeline with comprehensive performance tracking
    
    Args:
    - df: Input DataFrame
    - feature_column: Column with feature vectors
    - target_column: Column with encoded gesture labels
    - all_participants: List of all participants
    - test_participants: List of participants to hold out
    - model_type: 'CNN' or 'RNN'
    - training_trials_per_gesture: Trials to use per gesture for training
    - num_epochs: Number of training epochs
    
    Returns:
    - Trained model, performance metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare data
    data_splits = prepare_data(
        df, feature_column, target_column, 
        all_participants, test_participants, 
        training_trials_per_gesture
    )
    
    # Unique gestures and number of classes
    unique_gestures = np.unique(data_splits['train']['labels'])
    num_classes = len(unique_gestures)
    input_dim = data_splits['train']['features'].shape[1]
    
    # Select model
    if model_type == 'CNN':
        model = CNNModel(input_dim, num_classes).to(device)
    else:
        model = RNNModel(input_dim, num_classes).to(device)
    
    # Datasets and loaders
    train_dataset = GestureDataset(
        data_splits['train']['features'], 
        data_splits['train']['labels']
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    test_dataset = GestureDataset(
        data_splits['test']['features'], 
        data_splits['test']['labels']
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, optimizer)
    
    # Evaluation
    train_results = evaluate_model(model, train_loader)
    test_results = evaluate_model(model, test_loader)
    
    # Performance by participant and gesture
    train_performance = gesture_performance_by_participant(
        train_results['predictions'], 
        train_results['true_labels'], 
        data_splits['train']['participant_ids'], 
        all_participants, 
        unique_gestures
    )
    
    test_performance = gesture_performance_by_participant(
        test_results['predictions'], 
        test_results['true_labels'], 
        data_splits['test']['participant_ids'], 
        test_participants, 
        unique_gestures
    )
    
    return {
        'model': model,
        'train_performance': train_performance,
        'test_performance': test_performance,
        'train_accuracy': train_results['accuracy'],
        'test_accuracy': test_results['accuracy']
    }


def fine_tune_model(base_model, fine_tune_data, num_epochs=20, lr=0.0001, criterion=nn.CrossEntropyLoss()):
    """
    Fine-tune the base model on a small subset of data
    
    Args:
    - base_model: Pretrained model to fine-tune
    - fine_tune_data: Dictionary with features and labels for fine-tuning
    - num_epochs: Number of fine-tuning epochs
    
    Returns:
    - Fine-tuned model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset and loader for fine-tuning
    fine_tune_dataset = GestureDataset(fine_tune_data['features'], fine_tune_data['labels'])
    fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=16, shuffle=True)
    
    # Loss and optimizer (with lower learning rate for fine-tuning)
    optimizer = optim.Adam(base_model.parameters(), lr=lr)
    
    # Fine-tuning
    for epoch in range(num_epochs):
        train_model(base_model, fine_tune_loader, optimizer)
    
    return base_model


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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_filename = f"{timestamp}_{save_filename}.png"
    fig_save_path = os.path.join(save_path, fig_filename)
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
    test_performance_heatmap = plot_gesture_performance(
        results['test_performance'], 
        'Testing Set - Gesture Performance by Participant',
        'testing_performance_heatmap'
    )
    
    if print_results:
        # Print detailed performance
        print_detailed_performance(results['train_performance'], 'Training')
        print_detailed_performance(results['test_performance'], 'Testing')
        
        # Additional overall metrics
        print("\nOverall Model Performance:")
        print(f"Training Accuracy: {results['train_accuracy']:.2%}")
        print(f"Testing Accuracy: {results['test_accuracy']:.2%}")


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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
            if participant in results['test_performance'].keys():
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
        
        # Repeat for Testing Set
        print("\n--- Testing Set Performance ---")
        for participant, gesture_performance in results['test_performance'].items():
            print(f"\nParticipant {participant}:")
            participant_accuracies = []
            
            for gesture, accuracy in sorted(gesture_performance.items()):
                print(f"  Gesture {gesture}: {accuracy:.2%}")
                participant_accuracies.append(accuracy)
            
            print(f"  Average Accuracy: {np.mean(participant_accuracies):.2%}")
        
        # Testing Set Summary
        test_accuracies = [acc for participant in results['test_performance'].values() for acc in participant.values()]
        print("\nTesting Set Summary:")
        print(f"Mean Accuracy: {np.mean(test_accuracies):.2%}")
        print(f"Accuracy Standard Deviation: {np.std(test_accuracies):.2%}")
        print(f"Minimum Accuracy: {np.min(test_accuracies):.2%}")
        print(f"Maximum Accuracy: {np.max(test_accuracies):.2%}")
        
        # Overall Model Performance
        print("\nOverall Model Performance:")
        print(f"Training Accuracy: {results['train_accuracy']:.2%}")
        print(f"Testing Accuracy: {results['test_accuracy']:.2%}")
    
    finally:
        # Restore stdout
        sys.stdout = sys.stdout.terminal
    
    return log_path

