import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.preprocessing import LabelEncoder

import time
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from sklearn.metrics import confusion_matrix
import seaborn as sns


def create_data_loader(data_df, dataset_obj, labels=None, num_rows_per_gesture=64, batch_size=32, shuffle_bool=True, drop_last_bool=True):
    '''
    - data_df should be 2D...
    - dataset_obj should be GestureDatasetAE (for autoencoders, eg only returns data no labels) or GestureDatasetClustering (for clustering, returns X and Y)
    '''

    if data_df.ndim == 2: # This assumes its shape is (num_gestures x num_rows_per_gesture, num_features)
        num_gestures = len(data_df) // num_rows_per_gesture
        assert len(data_df) % num_rows_per_gesture == 0, "The total number of rows is not a multiple of the number of rows per gesture."
        num_features = data_df.shape[1]
        X_3D = data_df.to_numpy().reshape(num_gestures, num_rows_per_gesture, num_features)
    elif data_df.ndim == 3: # This assumes its shape is (num_gestures, num_rows_per_gesture, num_features)
        X_3D = data_df
    else:
        raise ValueError("Data df must be 2 or 3 dimensional")
    X_3DTensor = torch.tensor(X_3D, dtype=torch.float32)
    if labels is not None:
        dataset = dataset_obj(X_3DTensor, labels)
    else:
        # Autoencoder is implied if labels_df is None
        dataset = dataset_obj(X_3DTensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_bool, drop_last=drop_last_bool)
    

def plot_confusion_matrix(labels, preds, classes_plot_ticks_lst, dataset_name_str):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes_plot_ticks_lst, yticklabels=classes_plot_ticks_lst)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix for {dataset_name_str}')
    plt.show()
    

class GestureDatasetClustering(Dataset):
    def __init__(self, data_tensor, label_tensor):
        self.data = data_tensor
        self.labels = label_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y


class GestureLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.0):
        super(GestureLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Using the output of the last time step
        out = self.fc(out)
        return out


class GestureLSTMClassifier_Hidden(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.0):
        super(GestureLSTMClassifier_Hidden, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out[:, -1, :])  # Using the output of the last time step
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        # Allegedly should be called once per batch, in the training loop (NOT IN FORWARD)
        hidden_states = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cell_states = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden_states, cell_states


def train_LSTM_gesture_classifier(model, nn_train_loader, nn_test_loader, criterion=nn.CrossEntropyLoss(), optimizer_str='ADAM', lr=0.001, weight_decay=0.0, num_epochs=15, batch_size=32, use_hidden=True, print_loss=True, print_accuracy=True):
    if optimizer_str == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    print("Starting training!")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in nn_train_loader:
            #inputs, labels = inputs.to(model.fc.weight.device), labels.to(model.fc.weight.device)

            if use_hidden:
                hidden = model.init_hidden(inputs.size(0))
            
            model.zero_grad()

            if use_hidden:
                outputs, hidden = model(inputs, hidden)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        val_loss, val_accuracy = evaluate_model(model, nn_test_loader, criterion, use_hidden, batch_size)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(nn_train_loader):.4f}, Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    return model


def evaluate_model(model, data_loader, criterion, use_hidden, batch_size):
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(model.fc.weight.device), labels.to(model.fc.weight.device)

            if use_hidden:
                hidden = model.init_hidden(inputs.size(0))

            if use_hidden:
                outputs, hidden = model(inputs, hidden)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss /= len(data_loader)
    val_accuracy = 100 * correct_val / total_val
    return val_loss, val_accuracy


def prepare_LOUO_data_loaders(dataset, users, batch_size, seq_len=64):
    data_loaders = []
    
    # Convert Gesture_ID column to integer labels
    label_encoder = LabelEncoder()
    dataset['Gesture_ID'] = label_encoder.fit_transform(dataset['Gesture_ID'])
    num_gesture_classes = len(label_encoder.classes_)
    num_features = dataset.shape[-1] - 3 # -3 for the 3 rows of metadata
    
    for i in range(len(users)):
        test_user = users[i]
        train_users = np.delete(users, i)

        #print(test_user)
        #print(train_users)
        train_dataset = dataset[dataset['Participant'].isin(train_users)]
        train_input = torch.tensor(train_dataset.drop(columns=['Participant','Gesture_ID','Gesture_Num']).values.reshape(-1, seq_len, num_features), dtype=torch.float32)
        train_labels = torch.tensor(train_dataset['Gesture_ID'].iloc[::seq_len].values, dtype=torch.float32).long()
        test_dataset = dataset[dataset['Participant']==test_user]
        test_input = torch.tensor(test_dataset.drop(columns=['Participant','Gesture_ID','Gesture_Num']).values.reshape(-1, seq_len, num_features), dtype=torch.float32)
        test_labels = torch.tensor(test_dataset['Gesture_ID'].iloc[::seq_len].values, dtype=torch.float32).long()
        #print(train_input.shape)
        #print(train_labels.shape)
        #print()
        
        # Create TensorDataset
        train_dataset = TensorDataset(train_input, train_labels)
        test_dataset = TensorDataset(test_input, test_labels)
        
        # Create DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        data_loaders.append((train_loader, test_loader))
    
    return data_loaders


def split_users(dataset_df, train_ratio=0.7, test_ratio=0.15):
    '''Returns a list of strings for which users are train, val, and test'''
    #num_users = dataset.shape[0]
    pID_lst = dataset_df['Participant'].unique()
    num_users = len(pID_lst)
    indices = np.arange(num_users)
    np.random.shuffle(indices)
    
    train_end = int(train_ratio * num_users)
    val_end = train_end + int((1-train_ratio-test_ratio) * num_users)
    
    train_users_lst = pID_lst[indices[:train_end]]
    val_users_lst = pID_lst[indices[train_end:val_end]]
    test_users_lst = pID_lst[indices[val_end:]]
    
    return train_users_lst, val_users_lst, test_users_lst
    

def train_LSTM_gesture_classifier_ARCHIVE(model, nn_train_loader, nn_test_loader, criterion=nn.CrossEntropyLoss(), optimizer_str='ADAM', lr=0.001, num_epochs=15, batch_size=32, use_hidden=False, print_loss=True, plot_loss=False, print_accuracy=True, print_pre_rec_fscore=False, print_classification_report=False, save_model=False, model_save_dir=None, model_save_name=None):
    if save_model:
        # Could make these more robust by checking the dir exists and doing type checking (str)
        assert(model_save_dir is not None)
        assert(model_save_name is not None)
    if plot_loss:
        raise ValueError("plot_loss is not implemented yet")
    if save_model:
        raise ValueError("save_model is not implemented yet")
    
    print("Starting training!")
    start_time = time.time()

    if optimizer_str=='ADAM':
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        all_labels_train = []
        all_preds_train = []
        if use_hidden:
            hidden = model.init_hidden(batch_size)  # Initialize hidden states
    
        for i, (inputs, labels) in enumerate(nn_train_loader):
            model.zero_grad()
            inputs.detach_()
            
            # Forward pass
            if use_hidden:
                outputs, hidden = model(inputs, hidden)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Detach hidden state to prevent backpropagation through the entire sequence history
            if use_hidden:
                hidden = (hidden[0].detach(), hidden[1].detach())
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            all_labels_train.extend(labels.cpu().numpy())
            all_preds_train.extend(predicted.cpu().numpy())
        
        # Calculate training metrics for the epoch
        train_accuracy = 100 * correct_train / total_train
        if print_pre_rec_fscore:
            train_precision = precision_score(all_labels_train, all_preds_train, average='weighted')
            train_recall = recall_score(all_labels_train, all_preds_train, average='weighted')
            train_f1 = f1_score(all_labels_train, all_preds_train, average='weighted')
    
        # Validation step
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_labels_val = []
        all_preds_val = []
        with torch.no_grad():
            for inputs, labels in nn_test_loader:
                if use_hidden:
                    outputs, hidden = model(inputs, hidden)
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                all_labels_val.extend(labels.cpu().numpy())
                all_preds_val.extend(predicted.cpu().numpy())
        
        # Calculate validation metrics for the epoch
        val_accuracy = 100 * correct_val / total_val
        if print_pre_rec_fscore:
            val_precision = precision_score(all_labels_val, all_preds_val, average='weighted')
            val_recall = recall_score(all_labels_val, all_preds_val, average='weighted')
            val_f1 = f1_score(all_labels_val, all_preds_val, average='weighted')
    
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {running_loss/len(nn_train_loader):.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss/len(nn_test_loader):.4f}, '
              f'Val Accuracy: {val_accuracy:.2f}%')
        if print_pre_rec_fscore:
            print(f'Train Precision: {train_precision:.2f}, '
                  f'Train Recall: {train_recall:.2f}, '
                  f'Train F1 Score: {train_f1:.2f}, '
                  f'Val Precision: {val_precision:.2f}, '
                  f'Val Recall: {val_recall:.2f}, '
                  f'Val F1 Score: {val_f1:.2f}\n')
        
    # Print classification report for the validation set
    cls_report = classification_report(all_labels_val, all_preds_val)
    if print_classification_report:
        print("Classification Report:\n", cls_report)
    print(f"Training completed in {time.time() - start_time}")
    return model, cls_report, [all_labels_train, all_preds_train, all_labels_val, all_preds_val]
