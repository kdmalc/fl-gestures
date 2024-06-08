import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import time
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from sklearn.metrics import confusion_matrix
import seaborn as sns


def create_data_loader(df, dataset_obj, num_gestures, num_rows_per_gesture, num_features, batch_size, shuffle):
    '''
    dataset_obj should be GestureDatasetAE (for autoencoders, eg only returns data no labels) or GestureDatasetClustering (for clustering, returns X and Y)
    '''
    X_3D = df.to_numpy().reshape(num_gestures, num_rows_per_gesture, num_features)
    X_3DTensor = torch.tensor(X_3D, dtype=torch.float32)
    dataset = dataset_obj(X_3DTensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    

def plot_confusion_matrix(labels, preds, classes_plot_ticks_lst, dataset_name_str):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes_plot_ticks_lst, yticklabels=classes_plot_ticks_lst)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix for {dataset_name_str}')
    plt.show()


def create_dataloader(data_df, DatasetClass, batch_size=32, shuffle_bool=True, num_rows_per_gesture=64):
    num_gestures = len(data_df) // num_rows_per_gesture
    num_features = data_df.shape[1]
    assert len(data_df) % num_rows_per_gesture == 0, "The total number of rows is not a multiple of the number of rows per gesture."
    data_npy = data_df.to_numpy().reshape(num_gestures, num_rows_per_gesture, num_features)
    data_tensor = torch.tensor(data_npy, dtype=torch.float32)
    dataset = DatasetClass(data_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_bool)
    return data_loader
    

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
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GestureLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # This is init'ing the hidden states and cells to 0 each iter...
        hidden_states = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        cell_states = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (hidden_states, cell_states))
        out = self.fc(out[:, -1, :])
        return out


class GestureLSTMClassifier_Hidden(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GestureLSTMClassifier_Hidden, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, hidden):
        #self.init_hidden(x.size(0))
        
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # Using the output of the last time step
        return out, hidden

    def init_hidden(self, batch_size):
        hidden_states = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cell_states = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden_states, cell_states


def train_LSTM_gesture_classifier(model, nn_train_loader, nn_test_loader, criterion=nn.CrossEntropyLoss(), optimizer_str='ADAM', lr=0.001, num_epochs=15, batch_size=32, use_hidden=False, print_loss=True, plot_loss=False, print_accuracy=True, print_pre_rec_fscore=False, print_classification_report=False, save_model=False, model_save_dir=None, model_save_name=None):
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
