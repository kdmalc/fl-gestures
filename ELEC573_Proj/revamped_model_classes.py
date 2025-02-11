import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import *
import copy
from gesture_dataset_classes import *
from collections import deque


class SmoothedEarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, smoothing_window=5):
        self.patience = patience
        self.min_delta = min_delta
        self.smoothing_window = smoothing_window
        self.best_loss = np.inf
        self.num_bad_epochs = 0
        self.loss_buffer = deque(maxlen=smoothing_window)

    def __call__(self, current_loss):
        self.loss_buffer.append(current_loss)
        smoothed_loss = np.mean(self.loss_buffer)

        if smoothed_loss < self.best_loss - self.min_delta:
            self.best_loss = smoothed_loss
            self.num_bad_epochs = 0  # Reset patience if improvement
        else:
            self.num_bad_epochs += 1  # Increment if no improvement

        return self.num_bad_epochs >= self.patience  # Stop if patience exceeded


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False
    

def select_model(model_type, config):  #, device="cpu", input_dim=16, num_classes=10):
    if isinstance(model_type, str):
        if model_type == 'CNN':  # 2D
            # Not sure if this is working, not really using this one anymore
            model = DynamicCNNModel(config, input_dim=80, num_classes=10)
        elif model_type == 'ELEC573Net':  # 2D:
            model = ELEC573Net(config)
        elif model_type == 'HybridCNNLSTM':  # 3D
            model = HybridCNNLSTM()  # TODO: This needs to be modified to take input sizes smh
        elif model_type == 'CRNN':  # 3D
            model = CRNN(input_channels=80, window_size=1, num_classes=10)
        elif model_type == 'EMGHandNet':  # 4D
            model = EMGHandNet(input_channels=80, num_classes=10)
        elif model_type == 'MomonaNet':  # 2D
            model = MomonaNet(config)  # Arch is hardcoded in but it'e fine
            # Overwriting... basically hardcoding these in...
            ## There's a better way to do this (since you could use the default vals)
            #sequence_length = 1
            #time_steps = 64
            # Nvm just hardcode the reshape...
            ## It really ought to be 3 tho... but I have to reshape no matter what I think
        elif model_type == "DynamicMomonaNet":  # 2D
            model = DynamicMomonaNet(config) # Config should specify everything about arch?
        else:
            raise ValueError(f"{model_type} not recognized.")
    else:
        model = model_type

    return model


def select_dataset_class(model_str):
    if model_str in ["CNN",  "MomonaNet", "DynamicMomonaNet", "ELEC573Net"]:
        return GestureDataset  # This is 2D
    elif model_str in ["RNN", "HybridCNNLSTM", "CRNN"]:
        return GestureDataset_3D
    elif model_str in ["EMGHandNet"]:
        return GestureDataset_4D
    else:
        raise ValueError


# Utility functions for modular design
def get_conv_block(in_channels, out_channels, kernel_size, stride, padding, maxpool, use_batch_norm, activation=nn.ReLU()):
    """Create a convolutional block with optional batch normalization and max pooling."""
    layers = [nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)]
    if use_batch_norm:
        layers.append(nn.BatchNorm1d(out_channels))
    layers.append(activation)
    if maxpool > 1:
        layers.append(nn.MaxPool1d(maxpool))
    return nn.Sequential(*layers)


def calculate_flattened_size(input_dim, layers):
    """Dynamically calculate the flattened size of a convolutional block."""
    with torch.no_grad():
        test_input = torch.randn(1, 1, input_dim)  # Simulate a single input
        test_output = layers(test_input)
    return test_output.view(1, -1).size(1)


######################################
# FIXED MODELS
######################################


class ELEC573Net(nn.Module):
    def __init__(self, config):
        super(ELEC573Net, self).__init__()
        self.input_dim = config["num_channels"]
        self.num_classes = config["num_classes"]
        self.config = config

        # Activation and Pooling
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(config["maxpool"])
        self.softmax = nn.Softmax(dim=1)
        
        # Convolutional Layers
        self.conv1 = nn.Conv1d(1, config["conv_layers"][0][0], 
                               kernel_size=config["conv_layers"][0][1], 
                               stride=config["conv_layers"][0][2], 
                               padding=config["padding"])
        self.bn1 = nn.BatchNorm1d(config["conv_layers"][0][0]) if config["use_batchnorm"] else nn.Identity()
        
        self.conv2 = nn.Conv1d(config["conv_layers"][0][0], config["conv_layers"][1][0], 
                               kernel_size=config["conv_layers"][1][1], 
                               stride=config["conv_layers"][1][2],
                               padding=config["padding"])
        self.bn2 = nn.BatchNorm1d(config["conv_layers"][1][0]) if config["use_batchnorm"] else nn.Identity()
        
        self.conv3 = nn.Conv1d(config["conv_layers"][1][0], config["conv_layers"][2][0], 
                               kernel_size=config["conv_layers"][2][1], 
                               stride=config["conv_layers"][2][2], 
                               padding=config["padding"])
        self.bn3 = nn.BatchNorm1d(config["conv_layers"][2][0]) if config["use_batchnorm"] else nn.Identity()
        
        self.flattened_size = self.config["conv_layers"][2][0] * self.compute_final_length()

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flattened_size, config["fc_layers"][0])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc2 = nn.Linear(config["fc_layers"][0], self.num_classes)

    def compute_final_length(self):
            pool = self.config["maxpool"]
            P = self.config["padding"]
            L = self.input_dim  # Needs to be seq len, but for this model we have 1 input channel and treat the 80 features as our seq len...
            
            # Conv1 + MaxPool1
            L = ((L - self.config["conv_layers"][0][1] + 2*P) // self.config["conv_layers"][0][2]) + 1
            L = L // pool
            # Conv2 + MaxPool2
            L = ((L - self.config["conv_layers"][1][1] + 2*P) // self.config["conv_layers"][1][2]) + 1
            L = L // pool
            # Conv3 (no final maxpool)
            L = ((L - self.config["conv_layers"][2][1] + 2*P) // self.config["conv_layers"][2][2]) + 1
            return L

    def forward(self, x):        
        #print(f"x start shape: {x.shape}")
        x = x.unsqueeze(1)  # Reshape input to (batch_size, 1, sequence_length)
        #print(f"x after unsqueezing: {x.shape}")

        x = self.conv1(x)
        #print(f"After conv1: {x.shape}")
        x = self.bn1(x)
        #print(f"After bn1: {x.shape}")
        x = self.relu(x)
        #print(f"After relu: {x.shape}")
        x = self.maxpool(x)
        #rint(f"After maxpool: {x.shape}")

        x = self.conv2(x)
        #print(f"After conv2: {x.shape}")
        x = self.bn2(x)
        #print(f"After bn2: {x.shape}")
        x = self.relu(x)
        #print(f"After relu: {x.shape}")
        x = self.maxpool(x)
        #print(f"After maxpool: {x.shape}")

        x = self.conv3(x)
        #print(f"After conv3: {x.shape}")
        x = self.bn3(x)
        #print(f"After bn3: {x.shape}")
        x = self.relu(x)
        #print(f"After relu: {x.shape}")
        # This last maxpool shrinks it down too much so turn it off
        #x = self.maxpool(x)  

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
        # TODO: ought to remove softmax, but original version had it so keep it for now...
        x = self.softmax(x)
        #print(f"After softmax: {x.shape}")
        return x


class MomonaNet(nn.Module):
    #https://github.com/my-13/biosignals-gesture-analysis/blob/main/2024_UIST_CodeOcean/algorithms1.py
    def __init__(self, config):
        hidden_size = 12
        tlen = 62  # TODO: Pass this in as a param? Or dynamically calc it?
        super().__init__()

        #self.bs = config['batch_size']
        self.nc = config['num_channels']
        self.sl = config['sequence_length']

        # Replacing 88 everywhere with 16 (88 is 72IMU + 16EMG)
        #CNN
        self.conv1 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        # Dense layer connecting CNN to LSTM
        self.dense = nn.Linear(16,16)
        self.relu = nn.ReLU()
        # LSTM
        self.lstm = nn.LSTM(input_size=16, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.8)
        self.flatten = nn.Flatten()

        # Fully connected layers for classification
        self.linear_relu_stack = nn.Sequential( 
            nn.Linear(int(hidden_size*tlen), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x):
        #x = x.view(self.bs, self.nc, self.sl)
        # Dynamically infer batch size from input tensor
        batch_size = x.size(0)  # Get the batch size from the input tensor
        # Reshape input to (batch_size, num_channels, sequence_length)
        x = x.view(batch_size, self.nc, self.sl)  # Use dynamic batch size
        x = self.pool(self.conv1(x)) # CNN layer + pooling
        x = torch.swapaxes(x, 1, 2)
        x = self.relu(self.dense(x))
        x, _ = self.lstm(x) # 2 LSTM layers
        x = self.flatten(x) 
        logits = self.linear_relu_stack(x) # FFNN/dense layer for classification
        return logits
    

######################################
# DYNAMIC MODELS
######################################

class DynamicCNNModel(nn.Module):
    # THIS IS NOT UP TO DATE

    def __init__(self, config, input_dim, num_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(*[
            get_conv_block(
                config["conv_layer_sizes"][i - 1] if i > 0 else 1,
                config["conv_layer_sizes"][i],
                kernel_size=config["kernel_size"],
                stride=config["stride"],
                padding=config["padding"],
                maxpool=config["maxpool"],
                use_batch_norm=config["use_batchnorm"]
            ) for i in range(config["num_conv_layers"])
        ])
        flattened_size = calculate_flattened_size(input_dim, self.conv_layers)
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, seq_length)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DynamicMomonaNet(nn.Module):
    def __init__(self, config):
        """
        Generalized MomonaNet model.
        https://github.com/my-13/biosignals-gesture-analysis/blob/main/2024_UIST_CodeOcean/algorithms1.py

        Args:
            config (dict): Configuration dictionary containing the following keys:
                - batch_size (int): Batch size.
                - num_channels (int): Number of input channels.
                - sequence_length (int): Length of the input sequence.
                - conv_layers (list of tuples): List of tuples specifying convolutional layers.
                    Each tuple should contain (out_channels, kernel_size, stride).
                - pooling_layers (list of bool): List specifying whether to apply max pooling after each conv layer.
                - use_dense_cnn_lstm (bool): Whether to use a dense layer between CNN and LSTM.
                - lstm_hidden_size (int): Hidden size for LSTM layers.
                - lstm_num_layers (int): Number of LSTM layers.
                - lstm_dropout (float): Dropout probability for LSTM layers.
                - fc_layers (list of int): List of integers specifying the sizes of fully connected layers.
                - num_classes (int): Number of output classes.
        """
        super().__init__()

        self.config = config  # For debugging
        #self.bs = config['batch_size']  # Dynamically infer bs from the input, so model isn't locked in to one bs
        self.nc = config['num_channels']
        self.sl = config['sequence_length']
        self.cnn_dropout = config['cnn_dropout']
        self.dense_cnnlstm_dropout = config['dense_cnnlstm_dropout']
        self.fc_dropout = config['fc_dropout']
        self.use_dense_cnn_lstm = config["use_dense_cnn_lstm"]

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = self.nc
        for out_channels, kernel_size, stride in config['conv_layers']:
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride),
                nn.Dropout(self.cnn_dropout) if self.cnn_dropout > 0 else nn.Identity()
            ))
            in_channels = out_channels  # Update in_channels for the next layer

        # Max pooling
        self.pooling_layers = config.get('pooling_layers', [True] * len(config['conv_layers']))  # Default to all True
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)

        # Dense layer between CNN and LSTM
        if self.use_dense_cnn_lstm:
            self.dense_cnn_lstm = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.Dropout(self.dense_cnnlstm_dropout) if self.dense_cnnlstm_dropout > 0 else nn.Identity()
            )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=in_channels,  # Input size is the output channels of the last conv layer
            hidden_size=config['lstm_hidden_size'],
            num_layers=config['lstm_num_layers'],
            batch_first=True,
            dropout=config['lstm_dropout'] if config['lstm_num_layers'] > 1 else 0
        )

        # Calculate the sequence length after convolutions and pooling
        tlen = self.sl
        for i, (_, kernel_size, stride) in enumerate(config['conv_layers']):
            tlen = (tlen - kernel_size) // stride + 1
            if self.pooling_layers[i]:
                tlen = (tlen - 2) // 1 + 1  # Account for max pooling

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        in_features = config['lstm_hidden_size'] * tlen
        for out_features in config['fc_layers']:
            self.fc_layers.append(nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.Dropout(self.fc_dropout) if self.fc_dropout > 0 else nn.Identity()
            ))
            in_features = out_features

        # Output layer
        self.output_layer = nn.Linear(in_features, config['num_classes'])

        # Activation function
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Reshape input to (batch_size, num_channels, sequence_length)
        #x = x.view(self.bs, self.nc, self.sl)
        # Dynamically infer batch size from input tensor
        batch_size = x.size(0)  # Get the batch size from the input tensor
        # Reshape input to (batch_size, num_channels, sequence_length)
        x = x.view(batch_size, self.nc, self.sl)  # Use dynamic batch size
        # Apply convolutional layers
        for i, conv in enumerate(self.conv_layers):
            x = self.relu(conv(x))
            if self.pooling_layers[i]:  # Apply max pooling only if specified
                x = self.pool(x)
        # Swap axes for LSTM input (batch_size, sequence_length, num_channels)
        x = torch.swapaxes(x, 1, 2)
        # Apply dense layer between CNN and LSTM (if enabled)
        if self.use_dense_cnn_lstm:
            x = self.dense_cnn_lstm(x)
        # Apply LSTM layers
        x, _ = self.lstm(x)
        # Flatten the output for fully connected layers
        x = self.flatten(x)
        # Apply fully connected layers
        for fc in self.fc_layers:
            x = self.relu(fc(x))
        # Output layer
        logits = self.output_layer(x)
        return logits
    

######################################
# NOT VALIDATED
######################################


class EMGHandNet(nn.Module):
    def __init__(self, input_channels, num_classes, F1=64, F2=128, NL=128):
        """
        Args:
        - input_channels (int): Number of input channels (NC).
        - num_classes (int): Number of output classes for the classification task.
        - F1 (int): Number of filters in the final CNN output.
        - F2 (int): Flattened feature vector size.
        - NL (int): Number of LSTM units in each Bi-LSTM layer.
        """
        super(EMGHandNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.NL = NL
        
        # CNN layers
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, F1, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(F1, F1, kernel_size=3, stride=1, padding=1)
        
        # Bi-LSTM layers
        self.bi_lstm1 = nn.LSTM(F1, NL, bidirectional=True, batch_first=True)
        self.bi_lstm2 = nn.LSTM(2 * NL, NL, bidirectional=True, batch_first=True)
        
        # Dense layers
        self.fc1 = nn.Linear(2 * NL, F2)
        self.fc2 = nn.Linear(F2, num_classes)
        
    def forward(self, x):
        batch_size, SL, TS, NC = x.size()  # SL: Sequence length, TS: Time steps, NC: Channels
        x = x.view(-1, TS, NC).permute(0, 2, 1)  # Reshape to (SL * batch_size, NC, TS)
        
        # CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Flatten along the temporal dimension
        x = x.mean(dim=-1)  # Output size: (SL * batch_size, F1)
        x = x.view(batch_size, SL, -1)  # Reshape back to (batch_size, SL, F1)
        
        # Bi-LSTM layers
        x, _ = self.bi_lstm1(x)
        x, _ = self.bi_lstm2(x)  # Output size: (batch_size, SL, 2 * NL)
        
        # Flatten and process with dense layers
        x = x.view(-1, 2 * self.NL)  # Flatten: (batch_size * SL, 2 * NL)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Reshape output to (batch_size, SL, num_classes), then average over SL
        x = x.view(batch_size, SL, -1)
        x = x.mean(dim=1)  # Average over sequence length SL
        
        # TODO: Swap from softmax to ReLU if we are using CrossEntropyLoss?
        return F.softmax(x, dim=-1) 


class CRNN(nn.Module):
    def __init__(self, input_channels, num_classes, lstm_size_multiplier=3):
        """
        Args:
        - input_channels (int): Number of input channels (C).
        - num_classes (int): Number of output classes for classification.
        - lstm_size_multiplier (int): Multiplier to determine the size of LSTM hidden units based on input channels.
        """
        super(CRNN, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(input_channels, input_channels * 2, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv1d(input_channels * 2, input_channels * 4, kernel_size=2, stride=1, padding=1)
        self.conv3 = nn.Conv1d(input_channels * 4, input_channels * 8, kernel_size=2, stride=1, padding=1)
        
        # LSTM layers
        lstm_size = input_channels * lstm_size_multiplier
        self.lstm1 = nn.LSTM(input_channels * 8, lstm_size, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_size, lstm_size, batch_first=True)
        self.lstm3 = nn.LSTM(lstm_size, lstm_size, batch_first=True)
        
        # Dropout layers for LSTM
        self.dropout1 = nn.Dropout(p=0.8)
        self.dropout2 = nn.Dropout(p=0.8)
        self.dropout3 = nn.Dropout(p=0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Gradient clipping
        self.clip_value = 5.0
    
    def forward(self, x):
        batch_size, w, C = x.size()
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, input_channels, window_size)
        
        # CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  # Output size: (batch_size, C*8, w)
        
        # Reshape for LSTM layers
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, w, C*8)
        
        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        
        # Take the last output of the LSTM sequence
        x = x[:, -1, :]  # Output size: (batch_size, lstm_size)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Apply softmax for classification
        ## TODO: Switch from softmax to ReLU?
        return F.softmax(x, dim=-1)
    
    def clip_gradients(self):
        """
        Clip gradients to prevent exploding gradients.
        """
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-self.clip_value, self.clip_value)


class HybridCNNLSTM(nn.Module):
    def __init__(self, seq_len=64, num_channels=16):
        super(HybridCNNLSTM, self).__init__()
        self.seq_len = seq_len
        self.num_channels = num_channels

        # CNN Branch
        self.cnn_branch = nn.Sequential(
            nn.BatchNorm2d(1),  # Input shape: (batch_size, 1, seq_len, num_channels)
            nn.Conv2d(1, 64, kernel_size=(5, 5), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 64, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 1)),
            nn.ReLU()
        )

        # Dynamically compute flattened CNN features
        dummy_input = torch.zeros(1, 1, self.seq_len, self.num_channels)
        dummy_output = self.cnn_branch(dummy_input)
        flattened_size = dummy_output.numel()

        # CNN Fully Connected Layers
        self.cnn_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
        )

        # LSTM Branch
        self.lstm = nn.LSTM(input_size=self.num_channels, hidden_size=256, num_layers=3, batch_first=True, dropout=0.5)
        self.lstm_fc = nn.Linear(256, 128)

        # Fusion and Classification
        self.fusion_fc = nn.Sequential(
            nn.Linear(128 + 128, 128),  # Fusion of CNNFeat and LSTMFeat
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        # Example had a seq len of 300 (was that also what the paper had?)
        x = x.to(torch.bfloat16)  # Ensure input tensor is in BFloat16
        #features = features.to(torch.bfloat16)
        #labels = labels.to(torch.bfloat16)
        batch_size, sequence_len, features = x.size()

        # CNN Branch
        cnn_input = x.unsqueeze(1).to(torch.bfloat16)  # Shape: (batch_size, 1, 64, 16)
        cnn_features = self.cnn_branch(cnn_input)  # Shape: (batch_size, ?, ?, 4)
        cnn_features = self.cnn_fc(cnn_features)  # Shape: (batch_size, 128)

        # LSTM Branch
        lstm_input = x  # Shape: (batch_size, 300, 16)
        lstm_out, _ = self.lstm(lstm_input)  # Shape: (batch_size, 300, 256)
        lstm_features = self.lstm_fc(lstm_out[:, -1, :])  # Take the last time step, shape: (batch_size, 128)

        # Feature Fusion
        hybrid_features = torch.cat((cnn_features, lstm_features), dim=1)  # Shape: (batch_size, 256)
        output = self.fusion_fc(hybrid_features)  # Shape: (batch_size, 64)

        return output
