import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import *
import copy


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
# BASIC MODELS
######################################

class RNNModel(nn.Module):
    def __init__(self, input_dim, num_classes, rnn_type="LSTM", hidden_size=64, num_layers=2, use_batch_norm=False, dropout_rate=0.5):
        super().__init__()
        self.rnn_type = rnn_type
        self.rnn = {
            "LSTM": nn.LSTM,
            "GRU": nn.GRU
        }.get(rnn_type, None)(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        if self.rnn is None:
            raise ValueError("Invalid rnn_type. Choose 'LSTM' or 'GRU'.")
        
        self.bn = nn.BatchNorm1d(hidden_size) if use_batch_norm else nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)  # (batch_size, seq_length, 1)
        _, hidden = self.rnn(x)
        x = hidden[0][-1] if self.rnn_type == "LSTM" else hidden[-1]
        x = self.bn(x)
        return self.fc(x)


class CNNModel(nn.Module):
    def __init__(self, input_dim, num_classes, use_batch_norm=False, dropout_rate=0.5):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            get_conv_block(1, 32, kernel_size=3, stride=1, padding=1, maxpool=2, use_batch_norm=use_batch_norm),
            get_conv_block(32, 64, kernel_size=3, stride=1, padding=1, maxpool=2, use_batch_norm=use_batch_norm),
        )
        flattened_size = calculate_flattened_size(input_dim, self.conv_blocks)
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, seq_length)
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


######################################
# DYNAMIC MODELS
######################################

class DynamicCNNModel(nn.Module):
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


class GenMomonaNet(nn.Module):
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
        self.bs = config['batch_size']
        self.nc = config['num_channels']
        self.sl = config['sequence_length']

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = self.nc
        for out_channels, kernel_size, stride in config['conv_layers']:
            self.conv_layers.append(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
            )
            in_channels = out_channels  # Update in_channels for the next layer

        # Max pooling
        self.pooling_layers = config.get('pooling_layers', [True] * len(config['conv_layers']))  # Default to all True
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)

        # Dense layer between CNN and LSTM
        self.use_dense_cnn_lstm = config.get('use_dense_cnn_lstm', False)  # Default to False
        if self.use_dense_cnn_lstm:
            self.dense_cnn_lstm = nn.Linear(in_channels, in_channels)  # Transform CNN output to LSTM input

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=in_channels,  # Input size is the output channels of the last conv layer
            hidden_size=config['lstm_hidden_size'],
            num_layers=config['lstm_num_layers'],
            batch_first=True,
            dropout=config['lstm_dropout']
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
            self.fc_layers.append(nn.Linear(in_features, out_features))
            in_features = out_features  # Update in_features for the next layer

        # Output layer
        self.output_layer = nn.Linear(in_features, config['num_classes'])

        # Activation function
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Reshape input to (batch_size, num_channels, sequence_length)
        x = x.view(self.bs, self.nc, self.sl)
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


class MomonaNet(nn.Module):
    #https://github.com/my-13/biosignals-gesture-analysis/blob/main/2024_UIST_CodeOcean/algorithms1.py
    def __init__(self, config):
        hidden_size = 12
        tlen = 62  # TODO: Pass this in as a param? Or dynamically calc it?
        super().__init__()

        self.bs = config['batch_size']
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
        x = x.view(self.bs, self.nc, self.sl)
        x = self.pool(self.conv1(x)) # CNN layer + pooling
        x = torch.swapaxes(x, 1, 2)
        x = self.relu(self.dense(x))
        x, _ = self.lstm(x) # 2 LSTM layers
        x = self.flatten(x) 
        logits = self.linear_relu_stack(x) # FFNN/dense layer for classification
        return logits


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
