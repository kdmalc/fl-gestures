import torch
import torch.nn as nn
import torch.nn.functional as F


OLD_config = {
    "learning_rate": 0.0001,
    "batch_size": 32,
    "num_epochs": 50,
    "optimizer": "adam",
    "weight_decay": 1e-4,
    "dropout_rate": 0.3,
    "num_conv_layers": 3,
    "conv_layer_sizes": [32, 64, 128], 
    "kernel_size": 5,
    "stride": 2,
    "padding": 1,
    "maxpool": 1,
    "use_batchnorm": True
}

# Example configuration for 3 convolutional layers
dynamicCNN_config = {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "num_epochs": 50,
    "optimizer": "adam",
    "weight_decay": 1e-4,
    "num_conv_layers": 3,
    #
    "conv_layer_sizes": [16, 32, 64],  # Number of filters for each layer
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "maxpool": True,
    "use_batchnorm": True,
    "dropout_rate": 0.5
}

CRNN_config = {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "num_epochs": 50,
    "optimizer": "adam",
    "weight_decay": 1e-4, 
    "sequence_length": 2,
    "time_steps": 32
    }

HybridCNNLSTM_config = {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "num_epochs": 50,
    "optimizer": "adam",
    "weight_decay": 1e-4, 
    "sequence_length": 2,
    "time_steps": 32
    }

EMGHandNet_config = {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "num_epochs": 50,
    "optimizer": "adam",
    "weight_decay": 1e-4, 
    "sequence_length": 2,
    "time_steps": 32
    }

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


class CNNLSTMModel(nn.Module):
    def __init__(
        self,
        input_channels=1,
        num_classes=10,
        cnn_filters=[16, 32],
        kernel_size=(3, 3),
        pool_size=(2, 2),
        lstm_hidden_size=128,
        lstm_num_layers=2,
        time_steps=64,
        feature_size=16
    ):
        super(CNNLSTMModel, self).__init__()
        self.cnn_filters = cnn_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        # CNN layers
        cnn_layers = []
        in_channels = input_channels
        for out_channels in cnn_filters:
            cnn_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2))
            )
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))
            in_channels = out_channels
        self.cnn = nn.Sequential(*cnn_layers)
        # Calculate output shape from CNN layers
        cnn_output_time_steps = time_steps // (pool_size[0] ** len(cnn_filters))
        cnn_output_features = feature_size // (pool_size[1] ** len(cnn_filters))
        self.lstm_input_size = cnn_filters[-1] * cnn_output_features
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )
        # Fully connected layer
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass for CNN-LSTM model
        Args:
            x: Input tensor of shape (batch_size, input_channels, time_steps, feature_size)
        """
        # Pass through CNN layers
        x = self.cnn(x)
        # Reshape for LSTM layers
        batch_size, _, time_steps, features = x.size()
        x = x.permute(0, 2, 1, 3)  # (batch_size, time_steps, channels, features)
        x = x.reshape(batch_size, time_steps, -1)  # (batch_size, time_steps, channels * features)
        # Pass through LSTM layers
        x, _ = self.lstm(x)  # (batch_size, time_steps, lstm_hidden_size)
        x = x[:, -1, :]      # Take the output from the last time step (batch_size, lstm_hidden_size)
        # Pass through fully connected layer
        x = self.fc(x)       # (batch_size, num_classes)

        return x

## Example usage
#model = CNNLSTM(
#    input_channels=1,
#    num_classes=10,
#    cnn_filters=[16, 32],
#    kernel_size=(3, 3),
#    pool_size=(2, 2),
#    lstm_hidden_size=128,
#    lstm_num_layers=2,
#    time_steps=64,
#    feature_size=16
#)
#print(model)
#
## Simulate a batch of data with shape (batch_size, input_channels, time_steps, feature_size)
#dummy_input = torch.randn(32, 1, 64, 16)  # 32 samples, single channel, 64x16 input
#output = model(dummy_input)
#print(output.shape)  # Should output (32, 10) for 10 gesture classes


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
        """
        Forward pass for the EMGHandNet.
        
        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, SL, TS, NC).
        
        Returns:
        - torch.Tensor: Output probabilities for each class.
        """
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
        
        return F.softmax(x, dim=-1)

# Example usage
#model = EMGHandNet(input_channels=8, num_classes=6)  # Example: 8 input channels, 6 output classes
#print(model)


class CRNN(nn.Module):
    def __init__(self, input_channels, window_size, num_classes, lstm_size_multiplier=3):
        """
        Args:
        - input_channels (int): Number of input channels (C).
        - window_size (int): Width of the sliding window (w).
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
        """
        Forward pass for the C-RNN model.
        
        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, window_size, input_channels).
        
        Returns:
        - torch.Tensor: Output probabilities for each class.
        """

        # MODEL INPUT IS JUST (bs, 80) RN!
        ## We have no window size!
        ## The EMG feature engineering removed the time dimension, I forget oops
        ## So each trial is indeed just 1 row
        ## This is probably not optimal... not taking advantage of the sequence...
        if len(x.size())==2:
            x = x.unsqueeze(1)
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
        return F.softmax(x, dim=-1)
    
    def clip_gradients(self):
        """
        Clip gradients to prevent exploding gradients.
        """
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-self.clip_value, self.clip_value)

# Example usage
#model = CRNN(input_channels=8, window_size=100, num_classes=6)  # Example: 8 input channels, window size=100, 6 output classes
#print(model)


class HybridCNNLSTM(nn.Module):
    def __init__(self):
        super(HybridCNNLSTM, self).__init__()

        # CNN Branch for Spatial Feature Extraction
        self.cnn_branch = nn.Sequential(
            nn.BatchNorm2d(1),  # Input shape: (batch_size, 1, 300, 16)
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

        # Fully connected layers for CNN features
        self.cnn_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 74 * 4, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
        )

        # LSTM Branch for Temporal Feature Extraction
        self.lstm = nn.LSTM(input_size=16, hidden_size=256, num_layers=3, batch_first=True, dropout=0.5)
        self.lstm_fc = nn.Linear(256, 128)

        # Fusion and Classification
        self.fusion_fc = nn.Sequential(
            nn.Linear(128 + 128, 128),  # Fusion of CNNFeat and LSTMFeat
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        # MODEL INPUT IS JUST (bs, 80) RN!
        ## We have no window size!
        ## The EMG feature engineering removed the time dimension, I forget oops
        ## So each trial is indeed just 1 row
        ## This is probably not optimal... not taking advantage of the sequence...
        if len(x.size())==2:
            x = x.unsqueeze(1)
        # Example had a seq len of 300 (was that also what the paper had?)
        batch_size, sequence_len, features = x.size()

        # CNN Branch
        cnn_input = x.unsqueeze(1)  # Shape: (batch_size, 1, 300, 16)
        cnn_features = self.cnn_branch(cnn_input)  # Shape: (batch_size, 64, 74, 4)
        cnn_features = self.cnn_fc(cnn_features)  # Shape: (batch_size, 128)

        # LSTM Branch
        lstm_input = x  # Shape: (batch_size, 300, 16)
        lstm_out, _ = self.lstm(lstm_input)  # Shape: (batch_size, 300, 256)
        lstm_features = self.lstm_fc(lstm_out[:, -1, :])  # Take the last time step, shape: (batch_size, 128)

        # Feature Fusion
        hybrid_features = torch.cat((cnn_features, lstm_features), dim=1)  # Shape: (batch_size, 256)
        output = self.fusion_fc(hybrid_features)  # Shape: (batch_size, 64)

        return output

# Example usage
#model = HybridCNNLSTM()
#example_input = torch.randn(8, 300, 16)  # Batch size: 8, 300 frames, 16 features per frame
#output = model(example_input)
#print("Output shape:", output.shape)  # Expected shape: (8, 64)
