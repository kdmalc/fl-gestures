import torch
import torch.nn as nn


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


class CNNLSTM(nn.Module):
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
        super(CNNLSTM, self).__init__()
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
