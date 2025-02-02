import torch
import torch.nn as nn


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        torch.save(model.state_dict(), 'checkpoint.pt')
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')


######################################
# BASIC MODELS
######################################

class RNNModel(nn.Module):
    def __init__(self, input_dim, num_classes, 
                 rnn_type="LSTM", hidden_size=64, num_layers=2, 
                 use_batch_norm=False, dropout_rate=0.5):
        super(RNNModel, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.rnn_type = rnn_type

        # RNN Layer (LSTM/GRU configurable)
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError("Invalid rnn_type. Choose 'LSTM' or 'GRU'.")
        
        # Batch Norm for RNN Output
        self.bn = nn.BatchNorm1d(hidden_size) if self.use_batch_norm else nn.Identity()
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Reshape input to (batch_size, sequence_length, 1)
        x = x.unsqueeze(-1)
        
        if self.rnn_type == "LSTM":
            _, (hidden, _) = self.rnn(x)  # LSTM returns (output, (hidden_state, cell_state))
        else:
            _, hidden = self.rnn(x)  # GRU returns (output, hidden_state)
        
        # Take the last layer's hidden state
        x = hidden[-1]  # Shape: (batch_size, hidden_size)
        x = self.bn(x)  # Apply batch normalization if enabled
        return self.fc(x)


class CNNModel(nn.Module):
    def __init__(self, input_dim, num_classes, 
                 use_batch_norm=False, dropout_rate=0.5):
        super(CNNModel, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        #self.init_params = {"use_batch_norm": self.use_batch_norm, "dropout_rate": self.dropout_rate}
        
        # Convolutional Layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32) if self.use_batch_norm else nn.Identity()
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64) if self.use_batch_norm else nn.Identity()
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * (input_dim // 4), 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Activation and Pooling
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2)

    def forward(self, x):
        # Reshape input to (batch_size, 1, sequence_length)
        x = x.unsqueeze(1)
        
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Flatten and Fully Connected Layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
######################################
# MORE ADVANCED MODELS
######################################

## THIS ONE ISNT WORKING, THE ACTUAL SIZE IS ALWAYS SMALLER THAN THE CALCULATED (input*seq) SIZE
## Because input_dims was set to 100 instead of 80? Test later...
class DynamicCNNModel(nn.Module):
    def __init__(self, config, input_dim, num_classes):
        super(DynamicCNNModel, self).__init__()
        # This is redundant somewhat, but saving so I can access while debugging
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.config = config

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
            layers.append(nn.MaxPool1d(config["maxpool"]))
            torch_seq_len = torch.floor(torch.tensor((torch.floor(torch.tensor((seq_len + 2*config["padding"] - config["kernel_size"]) / config["stride"])) + 1) / config["maxpool"]))
            seq_len = (seq_len + 2 * config["padding"] - config["kernel_size"]) // config["stride"] + 1
            seq_len = seq_len // config["maxpool"]
            print(f"Conv layer {i}, in_channels: {in_channels}, out_channels: {out_channels}, seq_len: {seq_len}, torch_seq_len: {torch_seq_len}")
            in_channels = out_channels

        # Also saving for access during debugging:
        #self.seq_len = int(seq_len.item())
        self.seq_len = seq_len
        self.in_channels = in_channels

        # NOTE: These only create, not apply, the following funcs!
        ## They get applied in forward
        self.conv = nn.Sequential(*layers)
        self.fc1 = nn.Linear(self.in_channels * self.seq_len, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(config["dropout_rate"])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        x = x.unsqueeze(1)
        #print(f"After unsqueeze: {x.shape}")
        x = self.conv(x)
        #print(f"After conv: {x.shape}")
        x = x.view(x.size(0), -1)  # Flatten for FC layers
        #print(f"After flatten: {x.shape}")
        x = self.fc1(x)
        #print(f"After fc1: {x.shape}")
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        #print(f"After fc2: {x.shape}")
        x = self.softmax(x)
        return x
    
class CNNModel_2layer(nn.Module):
    def __init__(self, config, input_dim, num_classes):
        super(CNNModel_2layer, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.config = config

        # Activation and Pooling
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(config["maxpool"])
        self.softmax = nn.Softmax(dim=1)
        
        # Convolutional Layers
        self.conv1 = nn.Conv1d(1, config["conv_layer_sizes"][0], kernel_size=config["kernel_size"], stride=config["stride"], padding=config["padding"])
        self.bn1 = nn.BatchNorm1d(config["conv_layer_sizes"][0]) if config["use_batchnorm"] else nn.Identity()
        self.conv2 = nn.Conv1d(config["conv_layer_sizes"][0], config["conv_layer_sizes"][1], kernel_size=config["kernel_size"], stride=config["stride"], padding=config["padding"])
        self.bn2 = nn.BatchNorm1d(config["conv_layer_sizes"][1]) if config["use_batchnorm"] else nn.Identity()
        
        # Dynamically calculate flattened size
        test_input = torch.randn(1, 1, input_dim)
        # Run through conv layers to calculate final size
        with torch.no_grad():
            test_x = self.conv1(test_input)
            test_x = self.maxpool(test_x)
            test_x = self.conv2(test_x)
            #test_x = self.maxpool(test_x)
            flattened_size = test_x.view(1, -1).size(1)
            #print(f"flattened_size of test input: {flattened_size}")
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(config["dropout_rate"])
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        #print(f"Input x shape: {x.shape}")
        x = x.unsqueeze(1)  # Reshape input to (batch_size, 1, sequence_length)
        #print(f"After unsqueeze: {x.shape}")
        # Conv Block 1
        x = self.conv1(x)
        #print(f"After conv1: {x.shape}")
        x = self.bn1(x)
        #print(f"After bn1: {x.shape}")
        x = self.relu(x)
        #print(f"After relu: {x.shape}")
        x = self.maxpool(x)
        #print(f"After maxpool: {x.shape}")
        # Conv Block 2
        x = self.conv2(x)
        #print(f"After conv2: {x.shape}")
        x = self.bn2(x)
        #print(f"After bn2: {x.shape}")
        x = self.relu(x)
        #print(f"After relu: {x.shape}")
        #x = self.maxpool(x)
        # Flatten and Fully Connected Layers
        x = x.view(x.size(0), -1)
        #print(f"After flattening for FC: {x.shape}")
        x = self.fc1(x)
        #print(f"After fc1: {x.shape}")
        x = self.relu(x)
        #print(f"After relu: {x.shape}")
        x = self.dropout(x)
        #print(f"After dropout: {x.shape}")
        x = self.fc2(x)
        #print(f"After fc2: {x.shape}")
        x = self.softmax(x)
        #print(f"After softmax: {x.shape}")
        return x

class CNNModel_3layer(nn.Module):
    def __init__(self, config, input_dim, num_classes):
        super(CNNModel_3layer, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.config = config

        # Activation and Pooling
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(config["maxpool"])
        self.softmax = nn.Softmax(dim=1)
        
        # Convolutional Layers
        self.conv1 = nn.Conv1d(1, config["conv_layer_sizes"][0], 
                            kernel_size=config["kernel_size"], 
                            stride=config["stride"], 
                            padding=config["padding"])
        self.bn1 = nn.BatchNorm1d(config["conv_layer_sizes"][0]) if config["use_batchnorm"] else nn.Identity()
        
        self.conv2 = nn.Conv1d(config["conv_layer_sizes"][0], config["conv_layer_sizes"][1], 
                            kernel_size=config["kernel_size"], 
                            stride=config["stride"], 
                            padding=config["padding"])
        self.bn2 = nn.BatchNorm1d(config["conv_layer_sizes"][1]) if config["use_batchnorm"] else nn.Identity()
        
        self.conv3 = nn.Conv1d(config["conv_layer_sizes"][1], config["conv_layer_sizes"][2], 
                            kernel_size=config["kernel_size"], 
                            stride=config["stride"], 
                            padding=config["padding"])
        self.bn3 = nn.BatchNorm1d(config["conv_layer_sizes"][2]) if config["use_batchnorm"] else nn.Identity()
        
        # Dynamically calculate flattened size
        test_input = torch.randn(1, 1, input_dim)
        # Run through conv layers to calculate final size
        with torch.no_grad():
            test_x = self.conv1(test_input)
            test_x = self.maxpool(test_x)
            test_x = self.conv2(test_x)
            test_x = self.maxpool(test_x)
            test_x = self.conv3(test_x)
            if test_x.shape[-1]>1:
                test_x = self.maxpool(test_x)
            flattened_size = test_x.view(1, -1).size(1)
            #print(f"flattened_size of test input: {flattened_size}")
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(config["dropout_rate"])
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        #print(f"Input x shape: {x.shape}")
        # Ensure input is the right shape
        x = x.unsqueeze(1)  # Reshape input to (batch_size, 1, sequence_length)
        #print(f"After unsqueeze: {x.shape}")
        
        # Conv Block 1
        x = self.conv1(x)
        #print(f"After conv1: {x.shape}")
        x = self.bn1(x)
        #print(f"After bn1: {x.shape}")
        x = self.relu(x)
        #print(f"After relu: {x.shape}")
        x = self.maxpool(x)
        #print(f"After maxpool: {x.shape}")
        
        # Conv Block 2
        x = self.conv2(x)
        #print(f"After conv2: {x.shape}")
        x = self.bn2(x)
        #print(f"After bn2: {x.shape}")
        x = self.relu(x)
        #print(f"After relu: {x.shape}")
        x = self.maxpool(x)
        #print(f"After maxpool: {x.shape}")
        
        # Conv Block 3
        x = self.conv3(x)
        #print(f"After conv3: {x.shape}")
        x = self.bn3(x)
        #print(f"After bn3: {x.shape}")
        x = self.relu(x)
        #print(f"After relu: {x.shape}")
        if x.shape[-1]>1:
            x = self.maxpool(x)
        #print(f"After maxpool: {x.shape}")
        
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
        x = self.softmax(x)
        #print(f"After softmax: {x.shape}")
        return x