SHARED_BS = 35
NUM_CHANNELS = 16
# Acceptable divisible batchsizes: (1, 560), (2, 280), (4, 140), (5, 112), (7, 80), (10, 56), (14, 40), (16, 35), (20, 28)

OLD_config = {
    "learning_rate": 0.0001,
    "batch_size": SHARED_BS,
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
    "batch_size": SHARED_BS,
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
    "batch_size": SHARED_BS,
    "learning_rate": 0.0001,
    "num_epochs": 50,
    "optimizer": "adam",
    "weight_decay": 1e-4, 
    "sequence_length": 2,
    "time_steps": 32
    }

HybridCNNLSTM_config = {
    "batch_size": SHARED_BS,
    "learning_rate": 0.0001,
    "num_epochs": 50,
    "optimizer": "adam",
    "weight_decay": 1e-4, 
    "sequence_length": 2,
    "time_steps": 32
    }

EMGHandNet_config = {
    "batch_size": SHARED_BS,
    "learning_rate": 0.0001,
    "num_epochs": 50,
    "optimizer": "adam",
    "weight_decay": 1e-4, 
    "sequence_length": 2,
    "time_steps": 32
    }

MomonaNet_config = {
    "num_channels": NUM_CHANNELS, 
    "batch_size": SHARED_BS,
    "learning_rate": 0.0001,
    "num_epochs": 50,
    "optimizer": "adam",
    "weight_decay": 1e-4,
    #"use_batchnorm": True,
    #"dropout_rate": 0.5
    "sequence_length": 64, 
    "time_steps": None  # This doesn't get used at all, unlike the others!
}