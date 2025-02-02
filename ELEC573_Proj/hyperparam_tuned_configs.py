from datetime import datetime

NUM_CHANNELS = 16
timestamp = datetime.now().strftime("%Y%m%d_%H%M")


DynamicMomonaNet_config = {
    "num_channels": NUM_CHANNELS,  #(int): Number of input channels.
    "sequence_length": 64,  #(int): Length of the input sequence.
    # ^ 32 wasn't working, it doesn't support time_steps to do multiple sequence batches for each slice
    "time_steps": None,  # Required so that reused code doesn't break. Not used in DynamicMomonaNet
    "conv_layers": [(32, 5, 1), (64, 3, 1), (128, 2, 1)], 
    "pooling_layers": [True, False, False, False],  # Max pooling only after the first conv layer
    "use_dense_cnn_lstm": True,  # Use dense layer between CNN and LSTM
    "lstm_hidden_size": 16,  #(int): Hidden size for LSTM layers.
    "lstm_num_layers": 1,  #(int): Number of LSTM layers.
    "fc_layers": [128, 64],  #(list of int): List of integers specifying the sizes of fully connected layers.
    "num_classes": 10, #(int): Number of output classes.
    # HYPERPARAMS
    "batch_size": 16,  #SHARED_BS #(int): Batch size.
    "lstm_dropout": 0.8,  #(float): Dropout probability for LSTM layers.
    "cnn_dropout": 0.0,
    "dense_cnnlstm_dropout": 0.0, 
    "fc_dropout": 0.0, 
    "learning_rate": 0.001,
    "num_epochs": 500,
    "optimizer": "adam",
    "weight_decay": 1e-4,
    "ft_learning_rate": 0.01,
    "num_ft_epochs": 500,
    "ft_weight_decay": 1e-4, 
    "ft_batch_size": 1, 
    "use_earlystopping": True,  # Always use this to save time, in ft and earlier training
    # METADATA
    "results_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results\\{timestamp}",  # \\hyperparam_tuning
    "models_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\models\\{timestamp}",  # \\hyperparam_tuning
    "perf_log_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results\\performance_logs", 
    "timestamp": timestamp,
    "verbose": False,
    "log_each_pid_results": False, 
    "save_ft_models": False  # Not even applicable here
}