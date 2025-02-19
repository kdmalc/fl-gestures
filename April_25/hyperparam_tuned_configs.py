import copy
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
NUM_CHANNELS = 16  # Note that ELEC573Net is hardcoded in, doesn't use this...
NUM_TRAIN_GESTURES = 8  # For pretrained models
NUM_FT_GESTURES = 1  # Oneshot finetuning


# Base configuration, shared between all other configs (unless they get overwritten)
base_config = {
    "feature_engr": None, 
    "time_steps": 1,
    "sequence_length": 64,
    "num_train_gesture_trials": NUM_TRAIN_GESTURES, 
    "num_ft_gesture_trials": NUM_FT_GESTURES,
    "num_pretrain_users": 24, 
    "num_testft_users": 8, 
    "weight_decay": 0.0,
    "fc_dropout": 0.3,
    "cnn_dropout": 0.0,
    "batch_size": 16,
    "timestamp": timestamp,
    "optimizer": "adam",
    "num_epochs": 100,
    "num_classes": 10,
    "num_channels": NUM_CHANNELS,
    "user_split_json_filepath": "April_25\\fixed_user_splits\\24_8_user_splits.json",
    "results_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\April_25\\results\\{timestamp}",
    "models_save_dir": f"C:\\Users\\kdmen\\Repos\\fl-gestures\\April_25\\models\\{timestamp}",
    "learning_rate": 0.001,
    "finetune_strategy": "progressive_unfreeze",
    "progressive_unfreezing_schedule": 5,
    "ft_weight_decay": 0.0,
    "ft_learning_rate": 0.001,
    "ft_batch_size": 10,
    "num_ft_epochs": 100,
    "use_earlystopping": True,
    "lr_scheduler_patience": 4,
    "lr_scheduler_factor": 0.1, 
    "earlystopping_patience": 6,
    "earlystopping_min_delta": 0.01, 
    "ft_lr_scheduler_patience": 6, 
    "ft_lr_scheduler_factor": 0.1, 
    "ft_earlystopping_patience": 8,
    "ft_earlystopping_min_delta": 0.01,
    "verbose": False,
    "save_ft_models": False,
    "log_each_pid_results": False
}


DynamicMomonaNetR_config = copy.deepcopy(base_config)
DynamicMomonaNetR_config.update({
    "model_str": "DynamicMomonaNet",
    "dense_cnnlstm_dropout": 0.3,
    "conv_layers": [
        [64, 3, 1],
        [128, 3, 1]],
    "pooling_layers": [True, True, True, True],
    "lstm_num_layers": 1,
    "lstm_hidden_size": 64,
    "lstm_dropout": 0.0,  # When there is only 1 LSTM layer, dropout doesn't do anything!
    "fc_layers": [128, 64],
    "added_dense_ft_hidden_size": 64, 
    "use_dense_cnn_lstm": True,  
})
DynamicMomonaNetR_config["weight_decay"] = 1e-4
DynamicMomonaNetR_config["fc_dropout"] = 0.4
DynamicMomonaNetR_config["time_steps"] = None
DynamicMomonaNetR_config["progressive_unfreezing_schedule"] = 4
DynamicMomonaNetR_config["ft_weight_decay"] = 0.001
DynamicMomonaNetR_config["ft_learning_rate"] = 0.0001


DynamicMomonaNet_config = copy.deepcopy(base_config)
DynamicMomonaNet_config.update({
    "model_str": "DynamicMomonaNet",
    "dense_cnnlstm_dropout": 0.3,
    "conv_layers": [
        [64, 3, 1],
        [128, 3, 1]],
    "pooling_layers": [True, True, True, True],
    "lstm_num_layers": 1,
    "lstm_hidden_size": 8,
    "lstm_dropout": 0.0,
    "fc_layers": [128, 64],
    "added_dense_ft_hidden_size": 64, 
    "use_dense_cnn_lstm": False,  
})


ELEC573Net_config = copy.deepcopy(base_config)
ELEC573Net_config.update({
    "model_str": "ELEC573Net",
    "conv_layers": [
        [64, 3, 1],
        [128, 3, 1]],
    "fc_layers": [128],
    "use_batchnorm": True,  # I don't think this exists / is implemented rn?
    "padding": 1 , # I don't think this is implemented rn...
    "pooling_layers": None,  # In ELEC573Net pooling is on by default, see below. Should this be set to None or...
    "maxpool": 1,  # Just adding/passing this through --> Does this do anything...
})
ELEC573Net_config["batch_size"] = 32
ELEC573Net_config["ft_batch_size"] = 32  # How was this using 32 as its batch_size if there's only 30 total finetuning gestures...
ELEC573Net_config["num_channels"] = 80
ELEC573Net_config["cnn_dropout"] = 0.0
ELEC573Net_config["ft_learning_rate"] = 0.00001
ELEC573Net_config["ft_weight_decay"] = 0.05  # This is really high...
ELEC573Net_config["num_ft_epochs"] = 20
ELEC573Net_config["finetune_strategy"] = "full"


DynamicCNNLSTM_config = copy.deepcopy(base_config)
DynamicCNNLSTM_config.update({
    "model_str": "CNNLSTM",
    "dense_cnnlstm_dropout": 0.3,
    "conv_layers": [
        [64, 3, 1],
        [128, 3, 1]],
    "pooling_layers": [True, True, True, True],
    "lstm_num_layers": 1,
    "lstm_hidden_size": 8,
    "lstm_dropout": 0.0,
    "fc_layers": [128, 64],
    "added_dense_ft_hidden_size": 64, 
    "use_dense_cnn_lstm": False,  
})


DynamicCNN_config = base_config.copy()
DynamicCNN_config.update({
    "model_str": "CNN",
    "conv_layers": [
        [64, 3, 1],
        [128, 3, 1]],
    "pooling_layers": [True, True, True, True],
    "fc_layers": [128, 64],
})
DynamicCNN_config["cnn_dropout"] = 0.3
