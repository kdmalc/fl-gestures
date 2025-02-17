import torch
from torch.utils.data import Dataset
import numpy as np


def select_dataset_class(config):

    # These depend on the feature engineering...
    if config["model_str"] in ["CNN", "CNNLSTM"]:
        #if config["feature_engr"] is None:
        #    return GestureDataset_3D  # --> This doesnt work unless I add windowing or something... width=1 incompatible with 3x3 kernel
        #else:
        return GestureDataset  # This is 2D

    # THESE ARE FIXED
    if config["model_str"] in ["MomonaNet", "DynamicMomonaNet", "ELEC573Net"]:
        return GestureDataset  # This is 2D
    elif config["model_str"] in ["RNN", "HybridCNNLSTM", "CRNN"]:
        return GestureDataset_3D
    elif config["model_str"] in ["EMGHandNet"]:
        return GestureDataset_4D
    else:
        raise ValueError


class GestureDataset(Dataset):
    # 2D
    def __init__(self, features, labels, sl=1, ts=64, stride=None):
        # NOTE: SL, TS, Stride all NOT used, just a polymorphism relic...

        #self.features = torch.tensor(features, dtype=torch.float32)
        #self.labels = torch.tensor(labels, dtype=torch.long)
        self.features = torch.tensor(np.array(features), dtype=torch.float32)
        self.labels = torch.tensor(np.array(labels), dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    

class GestureDataset_3D(Dataset):
    def __init__(self, features, labels, sl=1, ts=64, stride=None, use_sliding_windows=False):
        """
        Dataset for gesture data with optional sliding window.
        Args:
        - features (numpy.ndarray): Shape (num_samples, total_time_steps, num_channels).
        - labels (numpy.ndarray): Corresponding labels for features.
        - sl (int) * ts (int) should be the desired window size
        - stride (int): Stride for sliding window. Defaults to ts (no overlap).
        """
        #self.features = torch.tensor(features, dtype=torch.float32)
        #self.labels = torch.tensor(labels, dtype=torch.long)
        self.features = torch.tensor(np.array(features), dtype=torch.float32)
        self.labels = torch.tensor(np.array(labels), dtype=torch.long)
        self.window_size = sl*ts
        self.stride = stride if stride else ts
        self.num_channels = 16  # Hardcoded based on known data structure.
        self.use_sliding_windows = use_sliding_windows

        # Apply sliding window transformation
        self.windows, self.window_labels = self._create_sliding_windows(self.use_sliding_windows)

    def _create_sliding_windows(self, use_sliding_windows):
        """
        Create sliding windows over the feature data.
        Returns:
        - all_windows: Numpy array of windows, shape (num_windows, window_size, num_channels).
        - all_labels: Numpy array of corresponding labels for the windows.
        """

        if use_sliding_windows == False:
            # If sliding windows are disabled, return reshaped features directly
            all_features = np.stack([f.reshape(-1, self.num_channels) for f in self.features], axis=0).astype(np.float32)
            return all_features, np.array(self.labels, dtype=np.int64)

        all_windows = []
        all_labels = []

        for i in range(len(self.features)):
            # Reshape each flattened feature vector into (total_time_steps, num_channels)
            feature = self.features[i].reshape(-1, self.num_channels)  # Shape: (total_time_steps, num_channels)
            label = self.labels[i]

            # Generate sliding windows for the current feature
            num_windows = (feature.shape[0] - self.window_size) // self.stride + 1
            for w in range(num_windows):
                start_idx = w * self.stride
                end_idx = start_idx + self.window_size
                if end_idx <= feature.shape[0]:
                    # Extract window and reshape into (window_size, num_channels)
                    window = feature[start_idx:end_idx].reshape(-1, self.num_channels)
                    all_windows.append(window)
                    all_labels.append(label)  # Replicate the label for the sequence

        # Convert lists to numpy arrays safely
        all_windows = np.stack(all_windows, axis=0)  # Shape: (num_windows, window_size, num_channels)
        all_labels = np.array(all_labels, dtype=np.int64)  # Shape: (num_windows,)
        return all_windows.astype(np.float32), all_labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.window_labels[idx]
    

class GestureDataset_4D(Dataset):
    def __init__(self, features, labels, sl=1, ts=64, stride=None):
        """
        Dataset for gesture data with optional sliding window.
        Args:
        - features (numpy.ndarray): Shape (num_samples, total_time_steps, num_channels).
        - labels (numpy.ndarray): Corresponding labels for features.
        - sl (int): Sequence length (number of windows).
        - ts (int): Time steps per window.
        - stride (int): Stride for sliding window. Defaults to ts (no overlap).
        """
        #self.features = torch.tensor(features, dtype=torch.float32)
        #self.labels = torch.tensor(labels, dtype=torch.long)
        self.features = torch.tensor(np.array(features), dtype=torch.float32)
        self.labels = torch.tensor(np.array(labels), dtype=torch.long)
        self.sl = sl
        self.ts = ts
        self.stride = stride if stride else ts
        # TODO: A way not to hardcode that is robust against adding IMU channels?
        if features.shape[1]!=1024:
            raise ValueError("Hardcoding no longer valid")
        self.num_channels = 16  # Hardcoded based on flattened shape (64 * 16). 16 emg channels.

        # Apply sliding window transformation
        self.windows, self.window_labels = self._create_sliding_windows()

    def _create_sliding_windows(self):
        all_windows = []
        all_labels = []

        for i in range(len(self.features)):
            # Reshape each flattened feature vector into (64, 16)
            feature = self.features[i].reshape(-1, self.num_channels)  # Shape: (64, 16)
            label = self.labels[i]

            # Generate sliding windows for the current feature
            num_windows = (feature.shape[0] - self.sl * self.ts) // self.stride + 1
            for w in range(num_windows):
                start_idx = w * self.stride
                end_idx = start_idx + self.sl * self.ts
                if end_idx <= feature.shape[0]:
                    # Extract window and reshape into (SL, TS, NC)
                    window = feature[start_idx:end_idx].reshape(self.sl, self.ts, self.num_channels)
                    all_windows.append(window)
                    all_labels.append(label)  # Replicate the label for the sequence

        return np.array(all_windows, dtype=np.float32), np.array(all_labels, dtype=np.int64)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32), self.window_labels[idx]
