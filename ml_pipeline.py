import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


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

