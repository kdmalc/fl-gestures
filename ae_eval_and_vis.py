import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class GestureDataset(Dataset):
    # NOTE: I think this formulation makes it so the dataloader won't return (X, Y) as per usual (eg like TensorDataset)
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def eval_on_testset_and_return_original_and_reconstructed(model, test_loader, criterion=nn.MSELoss(), batch_modulus=500, get_latent_representations=True):
    model.eval()
    test_losses = []
    sample_data_lst = []
    reconstructions_lst = []
    latent_representations_lst = []
    with torch.no_grad():
        for batch_num, batch in enumerate(test_loader):
            if get_latent_representations:
                #latent = model.encoder(batch)  # This module isn't defined the way PyTorch expects, so don't use this
                latent = model.encode(batch)  # Created my own encode() func
                latent_representations_lst.append(latent.cpu().numpy())
            output = model(batch)
            if batch_num%batch_modulus==0:
                # Since we are batching along the individual gestures, order shouldn't really matter
                sample_data_lst.append(batch)
                reconstructions_lst.append(output)
            loss = criterion(output, batch)
            test_losses.append(loss.item())
    average_test_loss = sum(test_losses) / len(test_losses)
    print(f"Average testing loss across the entire test_loader: {average_test_loss}")
    return average_test_loss, sample_data_lst, reconstructions_lst, latent_representations_lst


def visualize_original_vs_reconstructed_gestures(sample_data, reconstructions, selected_gestures=[1, 10, 30], selected_channels=[1, 10, 30], figure_size=(20, 15)):
    fig, axes = plt.subplots(len(selected_gestures), len(selected_channels), figsize=figure_size)
    
    for i, gesture in enumerate(selected_gestures):
        for j, channel in enumerate(selected_channels):
            original = sample_data[gesture, :, channel]
            reconstructed = reconstructions[gesture, :, channel]
            
            axes[i, j].plot(original, label='Original', color='blue')
            axes[i, j].plot(reconstructed, label='Reconstructed', linestyle='dotted', color='purple')
            axes[i, j].fill_between(range(len(original)), original, reconstructed, color='red', alpha=0.3)
            
            axes[i, j].set_title(f'Gesture {gesture}, Channel {channel}')
            if i == 0 and j == 0:
                axes[i, j].legend()
    
    plt.tight_layout()
    plt.show()


def latent_space_vis(latent_representations_lst, plot_TSNE=True, calc_PCA=True, plot_PCA=True, num_dims=2, point_size=7):
    if num_dims>2:
        print("Warning: num_dims > 2: visualizations will only use the first two dimensions")
        
    # Concat list into a single numpy array
    latent_representations = np.concatenate(latent_representations_lst)
    # Reshape to 2D since TSNE and such can't handle 3D inputs...
    latent_representations_reshaped = latent_representations.reshape(latent_representations.shape[0], -1)

    if plot_TSNE:
        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=num_dims, random_state=42)
        reduced_latent = tsne.fit_transform(latent_representations_reshaped)
        plt.figure(figsize=(10, 7))
        plt.scatter(reduced_latent[:, 0], reduced_latent[:, 1], s=point_size)  # s=2 controls point size
        plt.title("T-SNE Visualization of Encoder Outputs")
        plt.show()
    if calc_PCA or plot_PCA:
        pca = PCA(n_components=num_dims)
        latent_pca = pca.fit_transform(latent_representations_reshaped)
        explained_variance = pca.explained_variance_ratio_
        print(f"Explained variance by each principal component: {explained_variance}")
        print(f"Total explained variance: {np.sum(explained_variance):.4f}")
    if plot_PCA:
        plt.figure(figsize=(10, 7))
        plt.scatter(latent_pca[:, 0], latent_pca[:, 1], s=point_size)#, alpha=0.7)  # Idk why alpha is here...
        plt.title(f'PCA of Latent Representations: ExplnVar={np.sum(explained_variance)*100:.2f}%')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()
