import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from umap import UMAP
from sklearn.manifold import MDS
from sklearn.manifold import Isomap


def interpolate_dataframe(df, num_rows=64):
    # Create a new index array with num_rows evenly spaced values
    new_index = np.linspace(0, len(df) - 1, num_rows)
    
    # Create a new DataFrame with the new index and the original columns
    interpolated_df = pd.DataFrame(columns=df.columns, index=new_index)
    
    # Interpolate each column of the DataFrame using the new index
    for column in df.columns:
        interpolated_df[column] = np.interp(new_index, np.arange(len(df)), df[column])

    interpolated_df.index = range(num_rows)
    
    return interpolated_df
    

def apply_model(model_str, training_df, num_dims, hp):
    
    # Drop the metadata columns (eg cols that are not the actual timeseries data)
    training_df.drop(columns=['Participant', 'Gesture_ID', 'Gesture_Num', 'Gesture_Type', 'File_Type'], inplace=True)
    
    if not training_df.empty:
        if model_str.upper() == 'PCA':
            pca = PCA(n_components=num_dims)
            pca.fit(training_df)
            reduced_df = pd.DataFrame(pca.transform(training_df))
        elif model_str.upper() == 'T-SNE':
            tsne = TSNE(n_components=num_dims, perplexity=hp, random_state=42)
            reduced_df = pd.DataFrame(tsne.fit_transform(df))
        elif (model_str.upper() == 'INCREMENTALPCA') or (model_str.upper() == 'IPCA'):
            ipca = IncrementalPCA(n_components=num_dims)
            reduced_df = pd.DataFrame(ipca.fit_transform(training_df))
        elif (model_str.upper() == 'KERNELPCA') or (model_str.upper() == 'KPCA'):
            kpca = KernelPCA(n_components=num_dims)
            reduced_df = pd.DataFrame(kpca.fit_transform(training_df))
        elif model_str.upper() == 'UMAP':
            umap = UMAP(n_components=num_dims)
            reduced_df = pd.DataFrame(umap.fit_transform(training_df))
        elif model_str.upper() == 'MDS':
            mds = MDS(n_components=num_dims, random_state=42)
            reduced_df = pd.DataFrame(mds.fit_transform(training_df))
        elif model_str.upper() == 'ISOMAP':
            isomap = Isomap(n_components=num_dims)
            reduced_df = pd.DataFrame(isomap.fit_transform(training_df))
        else:
            raise ValueError(f"{model_str} not implemented. Choose an implemented model.")
    else:
        raise ValueError(f"training_df is empty!")
    
    return reduced_df
    
