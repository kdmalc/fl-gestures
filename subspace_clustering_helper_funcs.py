import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import MDS
from sklearn.manifold import Isomap

#from umap import UMAP


def apply_model(model_str, input_df, num_dims, hp):
    
    # Drop the metadata columns (eg cols that are not the actual timeseries data)
    training_df = input_df.drop(columns=['Participant', 'Gesture_ID', 'Gesture_Num'])
    
    if not training_df.empty:
        if model_str.upper() == 'PCA':
            dim_reduc_model = PCA(n_components=num_dims)
            dim_reduc_model.fit(training_df)
            reduced_df = pd.DataFrame(dim_reduc_model.transform(training_df))
        elif (model_str.upper() == 'T-SNE') or (model_str.upper() == 'TSNE'):
            dim_reduc_model = TSNE(n_components=num_dims, perplexity=hp, random_state=42)
            reduced_df = pd.DataFrame(dim_reduc_model.fit_transform(df))
        elif (model_str.upper() == 'INCREMENTALPCA') or (model_str.upper() == 'IPCA'):
            dim_reduc_model = IncrementalPCA(n_components=num_dims)
            reduced_df = pd.DataFrame(dim_reduc_model.fit_transform(training_df))
        elif (model_str.upper() == 'KERNELPCA') or (model_str.upper() == 'KPCA'):
            dim_reduc_model = KernelPCA(n_components=num_dims)
            reduced_df = pd.DataFrame(dim_reduc_model.fit_transform(training_df))
        #elif model_str.upper() == 'UMAP':
        #    raise ValueError("Need to install the umap library first...")
        #    dim_reduc_model = UMAP(n_components=num_dims)
        #    reduced_df = pd.DataFrame(dim_reduc_model.fit_transform(training_df))
        elif model_str.upper() == 'MDS':
            dim_reduc_model = MDS(n_components=num_dims, random_state=42)
            reduced_df = pd.DataFrame(dim_reduc_model.fit_transform(training_df))
        elif model_str.upper() == 'ISOMAP':
            dim_reduc_model = Isomap(n_components=num_dims)
            reduced_df = pd.DataFrame(dim_reduc_model.fit_transform(training_df))
        else:
            raise ValueError(f"{model_str} not implemented. Choose an implemented model.")
    else:
        raise ValueError(f"training_df is empty!")
    
    return reduced_df, dim_reduc_model


def interpolate_df(df, num_rows=64, columns_to_exclude=None):
    """
    Interpolates the dataframe to have a specified number of rows.
    Excludes specified columns from interpolation.
    """
    if columns_to_exclude is None:
        columns_to_exclude = []
    
    # Separate the columns to exclude
    excluded_columns_df = df[columns_to_exclude]
    subset_df = df.drop(columns=columns_to_exclude)
    
    # Interpolate the remaining columns
    x_old = np.linspace(0, 1, num=len(subset_df))
    x_new = np.linspace(0, 1, num=num_rows)
    interpolated_df = pd.DataFrame()

    for column in subset_df.columns:
        y_old = subset_df[column].values
        y_new = np.interp(x_new, x_old, y_old)
        interpolated_df[column] = y_new

    # Add the excluded columns back
    for col in columns_to_exclude:
        interpolated_df[col] = excluded_columns_df[col].iloc[0]

    # Reordering columns to match original DataFrame
    interpolated_df = interpolated_df[columns_to_exclude + list(subset_df.columns)]
    
    return interpolated_df

def interpolate_dataframe_Ben(df, num_rows=64):
    '''Old version from Ben's code (/Momona?), assumes no meta data'''
    # Create a new index array with num_rows evenly spaced values
    new_index = np.linspace(0, len(df) - 1, num_rows)
    
    # Create a new DataFrame with the new index and the original columns
    interpolated_df = pd.DataFrame(columns=df.columns, index=new_index)
    
    # Interpolate each column of the DataFrame using the new index
    for column in df.columns:
        interpolated_df[column] = np.interp(new_index, np.arange(len(df)), df[column])

    interpolated_df.index = range(num_rows)
    
    return interpolated_df
    

def apply_model(model_str, input_df, num_dims, hp, columns_to_exclude=['Participant', 'Gesture_ID', 'Gesture_Num']):
    
    # Drop the metadata columns (eg cols that are not the actual timeseries data)
    #input_df.drop(columns=['Participant', 'Gesture_ID', 'Gesture_Num', 'Gesture_Type', 'File_Type'], inplace=True)
    training_df = input_df.drop(columns=columns_to_exclude)
    
    if not training_df.empty:
        if model_str.upper() == 'PCA':
            dim_reduc_model = PCA(n_components=num_dims)
            dim_reduc_model.fit(training_df)
            reduced_df = pd.DataFrame(dim_reduc_model.transform(training_df))
        elif model_str.upper() == 'T-SNE':
            dim_reduc_model = TSNE(n_components=num_dims, perplexity=hp, random_state=42)
            reduced_df = pd.DataFrame(dim_reduc_model.fit_transform(df))
        elif (model_str.upper() == 'INCREMENTALPCA') or (model_str.upper() == 'IPCA'):
            dim_reduc_model = IncrementalPCA(n_components=num_dims)
            reduced_df = pd.DataFrame(dim_reduc_model.fit_transform(training_df))
        elif (model_str.upper() == 'KERNELPCA') or (model_str.upper() == 'KPCA'):
            dim_reduc_model = KernelPCA(n_components=num_dims)
            reduced_df = pd.DataFrame(dim_reduc_model.fit_transform(training_df))
        #elif model_str.upper() == 'UMAP':
        #    raise ValueError("Need to install the umap library first...")
        #    dim_reduc_model = UMAP(n_components=num_dims)
        #    reduced_df = pd.DataFrame(dim_reduc_model.fit_transform(training_df))
        elif model_str.upper() == 'MDS':
            dim_reduc_model = MDS(n_components=num_dims, random_state=42)
            reduced_df = pd.DataFrame(dim_reduc_model.fit_transform(training_df))
        elif model_str.upper() == 'ISOMAP':
            dim_reduc_model = Isomap(n_components=num_dims)
            reduced_df = pd.DataFrame(dim_reduc_model.fit_transform(training_df))
        else:
            raise ValueError(f"{model_str} not implemented. Choose an implemented model.")
    else:
        raise ValueError(f"training_df is empty!")
    
    return reduced_df, dim_reduc_model
    
