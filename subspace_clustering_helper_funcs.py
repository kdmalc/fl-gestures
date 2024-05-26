import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
# Have not been tested/implemented yet:
from sklearn.decomposition import IncrementalPCA, KernelPCA
from sklearn.manifold import MDS, Isomap

#from umap import UMAP


def dunn_index(X, labels):
    # Compute pairwise distances between points in the same cluster
    intra_cluster_distances = []
    for cluster_label in np.unique(labels):
        cluster_points = X[labels == cluster_label]
        if len(cluster_points) > 1:
            nearest_neighbors = NearestNeighbors(n_neighbors=2).fit(cluster_points)
            distances, _ = nearest_neighbors.kneighbors()
            intra_cluster_distances.extend(distances[:, 1])

    # Compute pairwise distances between points in different clusters
    inter_cluster_distances = []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if labels[i] != labels[j]:
                inter_cluster_distances.append(np.linalg.norm(X[i] - X[j]))

    # Compute Dunn Index
    min_inter_cluster_distance = min(inter_cluster_distances)
    max_intra_cluster_distance = max(intra_cluster_distances)
    return min_inter_cluster_distance / max_intra_cluster_distance

def gap_statistics(X, labels):
    # Compute within-cluster dispersion
    Wk = np.sum([np.mean(np.linalg.norm(X[np.where(labels == k)] - np.mean(X[np.where(labels == k)], axis=0), axis=1)) for k in np.unique(labels)])

    # Generate reference datasets and compute within-cluster dispersion for each
    reference_Wk = []
    for i in range(10):
        random_data = np.random.rand(*X.shape)
        reference_labels = KMeans(n_clusters=len(np.unique(labels))).fit_predict(random_data)
        reference_Wk.append(np.sum([np.mean(np.linalg.norm(random_data[np.where(reference_labels == k)] - np.mean(random_data[np.where(reference_labels == k)], axis=0), axis=1)) for k in np.unique(reference_labels)]))
    
    # Compute Gap Statistics
    gap = np.mean(np.log(reference_Wk)) - np.log(Wk)
    return gap

def run_clustering_algorithm(clustering_algo_str, X, num_clusters=2, eps=0.5, min_samples=5):
    if clustering_algo_str == 'KMeans':
        model = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto') 
        # ^ Read about auto, basically controls how many times it is rerun
    elif clustering_algo_str == 'DBSCAN':
        #It's suitable for datasets with complex cluster shapes and noise.
        #However, it may struggle with clusters of varying densities or high-dimensional data.
        #--> Probably both are true for us?...
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif clustering_algo_str == 'OPTICS':
        #It's suitable for datasets with varying cluster densities and non-convex shapes.
        #It can handle noise and doesn't require specifying the number of clusters in advance.
        model = OPTICS(eps=eps, min_samples=min_samples)

    labels = model.fit_predict(X)

    # Compute clustering metrics
    silhouette = silhouette_score(X, labels)
    db_index = davies_bouldin_score(X, labels)
    ch_index = calinski_harabasz_score(X, labels)
    di_index = dunn_index(X, labels)
    gap_stat = gap_statistics(X, labels)

    return silhouette, db_index, ch_index, di_index, gap_stat



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
    
