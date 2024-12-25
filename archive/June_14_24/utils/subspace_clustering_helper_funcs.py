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
    

def apply_model(model_str, input_df, num_dims, hp, columns_to_exclude=['Participant', 'Gesture_ID', 'Gesture_Num']):
    
    # Drop the metadata columns (eg cols that are not the actual timeseries data)
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
    

def apply_dim_reduc(data_df, model_str='PCA', use_full_dataset=False, num_dims=40, hp=None, modality=['EMG and IMU'], participant_inclusion=['All'], apply='ALL'):

    print("Start")

    gestures = ['pan', 'duplicate', 'zoom-out', 'zoom-in', 'move', 'rotate', 'select-single', 'delete', 'close', 'open']
    data_types = modality
    participant_types = participant_inclusion

    if use_full_dataset:        
        sel_df = data_df
        df_t, dim_reduc_model = apply_model(model_str, sel_df, num_dims, hp)
    else:
        for f_type in data_types:
            if f_type == 'EMG and IMU':
                sel_df = data_df
            elif f_type[0] == 'IMU':
                # slice just the IMU columns (cols with IMU in name)
                sel_df = data_df.iloc[:, 3:-16]
            elif f_type[0] == 'EMG':
                # slice just the EMG columns (cols with EMG in name)
                sel_df = data_df.iloc[:, -16:]
            else:
                raise ValueError(f"f_type {f_type} not found in [EMG, IMU, EMG and IMU]")

            for p_type in participant_types:
                if p_type == "All":
                    pIDs = sel_df['Participant'].unique()
                elif p_type == "Impaired":
                    # Idk what this indexing by ['Participant'] the second time is doing, presumably is broken
                    pIDs = sel_df[sel_df['Participant'].isin(pIDs_impaired)]['Participant'].unique()
                elif p_type == "Unimpaired":
                    # Idk what this indexing by ['Participant'] the second time is doing, presumably is broken
                    pIDs = sel_df[sel_df['Participant'].isin(pIDs_unimpaired)]['Participant'].unique()
                else:
                    raise ValueError(f"Participant type {p_type} not supported, check supported versions.")

                if apply.upper() == 'ALL':
                    df_t, dim_reduc_model = apply_model(model_str, sel_df, num_dims, hp)
                elif apply.upper() == 'BY USER':
                    for pid in pIDs:
                        for file_type in file_types:
                                user_df = sel_df[(sel_df['Participant'] == pid)]
                                df_t, dim_reduc_model = apply_model(model_str, user_df, num_dims, hp)
                elif apply.upper() == 'BY GESTURE':
                    for file_type in file_types:
                        for gesture in gestures:
                            gesture_df = sel_df[(data_df['Gesture_ID'] == gesture)]
                            df_t, dim_reduc_model = apply_model(model_str, gesture_df, num_dims, hp)
    print("Success")
    return df_t, dim_reduc_model
