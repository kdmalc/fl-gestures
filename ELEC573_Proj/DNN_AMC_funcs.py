import pandas as pd
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import copy

np.random.seed(42) 

from moments_engr import *
from agglo_model_clust import *
from DNN_FT_funcs import *

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import torch
from sklearn.metrics import accuracy_score


def train_and_cv_DNN_cluster_model(train_df, model_type, cluster_ids, config, 
                                   n_splits=3, cluster_column='Cluster_ID', feature_column='feature', 
                                   target_column='Gesture_Encoded', criterion=nn.CrossEntropyLoss()):
    """
    Perform k-fold cross-validation for models trained on each cluster in the dataset.
    
    Parameters:
    - userdef_df (DataFrame): The input dataframe with cluster, feature, and target data.
    - model 
    - cluster_ids (list): List of cluster IDs to process.
    - cluster_column (str): Column name representing the cluster IDs.
    - feature_column (str): Column name containing feature arrays.
    - target_column (str): Column name for target labels.
    - n_splits (int): Number of cross-validation splits.
    
    Returns:
    - avg_val_accuracy (float): The average validation accuracy across all folds and clusters.
    """

    num_epochs = config["num_epochs"]
    bs = config["batch_size"]
    lr = config["learning_rate"]
    
    unique_gestures = np.unique(train_df[target_column])
    num_classes = len(unique_gestures)
    input_dim = len(train_df[feature_column].iloc[0])
    
    model = select_model(model_type, config, device="cpu", input_dim=input_dim, num_classes=num_classes)
    initial_state = copy.deepcopy(model.state_dict())

    total_val_accuracy = 0
    num_folds_processed = 0
    clus_model_dict = {}
    for cluster in cluster_ids:
        # Filter data for the current cluster
        cluster_data = train_df[train_df[cluster_column] == cluster]
        X = np.array([x for x in cluster_data[feature_column]])
        y = np.array(cluster_data[target_column])

        # Stratified K-Fold for validation splits
        ## IDK IF THIS WILL WORK WITH PYTORCH FORMATTED DATA...
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cluster_val_accuracy = 0
        for idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)
            # Create TensorDataset
            # TODO: Maybe I need to use my custom dataset classes? Maybe not for MomonaNet tho...
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            # Create DataLoader
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
            
            # Create a fresh instance 
            fold_model = model.__class__(config)  # , input_dim, num_classes
            # Load the saved initial weights from the variable
            fold_model.load_state_dict(initial_state)
            optimizer = set_optimizer(fold_model, lr=lr, use_weight_decay=config["weight_decay"]>0, weight_decay=config["weight_decay"])
            
            # Now train your fold_model
            fold_model.train()  # Ensure the model is in training mode
            for epoch in range(num_epochs):
                # Training loop with your X_train and y_train
                for X_batch, y_batch in train_loader:  # Assuming you have a DataLoader
                    optimizer.zero_grad()
                    outputs = fold_model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

            # Evaluate the model on the validation set
            fold_model.eval()
            fold_predictions = []
            fold_true_labels = []
            with torch.no_grad():
                # Validation loop
                #predictions = []
                for X_val, y_val in val_loader:
                    outputs = fold_model(X_val)
                    _, preds = torch.max(outputs, 1)  # ... the output is... logits? Why do we have to take the max? Shouldnt CrossEntropy softmax it anyways?...
                    fold_predictions.extend(preds.cpu().numpy())
                    fold_true_labels.extend(y_val.cpu().numpy())
            # Predict on the validation set and calculate accuracy
            #y_pred = fold_model.predict(X_val)
            #cluster_val_accuracy += accuracy_score(y_val, y_pred)
            # Calculate accuracy for the current fold
            fold_accuracy = accuracy_score(fold_true_labels, fold_predictions)
            cluster_val_accuracy += fold_accuracy

            if idx==0:
                # No great way to save the models... I really only want to use 1...
                ## So for now I'll just save the first kfold's model...
                ## Consistently biased but hopefully the val splits are all roughly equivalent
                clus_model_dict[cluster] = copy.deepcopy(fold_model)

        # REWRITE THIS!!! --> Wait why, what's the issue...
        # Average accuracy for this cluster
        #cluster_val_accuracy /= n_splits
        # I think this really ought to append not add...
        ## TOTAL maintains the acc of the entire process (across all clusters)
        #total_val_accuracy += cluster_val_accuracy
        #num_folds_processed += 1
    # Overall average accuracy across all clusters
    ## Why is this calcualted if it isn't returned?
    #avg_val_accuracy = total_val_accuracy / num_folds_processed
    #print(f"\nOverall Average Validation Accuracy: {avg_val_accuracy:.4f}")
    
    return clus_model_dict

def DNN_agglo_merge_procedure(data_dfs_dict, model_type, config, n_splits=2):
    """
    Parameters:
    - data_dfs_dict
    - model
    - config
    """

    config_batch_size = config["batch_size"]
    
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Unique gestures and number of classes
    unique_gestures = np.unique(data_dfs_dict['train']['Gesture_Encoded'])
    num_classes = len(unique_gestures)
    input_dim = len(data_dfs_dict['train']['feature'].iloc[0])
    
    # Get the model object if a string is provided
    model = select_model(model_type, config, device="cpu", input_dim=input_dim, num_classes=num_classes)
        
    train_df = data_dfs_dict['train']
    test_df = data_dfs_dict['test']
        
    # Data structures for logging cluster merging procedure
    merge_log = []  # List of tuples: [(cluster1, cluster2, distance, new_cluster), ...]
    unique_clusters_log = []  # List of lists: [list of unique clusters at each step]
    # Dictionary to store self-performance over iterations
    intra_cluster_performance = {}
    cross_cluster_performance = {}
    nested_clus_model_dict = {}
    # Simulate cluster merging and model performance tracking
    iterations = 0
    # Main loop for cluster merging
    while len(train_df['Cluster_ID'].unique()) > 1:
        print(f"Iter {iterations}: {len(train_df['Cluster_ID'].unique())} Clusters Remaining")
        # Log the current state of clusters
        unique_clusters_log.append(sorted(train_df['Cluster_ID'].unique()))

        # userdef_df continuously changes wrt cluster ID
        ## So... do train_df and test_df continuously change? Cluster ID changes... 
        ## but that... may or may not affect stratificiation in a meaningful way (I'm not using cluster metadata...)
        # These 2 should be the same... it's stratified...
        current_train_cluster_ids = sorted(train_df['Cluster_ID'].unique())
        current_test_cluster_ids = sorted(test_df['Cluster_ID'].unique())
        if current_train_cluster_ids == current_test_cluster_ids:
            current_cluster_ids = current_train_cluster_ids
        else:
            raise ValueError("Train and test datasets have mismatched Cluster_IDs (diff length?). Ensure proper stratification during data splitting.")

        # Train models with logging for specified clusters
        clus_model_dict = train_and_cv_DNN_cluster_model(train_df, model, current_cluster_ids, config, n_splits=n_splits)
        clus_model_lst = list(clus_model_dict.values())
        nested_clus_model_dict[f"Iter{iterations}"] = copy.deepcopy(clus_model_dict)
        # Pairwise test models with logging for specified clusters
        sym_acc_arr = test_models_on_clusters(test_df, clus_model_lst, current_cluster_ids, pytorch_bool=True, bs=config_batch_size)

        for idx, cluster_id in enumerate(current_cluster_ids):
            cross_acc_sum = 0
            cross_acc_count = 0

            for idx2, cluster_id2 in enumerate(current_cluster_ids):
                if cluster_id not in intra_cluster_performance:
                    intra_cluster_performance[cluster_id] = []  # Initialize list

                if idx == idx2:  # Diagonal, so intra-cluster
                    # Ensure the logic assumption holds
                    if cluster_id != cluster_id2:
                        raise ValueError("This code isn't working as expected...")
                    intra_cluster_performance[cluster_id].append((iterations, sym_acc_arr[idx, idx2]))
                else:  # Non-diagonal, so cross-cluster
                    cross_acc_sum += sym_acc_arr[idx, idx2]
                    cross_acc_count += 1

            # Calculate average cross-cluster accuracy
            if cross_acc_count > 0:
                avg_cross_acc = cross_acc_sum / cross_acc_count
            else:
                avg_cross_acc = None  # Handle the case where no cross-cluster pairs exist
            # Append the average cross-cluster accuracy to all relevant clusters
            if cluster_id not in cross_cluster_performance:
                cross_cluster_performance[cluster_id] = []  # Initialize list
            cross_cluster_performance[cluster_id].append((iterations, avg_cross_acc))

        masked_diag_array = sym_acc_arr.copy()
        np.fill_diagonal(masked_diag_array, 0.0)
        similarity_score = np.max(masked_diag_array)
        max_index = np.unravel_index(np.argmax(masked_diag_array), masked_diag_array.shape)
        row_idx_to_merge = max_index[0]
        col_idx_to_merge = max_index[1]
        # Get actual cluster IDs to merge
        row_cluster_to_merge = current_cluster_ids[row_idx_to_merge]
        col_cluster_to_merge = current_cluster_ids[col_idx_to_merge]
        # Create a new cluster ID for the merged cluster
        new_cluster_id = max(current_cluster_ids) + 1
        #print(f"MERGE: {row_cluster_to_merge, col_cluster_to_merge} @ {similarity_score*100:.2f}. New cluster: {new_cluster_id}")
        # Log the merge
        merge_log.append((iterations, row_cluster_to_merge, col_cluster_to_merge, similarity_score, new_cluster_id))
        # Update the DataFrame with the new merged cluster
        #userdef_df.loc[userdef_df['Cluster_ID'].isin([row_cluster_to_merge, col_cluster_to_merge]), 'Cluster_ID'] = new_cluster_id
        train_df.loc[train_df['Cluster_ID'].isin([row_cluster_to_merge, col_cluster_to_merge]), 'Cluster_ID'] = new_cluster_id
        test_df.loc[test_df['Cluster_ID'].isin([row_cluster_to_merge, col_cluster_to_merge]), 'Cluster_ID'] = new_cluster_id
        
        # Remove merged clusters from tracking (mark end with None)
        intra_cluster_performance[row_cluster_to_merge].append((iterations, None))
        intra_cluster_performance[col_cluster_to_merge].append((iterations, None))
        cross_cluster_performance[row_cluster_to_merge].append((iterations, None))
        cross_cluster_performance[col_cluster_to_merge].append((iterations, None))

        iterations += 1
    
    return merge_log, intra_cluster_performance, cross_cluster_performance, nested_clus_model_dict