import pandas as pd
import numpy as np
#from sklearn.cross_decomposition import CCA
#from sklearn.decomposition import PCA
import copy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import torch

np.random.seed(42) 

from moments_engr import *
from agglo_model_clust import *
from DNN_FT_funcs import *


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

    max_epochs = config["num_epochs"]
    bs = config["batch_size"]
    lr = config["learning_rate"]
    
    unique_gestures = np.unique(train_df[target_column])
    num_classes = len(unique_gestures)
    input_dim = len(train_df[feature_column].iloc[0])
    
    model = select_model(model_type, config)  #, device="cpu", input_dim=input_dim, num_classes=num_classes)
    initial_state = copy.deepcopy(model.state_dict())

    total_val_accuracy = 0
    num_folds_processed = 0
    clus_model_dict = {}
    for cluster in cluster_ids:
        print(f"train_and_cv_DNN_cluster_model cluster {cluster}")

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
            
            ######################################################################################
            ######################################################################################
            ######################################################################################
            # Now train your fold_model
            epoch = 0
            done = False
            earlystopping = EarlyStopping()
            # Open a text file for logging
            # Toggle timestamp? Is this by participant? Or just once per config?
            #timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            #log_file = open(f"{config['results_save_dir']}\\{timestamp}_{model_type}_pretrained_training_log.txt", "w")
            while not done and epoch < max_epochs:
                epoch += 1
                train_loss = train_model(model, train_loader, optimizer)
                #train_loss_log.append(train_loss)

                # Validation
                intra_test_res = evaluate_model(model, val_loader)
                intra_test_loss = intra_test_res['loss']
                #intra_test_loss_log.append(intra_test_loss)
                #cross_test_loss = evaluate_model(model, cross_test_loader)['loss']
                #cross_test_loss_log.append(cross_test_loss)

                # Early stopping check
                if earlystopping(model, intra_test_loss):
                    done = True
                # Log metrics to the console and the text file
                log_message = (
                    f"Epoch {epoch}/{max_epochs}, "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Intra Testing Loss: {intra_test_loss:.4f}, "
                    #f"Cross Testing Loss: {cross_test_loss:.4f}, "
                    f"{earlystopping.status}\n")
                if config["verbose"]:
                    print(log_message, end="")  # Print to console
                #log_file.write(log_message)  # Write to file
            # Close the log file
            #log_file.close()
            fold_accuracy = intra_test_res['accuracy']
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
    print("DNN_agglo_merge_procedure started!")

    config_batch_size = config["batch_size"]
    
    train_df = data_dfs_dict['train']
    test_df = data_dfs_dict['test']
    
    # Initialize model tracking structures
    clus_model_dict = {}  # Persists across iterations
    previous_clusters = set()
    nested_clus_model_dict = {}

    # Data structures for logging
    merge_log = []
    unique_clusters_log = []
    intra_cluster_performance = {}
    cross_cluster_performance = {}

    iterations = 0
    while len(train_df['Cluster_ID'].unique()) > 1:
        print(f"Iter {iterations}: {len(train_df['Cluster_ID'].unique())} Clusters Remaining")
        current_clusters = sorted(train_df['Cluster_ID'].unique())
        current_cluster_set = set(current_clusters)
        unique_clusters_log.append(current_clusters)

        # Identify new clusters that need training
        new_clusters = current_cluster_set - previous_clusters
        print(f"New clusters to train: {new_clusters}")

        # Train only new clusters
        if new_clusters:
            new_models = train_and_cv_DNN_cluster_model(
                train_df, model_type, list(new_clusters), config, n_splits=n_splits
            )
            clus_model_dict.update(new_models)

        # Remove models for merged clusters (no longer exist)
        ## I don't think I need this? I don't think it matters that much
        # This prevents memory bloat and ensures we only keep relevant models
        #clus_model_dict = {k: v for k in clus_model_dict if k in current_cluster_set}  # Also an error saying v isn't defined...
        
        # Log current state of models
        nested_clus_model_dict[f"Iter{iterations}"] = copy.deepcopy(clus_model_dict)

        # Test all current models on current clusters
        current_models = [clus_model_dict[clus] for clus in current_clusters]
        sym_acc_arr = test_models_on_clusters(
            test_df, current_models, current_clusters, 
            pytorch_bool=True, bs=config_batch_size
        )

        for idx, cluster_id in enumerate(current_clusters):
            cross_acc_sum = 0
            cross_acc_count = 0

            for idx2, cluster_id2 in enumerate(current_clusters):
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

        # Find and merge clusters
        masked_diag_array = sym_acc_arr.copy()
        np.fill_diagonal(masked_diag_array, 0.0)
        similarity_score = np.max(masked_diag_array)
        max_index = np.unravel_index(np.argmax(masked_diag_array), masked_diag_array.shape)
        
        row_idx_to_merge = max_index[0]
        col_idx_to_merge = max_index[1]
        # Get actual cluster IDs to merge
        row_cluster_to_merge = current_clusters[row_idx_to_merge]
        col_cluster_to_merge = current_clusters[col_idx_to_merge]
        # Create a new cluster ID for the merged cluster
        new_cluster_id = max(current_clusters) + 1
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

        # Update cluster tracking
        previous_clusters = current_cluster_set
        iterations += 1

    return merge_log, intra_cluster_performance, cross_cluster_performance, nested_clus_model_dict