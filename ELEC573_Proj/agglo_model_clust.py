import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def agglo_merge_procedure(userdef_df, model):
    """
    Parameters:
    - model (str or sklearn model object): The model to train. If string, it must be one of:
      ['LogisticRegression', 'SVC', 'RF', 'GradientBoosting', 'KNN', 'XGBoost'].
    """
    
    # Mapping model names to objects
    model_map = {
        'LogisticRegression': LogisticRegression(random_state=42),
        'SVC': SVC(kernel='rbf', random_state=42),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        # 'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    # Get the model object if a string is provided
    if isinstance(model, str):
        if model not in model_map:
            raise ValueError(f"Unsupported model: {model}. Choose from {list(model_map.keys())}.")
        base_model = model_map[model]
    else:
        base_model = model
        
    # Data structures for logging cluster merging procedure
    merge_log = []  # List of tuples: [(cluster1, cluster2, distance, new_cluster), ...]
    unique_clusters_log = []  # List of lists: [list of unique clusters at each step]
    # Dictionary to store self-performance over iterations
    intra_cluster_performance = {}
    cross_cluster_performance = {}
    # Simulate cluster merging and model performance tracking
    iterations = 0

    # Main loop for cluster merging
    while len(userdef_df['Cluster_ID'].unique()) > 1:
        print(f"{len(userdef_df['Cluster_ID'].unique())} Clusters Remaining")
        # Log the current state of clusters
        unique_clusters_log.append(sorted(userdef_df['Cluster_ID'].unique()))

        # userdef_df continuously changes wrt cluster ID
        ## So... do train_df and test_df continuously change? Cluster ID changes... 
        ## but that... may or may not affect stratificiation in a meaningful way (I'm not using cluster metadata...)
        train_df, test_df = split_train_test(userdef_df, test_size=0.3, random_state=42)
        # These 2 should be the same... it's stratified...
        current_train_cluster_ids = sorted(train_df['Cluster_ID'].unique())
        current_test_cluster_ids = sorted(test_df['Cluster_ID'].unique())
        if current_train_cluster_ids == current_test_cluster_ids:
            current_cluster_ids = current_train_cluster_ids
        else:
            raise ValueError("Train/test Cluster ID lists not the same length... Stratify failed")

        # Train models with logging for specified clusters
        clus_model_lst = train_and_cv_cluster_model(train_df, model, current_cluster_ids)
        # Pairwise test models with logging for specified clusters
        sym_acc_arr = test_models_on_clusters(test_df, clus_model_lst, current_cluster_ids)

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
        userdef_df.loc[userdef_df['Cluster_ID'].isin([row_cluster_to_merge, col_cluster_to_merge]), 'Cluster_ID'] = new_cluster_id

        # Remove merged clusters from tracking (mark end with None)
        intra_cluster_performance[row_cluster_to_merge].append((iterations, None))
        intra_cluster_performance[col_cluster_to_merge].append((iterations, None))
        cross_cluster_performance[row_cluster_to_merge].append((iterations, None))
        cross_cluster_performance[col_cluster_to_merge].append((iterations, None))

        iterations += 1
    
    return merge_log, intra_cluster_performance, cross_cluster_performance


def split_train_test(userdef_df, test_size=0.3, random_state=42, stratify_column='Gesture_Encoded'):
    """
    Perform a train-test split on the entire dataset.

    Parameters:
    - userdef_df (DataFrame): The input dataframe with cluster, feature, and target data.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Random seed for reproducibility.
    - stratify_column (str): Column used for stratification.

    Returns:
    - train_df (DataFrame): Training subset of the data.
    - test_df (DataFrame): Test subset of the data.
    """
    stratify_labels = userdef_df[stratify_column]
    train_df, test_df = train_test_split(
        userdef_df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels
    )
    return train_df, test_df


def train_and_cv_cluster_model(train_df, model, cluster_ids, cluster_column='Cluster_ID', feature_column='feature', target_column='Gesture_Encoded', n_splits=3):
    """
    Perform k-fold cross-validation for models trained on each cluster in the dataset.
    
    Parameters:
    - userdef_df (DataFrame): The input dataframe with cluster, feature, and target data.
    - model (str or sklearn model object): The model to train. If string, it must be one of:
      ['LogisticRegression', 'SVC', 'RF', 'GradientBoosting', 'KNN'].
    - cluster_ids (list): List of cluster IDs to process.
    - cluster_column (str): Column name representing the cluster IDs.
    - feature_column (str): Column name containing feature arrays.
    - target_column (str): Column name for target labels.
    - n_splits (int): Number of cross-validation splits.
    
    Returns:
    - avg_val_accuracy (float): The average validation accuracy across all folds and clusters.
    """
    
    # Mapping model names to objects
    model_map = {
        'LogisticRegression': LogisticRegression(random_state=42),
        'SVC': SVC(kernel='rbf', random_state=42),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
    }
    
    # Get the model object if a string is provided
    if isinstance(model, str):
        if model not in model_map:
            raise ValueError(f"Unsupported model: {model}. Choose from {list(model_map.keys())}.")
        base_model = model_map[model]
    else:
        base_model = model

    total_val_accuracy = 0
    num_folds_processed = 0
    clus_model_lst = []

    for cluster in cluster_ids:
        # Filter data for the current cluster
        cluster_data = train_df[train_df[cluster_column] == cluster]
        X = np.array([x.flatten() for x in cluster_data[feature_column]])
        y = np.array(cluster_data[target_column])

        # Stratified K-Fold for validation splits
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cluster_val_accuracy = 0

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create a fresh model for each fold
            fold_model = base_model.__class__(**base_model.get_params())
            fold_model.fit(X_train, y_train)

            # Predict on the validation set and calculate accuracy
            y_pred = fold_model.predict(X_val)
            cluster_val_accuracy += accuracy_score(y_val, y_pred)
            
        # No great way to save the models... I really only want to use 1...
        ## So for now I'll just save the last kfold's model...
        ## So this is consistently biased in one direction but hopefully the val splits are all roughly equivalent
        ## TODO: Check this assumption
        clus_model_lst.append(fold_model)

        # Average accuracy for this cluster
        cluster_val_accuracy /= n_splits
        # I think this really ought to append not add...
        total_val_accuracy += cluster_val_accuracy
        num_folds_processed += 1

        #print(f"Cluster {cluster}: Average validation accuracy = {cluster_val_accuracy:.4f}")

    # Overall average accuracy across all clusters
    avg_val_accuracy = total_val_accuracy / num_folds_processed
    #print(f"\nOverall Average Validation Accuracy: {avg_val_accuracy:.4f}")
    
    #return avg_val_accuracy
    return clus_model_lst


def train_cluster_model(userdef_df, model, cluster_ids, cluster_column='Cluster_ID', feature_column='feature', target_column='Gesture_Encoded'):
    ### NOT USED ANYMORE! REPLACED BY KFCV ABOVE!! ###
    
    """
    Train a specific model for each cluster in the dataset.

    Parameters:
    - userdef_df (DataFrame): The input dataframe with cluster, feature, and target data.
    - model (str or sklearn model object): The model to train. If string, it must be one of:
      ['LogisticRegression', 'SVC', 'RF', 'GradientBoosting', 'KNN', 'XGBoost'].
    - cluster_column (str): Column name representing the cluster IDs.
    - feature_column (str): Column name containing feature arrays.
    - target_column (str): Column name for target labels.

    Returns:
    - model_list (list): A list of trained models, one per cluster.
    """
    
    # Mapping model names to objects
    model_map = {
        'LogisticRegression': LogisticRegression(random_state=42),
        'SVC': SVC(kernel='rbf', random_state=42),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        # 'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    # Get the model object if a string is provided
    if isinstance(model, str):
        if model not in model_map:
            raise ValueError(f"Unsupported model: {model}. Choose from {list(model_map.keys())}.")
        base_model = model_map[model]
    else:
        base_model = model

    model_list = []
    num_clusters = len(cluster_ids)
    
    for clus_idx, cluster in enumerate(cluster_ids):
        #print(f"Training models for clus_idx {clus_idx} of {num_clusters} (ID: {cluster})")
        
        # Get data for the current cluster
        clus_trainset = userdef_df[userdef_df[cluster_column] == cluster]

        # Split into training and testing data
        X_train, _, y_train, _ = train_test_split(
            np.array(clus_trainset[feature_column]), 
            np.array(clus_trainset[target_column]), 
            test_size=0.3, 
            random_state=42, 
            stratify=np.array(clus_trainset[target_column])
        )
        X_train = np.array([x.flatten() for x in X_train])

        # CHECKING IF TRAIN DATASET IS THE SAME EVERY TIME
        ## For the current code, it is supposed to be
        #if cluster in log_clusters:
        #    print(f"Progress: {clus_idx}/{num_clusters} (Training clus: {cluster})")
        #    print(f"[LOG] Trainset for clus {cluster}: Norm={np.linalg.norm(X_train):.4f}")
        #    #print(f"[LOG] Train labels for cluster {cluster}: {y_train[:3]}, Norm={np.linalg.norm(y_train):.4f}")
        
        # Train the model
        model = base_model.__class__(**base_model.get_params())
        model.fit(X_train, y_train)
        model_list.append(model)

    return model_list


def test_models_on_clusters(test_df, trained_clus_models_lst, cluster_ids, cluster_column='Cluster_ID', feature_column='feature', target_column='Gesture_Encoded', verbose=False):
    """
    Test trained models for each cluster on a pre-split test set.

    Parameters:
    - test_df (DataFrame): The test subset of the data.
    - trained_clus_models_lst (list): List of trained models, one per cluster.
    - cluster_ids (list): List of cluster IDs corresponding to the models.
    - cluster_column (str): Column name representing the cluster IDs.
    - feature_column (str): Column name containing feature arrays.
    - target_column (str): Column name for target labels.
    - verbose (bool): If True, logs additional details.

    Returns:
    - acc_matrix (ndarray): Accuracy matrix where (i, j) represents the accuracy
      of the i-th cluster's model on the j-th cluster's data.
    """
    num_clusters = len(cluster_ids)
    acc_matrix = np.zeros((num_clusters, num_clusters))

    for clus_idx, cluster in enumerate(cluster_ids):
        if verbose:
            print(f"Testing model for Cluster ID {cluster} ({clus_idx + 1}/{num_clusters})")

        model = trained_clus_models_lst[clus_idx]

        for clus_idx2, cluster2 in enumerate(cluster_ids):
            clus_testset = test_df[test_df[cluster_column] == cluster2]

            # Ensure there is test data for this cluster
            if clus_testset.empty:
                if verbose:
                    print(f"No test data for Cluster ID {cluster2}. Skipping.")
                acc_matrix[clus_idx, clus_idx2] = np.nan  # Or use 0 if nan is undesired
                continue

            X_test = np.array([x.flatten() for x in clus_testset[feature_column]])
            y_test = np.array(clus_testset[target_column])

            # Compute accuracy
            acc_matrix[clus_idx, clus_idx2] = accuracy_score(y_test, model.predict(X_test))

            if verbose:
                print(f"Model {clus_idx} on Cluster {cluster2}: Accuracy={acc_matrix[clus_idx, clus_idx2]:.4f}")

    return acc_matrix


def compute_performance_ratios(cluster_performance, cross_cluster_performance):
    """
    Compute and print the performance ratio (intra / cross) for each cluster.

    Parameters:
        cluster_performance (dict): Intra-cluster performance dictionary {cluster_id: [(iteration, accuracy), ...]}.
        cross_cluster_performance (dict): Cross-cluster performance dictionary.
    """
    print("\n=== Performance Ratios ===")
    
    intra_mean_lst = []
    cross_mean_lst = []
    ratio_lst = []
    
    for cluster_id in cluster_performance:
        intra_perf = [perf for it, perf in cluster_performance[cluster_id] if perf is not None]
        cross_perf = [perf for it, perf in cross_cluster_performance.get(cluster_id, []) if perf is not None]
        
        if not intra_perf or not cross_perf:
            continue
        
        intra_mean = np.mean(intra_perf)
        cross_mean = np.mean(cross_perf)
        ratio = intra_mean / cross_mean if cross_mean != 0 else float('inf')
        
        intra_mean_lst.append(intra_mean)
        cross_mean_lst.append(cross_mean)
        ratio_lst.append(ratio)
        
        print(f"Cluster {cluster_id}: Intra Mean = {intra_mean:.3f}, Cross Mean = {cross_mean:.3f}, Ratio = {ratio:.3f}")

    return intra_mean_lst, cross_mean_lst, ratio_lst


# Helper function to compute statistics for clusters
def compute_statistics(cluster_data, description):
    """
    Compute and print summary statistics for given cluster data.

    Parameters:
        cluster_data (dict): Dictionary where keys are cluster IDs and values are [(iteration, accuracy), ...].
        description (str): Description of the dataset (e.g., intra-cluster or cross-cluster).
    """
    print(f"\n=== Statistics for {description} ===")
    
    # Separate clusters that start at iteration 0 and others
    starts_at_zero = {k: v for k, v in cluster_data.items() if v[0][0] == 0}
    starts_later = {k: v for k, v in cluster_data.items() if v[0][0] != 0}
    
    # Summary function
    def summarize_clusters(clusters, label):
        print(f"\nClusters {label}:")
        print(f"Count: {len(clusters)}")
        print()
        
        for cluster_id, data in clusters.items():
            iterations = [it for it, perf in data if perf is not None]
            performances = [perf for it, perf in data if perf is not None]
            if not performances:
                continue  # Skip clusters with no valid data
            
            start_acc = performances[0]
            final_acc = performances[-1]
            duration = len(iterations)
            mean_acc = np.mean(performances)
            std_acc = np.std(performances)
            
            # Print statistics for the cluster
            print(f"Cluster {cluster_id}:")
            print(f"  Started at Iteration: {iterations[0]}")
            print(f"  Duration (Iterations): {duration}")
            print(f"  Start Accuracy: {start_acc:.3f}")
            print(f"  Final Accuracy: {final_acc:.3f}")
            print(f"  Mean Accuracy: {mean_acc:.3f}")
            print(f"  Std Dev Accuracy: {std_acc:.3f}")
            print(f"  Performance {'Improved' if final_acc > start_acc else 'Worsened'} over time.")
            print()
        
        # Compute overall trends
        if clusters:
            durations = [len([it for it, perf in data if perf is not None]) for data in clusters.values()]
            improvements = [
                1 if (perf[-1] > perf[0]) else -1 for data in clusters.values()
                if (perf := [perf for it, perf in data if perf is not None])
            ]
            print("\nOverall Trends:")
            print(f"  Avg Duration: {np.mean(durations):.2f}")
            print(f"  Std Duration: {np.std(durations):.2f}")
            print(f"  Improvement (1=Improved, -1=Worsened): Mean={np.mean(improvements):.2f}")

    # Summarize clusters starting at iteration 0
    summarize_clusters(starts_at_zero, "Starting at Iteration 0")
    # Summarize clusters starting later
    summarize_clusters(starts_later, "Starting Later")