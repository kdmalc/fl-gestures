import matplotlib.pyplot as plt
import seaborn as sns


def viz_merged_intracluster_acc(intra_cluster_performance, model_str):
    # Visualization
    plt.figure(figsize=(12, 6))

    for cluster_id in intra_cluster_performance:
        # Extract valid iterations and performance
        data = intra_cluster_performance[cluster_id]
        valid_iterations = [it for it, perf in data if perf is not None]
        valid_performance = [perf for it, perf in data if perf is not None]
        if valid_iterations[0]==0:
            continue
        #print(valid_iterations)
        #print(valid_performance)
        #print()
        plt.plot(valid_iterations, valid_performance, label=f"Cluster {cluster_id}")

    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Intra-Cluster Test Accuracy", fontsize=18)
    plt.title(f"{model_str} Intra-Cluster Acc: Merged Clusters Only", fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    

def viz_orig_intracluster_acc(intra_cluster_performance, model_str):
    # Visualization
    plt.figure(figsize=(12, 6))

    for cluster_id in intra_cluster_performance:
        # Extract valid iterations and performance
        data = intra_cluster_performance[cluster_id]
        valid_iterations = [it for it, perf in data if perf is not None]
        valid_performance = [perf for it, perf in data if perf is not None]
        if valid_iterations[0]!=0:
            continue
        #print(valid_iterations)
        #print(valid_performance)
        #print()
        plt.plot(valid_iterations, valid_performance, label=f"Cluster {cluster_id}")

    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Intra-Cluster Test Accuracy", fontsize=18)
    plt.title(f"{model_str} Intra-Cluster Acc: Original Clusters Only", fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def viz_merged_crosscluster_acc(cross_cluster_performance, model_str):
    # Visualization
    plt.figure(figsize=(12, 6))

    for cluster_id in cross_cluster_performance:
        # Extract valid iterations and performance
        data = cross_cluster_performance[cluster_id]
        valid_iterations = [it for it, perf in data if perf is not None]
        valid_performance = [perf for it, perf in data if perf is not None]
        if valid_iterations[0]==0:
            continue
        #print(valid_iterations)
        #print(valid_performance)
        #print()
        plt.plot(valid_iterations, valid_performance, label=f"Cluster {cluster_id}")

    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Cross-Cluster Test Accuracy", fontsize=18)
    plt.title(f"{model_str} Cross-Cluster Acc: Merged Clusters Only", fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def viz_orig_crosscluster_acc(cross_cluster_performance, model_str):
    # Visualization
    plt.figure(figsize=(12, 6))

    for cluster_id in cross_cluster_performance:
        # Extract valid iterations and performance
        data = cross_cluster_performance[cluster_id]
        valid_iterations = [it for it, perf in data if perf is not None]
        valid_performance = [perf for it, perf in data if perf is not None]
        if valid_iterations[0]!=0:
            continue
        #print(valid_iterations)
        #print(valid_performance)
        #print()
        plt.plot(valid_iterations, valid_performance, label=f"Cluster {cluster_id}")

    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Cross-Cluster Test Accuracy", fontsize=18)
    plt.title(f"{model_str} Cross-Cluster Acc: Original Clusters Only", fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def viz_crosscluster_acc_with_merges(cross_cluster_performance, merge_log, model_str):
    # Visualization with Merge Log and Connections
    plt.figure(figsize=(14, 8))

    # Dictionary to track the last valid point for each cluster
    last_points = {}

    # Plot original clusters
    for cluster_id in cross_cluster_performance:
        data = cross_cluster_performance[cluster_id]
        valid_iterations = [it for it, perf in data if perf is not None]
        valid_performance = [perf for it, perf in data if perf is not None]
        if valid_iterations[0] != 0:
            continue

        # Plot original cluster performance
        plt.plot(valid_iterations, valid_performance, label=f"Cluster {cluster_id}")
        last_points[cluster_id] = (valid_iterations[-1], valid_performance[-1])  # Store the last point

    # Handle merged clusters and connect original clusters
    for iteration, cluster1, cluster2, _, new_cluster in merge_log:
        # Plot and connect the merged clusters
        for cluster in [cluster1, cluster2]:
            if cluster in cross_cluster_performance:
                data = cross_cluster_performance[cluster]
                merge_perf = next((perf for it, perf in data if it == iteration), None)
                if merge_perf is not None:
                    plt.scatter(iteration, merge_perf, color='red', marker='^')#, label=f"Merge {cluster} → {new_cluster}")
                    
            if cluster in last_points:  # If it's an original cluster
                last_iteration, last_perf = last_points[cluster]

                # Connect to the newly merged cluster
                if new_cluster in cross_cluster_performance:
                    new_data = cross_cluster_performance[new_cluster]
                    valid_iterations = [it for it, perf in new_data if perf is not None and it >= iteration]
                    valid_performance = [perf for it, perf in new_data if perf is not None and it >= iteration]

                    if valid_iterations:
                        # Draw a line connecting the original cluster to the new merged cluster
                        plt.plot(
                            [last_iteration, valid_iterations[0]],
                            [last_perf, valid_performance[0]],
                            linestyle='--', color='gray'
                        )

                        # Continue plotting the merged cluster's performance
                        plt.plot(valid_iterations, valid_performance, linestyle='--')

                    # Update the last points for the newly merged cluster
                    if valid_iterations:
                        last_points[new_cluster] = (valid_iterations[-1], valid_performance[-1])

    # Add labels, legend, and formatting
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Cross-Cluster Test Accuracy", fontsize=18)
    plt.title(f"{model_str} Cross-Cluster Acc with Merge Connections", fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def viz_intracluster_acc_with_merges(intra_cluster_performance, merge_log, model_str):
    # Visualization with Merge Log and Connections
    plt.figure(figsize=(14, 8))

    # Dictionary to track the last valid point for each cluster
    last_points = {}

    # Plot original clusters
    for cluster_id in intra_cluster_performance:
        data = intra_cluster_performance[cluster_id]
        valid_iterations = [it for it, perf in data if perf is not None]
        valid_performance = [perf for it, perf in data if perf is not None]
        if valid_iterations[0] != 0:
            continue

        # Plot original cluster performance
        plt.plot(valid_iterations, valid_performance, label=f"Cluster {cluster_id}")
        last_points[cluster_id] = (valid_iterations[-1], valid_performance[-1])  # Store the last point

    # Handle merged clusters and connect original clusters
    for iteration, cluster1, cluster2, _, new_cluster in merge_log:
        # Plot and connect the merged clusters
        for cluster in [cluster1, cluster2]:
            if cluster in intra_cluster_performance:
                data = intra_cluster_performance[cluster]
                merge_perf = next((perf for it, perf in data if it == iteration), None)
                if merge_perf is not None:
                    plt.scatter(iteration, merge_perf, color='red', marker='^')#, label=f"Merge {cluster} → {new_cluster}")
                    
            if cluster in last_points:  # If it's an original cluster
                last_iteration, last_perf = last_points[cluster]

                # Connect to the newly merged cluster
                if new_cluster in intra_cluster_performance:
                    new_data = intra_cluster_performance[new_cluster]
                    valid_iterations = [it for it, perf in new_data if perf is not None and it >= iteration]
                    valid_performance = [perf for it, perf in new_data if perf is not None and it >= iteration]

                    if valid_iterations:
                        # Draw a line connecting the original cluster to the new merged cluster
                        plt.plot(
                            [last_iteration, valid_iterations[0]],
                            [last_perf, valid_performance[0]],
                            linestyle='--', color='gray'
                        )

                        # Continue plotting the merged cluster's performance
                        plt.plot(valid_iterations, valid_performance, linestyle='--')

                    # Update the last points for the newly merged cluster
                    if valid_iterations:
                        last_points[new_cluster] = (valid_iterations[-1], valid_performance[-1])

    # Add labels, legend, and formatting
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Intra-Cluster Test Accuracy", fontsize=18)
    plt.title(f"{model_str} Intra-Cluster Acc with Merge Connections", fontsize=18)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
