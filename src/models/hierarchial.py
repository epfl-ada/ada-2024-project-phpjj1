import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform


def hierarchial(df_hierarchial, EMOTIONS):
    sample_df = df_hierarchial.sample(500, random_state=42)
    cluster_df = sample_df[EMOTIONS].copy()

    distance_matrix = pdist(cluster_df, metric='cosine')
    linkage_matrix = linkage(distance_matrix, method='average')
    return sample_df, cluster_df, distance_matrix, linkage_matrix


def compute_inner_cluster_distances(labels, distances):
    unique_clusters = np.unique(labels)
    inner_distances = {}
    for cluster in unique_clusters:
        indices = np.where(labels == cluster)[0]
        if len(indices) > 1:
            intra_dist = distances[np.ix_(indices, indices)].mean()
        else:
            intra_dist = 0  # Single-point cluster
        inner_distances[cluster] = intra_dist
    return inner_distances


def assign_clusters(distance_matrix, linkage_matrix, cluster_df):
    cutoff = 0.36
    cluster_labels = fcluster(linkage_matrix, t=cutoff, criterion='distance')
    cluster_df['cluster'] = cluster_labels
    distance_square = squareform(distance_matrix)
    return cluster_labels, distance_square
