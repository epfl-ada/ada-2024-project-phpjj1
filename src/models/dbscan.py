import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from models.kmeans import *


def calculate_sum_of_within_cluster_distances(df_emotions, dbscan):
    """
    Calculates sum of of within cluster distances as a metric for cluster quality.
    """
    total = 0.0
    for label in np.unique(dbscan.labels_):
        if label != -1:  # Ignore noise
            members = df_emotions[dbscan.labels_ == label]
            centroid = members.mean(axis=0)
            distances = np.sqrt(((members - centroid) ** 2).sum(axis=1))
            avg_distance = distances.mean()
            total += avg_distance
    print ("total within cluster distance", total)
    return total


def dbscan_by_epsilon(df, df_emotions, EMOTIONS):
    """
    Performs DBSCAN for epsilon grids.
    """
    eps_list = np.linspace(0.01,0.3, 20)
    for i in range(0, len(eps_list)):
        print("=" * 50)
        print(f'Current epsilon: {eps_list[i]}')
        dbscan = DBSCAN(eps=eps_list[i], min_samples=2)
        dbscan_raw = dbscan.fit(df_emotions)
        labels = dbscan_raw.labels_
        valid_clusters = df_emotions[labels != -1]
        valid_labels = labels[labels != -1]
        sum_distance =  calculate_sum_of_within_cluster_distances(df_emotions, dbscan_raw)
        print ("Sum of within cluster distances", sum_distance)
        df_cluster_emotions, cluster_ratings = calculate_cluster_ratings(df, df_emotions, dbscan_raw, EMOTIONS)
        print("Number of clusters", len(cluster_ratings))
        print("Number of clusters with number of movies >5", (cluster_ratings['movies_count'] > 5).sum())


def filter_clusters_dbscan(df, df_emotions, EMOTIONS):
    """
    Filter specific clusters from DBSCAN configuration. 
    """
    dbscan = DBSCAN(eps=0.03, min_samples=2)
    dbscan_raw = dbscan.fit(df_emotions)
    df_cluster_emotions, cluster_ratings = calculate_cluster_ratings(df, df_emotions, dbscan_raw, EMOTIONS)
    merged_inner = pd.merge(df_cluster_emotions, cluster_ratings, on='cluster', how='inner')
    filtered_clusters_raw = merged_inner[(merged_inner['rating_movies_count'] > 3) & (merged_inner['mean_ratings'] <2.2)]
    return filtered_clusters_raw[['mean_ratings', "rating_movies_count"]]
