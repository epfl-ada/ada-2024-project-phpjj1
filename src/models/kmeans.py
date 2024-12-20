import numpy as np
from sklearn.cluster import KMeans


def find_best_init(X, optimal_k):
    km = KMeans(
        n_clusters=optimal_k,
        init='k-means++',
        n_init=1000,    # number of initial centroid sets
        random_state=42 # ensures reproducibility
    )
    km.fit(X)
    # The best run is now stored in km. 
    # You have reproducible results and the best solution without manual looping.
    labels = km.labels_
    inertia = km.inertia_
    print("Best inertia:", inertia)
    return km


def calculate_cluster_ratings(df, X, cluster_algorithm, EMOTIONS):
    km = cluster_algorithm
    df_cluster_ratings = X.drop(columns=EMOTIONS)
    df_cluster_emotions = X.copy()
    df_cluster_emotions["cluster"] = km.labels_
    df_cluster_emotions = df_cluster_emotions.groupby("cluster").mean().reset_index()
    df_cluster_ratings["cluster"] = km.labels_
    df_cluster_ratings["mean_ratings"] = df.loc[df_cluster_ratings.index, "mean_ratings"]
    df_cluster_ratings["count_ratings"] = df.loc[df_cluster_ratings.index, "count_ratings"]

    column_std = []
    column_mean = []
    unique = []
    for cluster in (np.unique(km.labels_)):
        column_std.append(df_cluster_ratings[df_cluster_ratings["cluster"] == cluster]["mean_ratings"].std())
        column_mean.append(df_cluster_ratings[df_cluster_ratings["cluster"] == cluster]["mean_ratings"].mean())
        unique.append(cluster)

    df_cluster_catings_grouped = df_cluster_ratings.groupby("cluster").mean()
    df_cluster_catings_grouped_counts = df_cluster_ratings.groupby('cluster').agg({
        'mean_ratings': 'count', "cluster": "count"  # Counts non-NaN entries in the 'ratings' column for each cluster
    })
    df_cluster_catings_grouped["rating_movies_count"] = df_cluster_catings_grouped_counts["mean_ratings"]
    df_cluster_catings_grouped["movies_count"] = df_cluster_catings_grouped_counts["cluster"]
    return df_cluster_emotions, df_cluster_catings_grouped
