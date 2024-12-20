import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pandas.plotting import parallel_coordinates
from scipy.cluster.hierarchy import dendrogram
from models.kmeans import *


########### Q1 ###########
def plot_top_genres(genre_count):
    """
    Plots the counts for top 40 genres on a horizontal bar plot
    """
    plt.figure(figsize=(10, 8))
    genre_count[: 40].sort_values().plot(kind='barh')
    plt.title('Top 40 genres')
    plt.xlabel('Number of movies')
    plt.ylabel('Genres')
    plt.show()


def plot_heatmap(genre_emotion_mean_df):
    """
    Plots heatmap between Genres and Emotions
    """ 
    plt.figure(figsize=(14, 10))
    sns.heatmap(genre_emotion_mean_df.drop('count', axis=1), annot=True, fmt='.2f', cbar=True, cmap='YlOrRd')
    plt.title('Heatmap Between Genres and Emotions')
    plt.ylabel('Genre')
    plt.xlabel('Emotion')
    plt.show()


def plot_emotion_scores_by_genre(genre_emotion_mean_df, weight_avg, emotions):
    """
    Plots emotion scores across different movie genres with a comparison to the weighted average for each emotion.
    """
    n_emotions = len(emotions)
    n_cols = 4  # Number of columns for subplots
    n_rows = -(-n_emotions // n_cols)  # Calculate rows needed, using ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))  # Adjust figure size for readability
    axes = axes.flatten()  # Flatten axes to easily iterate over them

    # Iterate through emotions and create subplots
    for i, emotion in enumerate(emotions):
        ax = axes[i]
        genre_emotion_mean_df[emotion].plot(kind='bar', ax=ax, color='skyblue', title=f"{emotion.capitalize()} Scores Across Movie Genres")
        ax.axhline(y=weight_avg[emotion], color='r', linestyle='--', label='Average Score')
        ax.set_xlabel('Genres')
        ax.set_ylabel(f"{emotion.capitalize()} Score")
        ax.set_xticks(range(len(genre_emotion_mean_df)))
        ax.set_xticklabels(genre_emotion_mean_df.index, rotation=45, ha='right')
        ax.grid(axis='y')
        ax.legend()

    # Turn off empty subplots
    for j in range(len(emotions), len(axes)):
        axes[j].axis('off')

    plt.suptitle("Emotion Scores by Genre: Comparison with Weighted Average", fontsize=16)
    plt.tight_layout()
    plt.show()


def find_significant_emotions_by_genre(genre_emotion_mean_df, weight_avg, emotions):
    """
    Identifies and visualizes statistically significant emotions by genre, based on a one-sample t-test.
    It compares the mean emotion scores per genre against a specified average (weight_avg) and shows emotions that show significant differences from the average score.
    """
    # Calculate statistically significant emotions
    stat_significant_emotions = {}
    for genre in genre_emotion_mean_df.index:
        genre_emotions = genre_emotion_mean_df.loc[genre, emotions]
        significant_emotions = []
        for emotion in emotions:
            # Perform a one-sample t-test
            stat, p_value = ttest_1samp(genre_emotion_mean_df[emotion], genre_emotions[emotion])
            if p_value / 2 < 0.05 and genre_emotions[emotion] > weight_avg[emotion]:
                significant_emotions.append(emotion)
        stat_significant_emotions[genre] = significant_emotions

    # Create subplots
    n_genres = len(genre_emotion_mean_df.index)
    n_cols = 4  # Number of columns for subplots
    n_rows = (n_genres + n_cols - 1) // n_cols  # Calculate rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), squeeze=False)
    axes = axes.flatten()  # Flatten axes for easy indexing

    for i, (genre, significant_emotions) in enumerate(stat_significant_emotions.items()):
        genre_emotions = genre_emotion_mean_df.loc[genre, emotions]
        ax = axes[i]
        genre_emotions.plot(
            kind='bar', 
            color=['green' if emotion in significant_emotions else 'gray' for emotion in emotions], 
            ax=ax
        )
        ax.set_title(f"Statistically Significant Emotions for {genre}")
        ax.set_xlabel("Emotions")
        ax.set_ylabel("Score")
        ax.set_xticks(np.arange(len(emotions)))
        ax.set_xticklabels(emotions, rotation=0)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Statistically Significant Emotions by Genre", fontsize=16)
    plt.tight_layout()
    plt.show()

    
########### Q2 ###########
def plot_emotions_by_time(emotion_by_time):
    """
    Plots the time series of different emotions in a single plot.
    """
    plt.figure(figsize=(12, 6))
    for column in emotion_by_time.columns:
        plt.plot(emotion_by_time.index, emotion_by_time[column], label=column)
    plt.legend(title='Emotions', loc='best')
    plt.title('Emotions Over Time')  # Title for the plot
    plt.xlabel('Year')  # Label for the x-axis
    plt.ylabel('Emotion')  # Label for the y-axis


def plot_indiv_emotions_by_time(emotions_by_time):
    """
    Plots the time series of individual emotions in subplots.
    """
    num_emotions = len(emotions_by_time.columns)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    colors = sns.color_palette("tab10", len(emotions_by_time.columns))

    for ax, column, color in zip(axes, emotions_by_time.columns, colors):
        ax.plot(emotions_by_time.index, emotions_by_time[column], color=color)
        ax.set_title(column)

    for ax in axes[num_emotions: ]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def timeseries_plots(data: pd.DataFrame, genre: str):
    tones = data.columns[1:]
    palette = sns.color_palette("bright", len(tones))

    plt.figure(figsize=(12, 6))
    for tone, color in zip(tones, palette):
        sns.lineplot(data=data, x="merge_year", y=tone, label=tone, color= color)
    plt.title(f"{genre} Emotions Over Time")
    plt.xlabel("Year")
    plt.ylabel("Emotion Value")
    plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.grid()
    plt.show()

    for tone, color in zip(tones, palette):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        plot_acf(data[tone], lags = 20, ax = axes[0], title = f"ACF of {tone} in {genre} Genre", color = color)
        plot_pacf(data[tone], lags = 20, ax = axes[1], title = f"PACF of {tone} in {genre} Genre", color = color)

        axes[0].lines[0].set_color(color)
        axes[1].lines[0].set_color(color)
        plt.tight_layout()
        plt.show()


########### Q4 ###########
def plot_language_distribution(language_count):
    """
    Plots distribution of languages.
    """
    language_count[:30].plot(kind='bar', figsize=(15, 7), logy=True)
    plt.title('Distribution of Languages in Movies (Log Scale)')
    plt.xlabel('Languages')
    plt.ylabel('Number of Movies')
    plt.show()


def plot_language_pie_chart(language_count):
    """
    Creates a pie chart of the top 15 languages plus "Other"
    """
    plt.figure(figsize=(25, 25))

    # Get top 15 languages and sum the rest into "Other"
    top_15 = language_count[:15]
    other = pd.Series({'Other': language_count[15:].sum()})
    plot_data = pd.concat([top_15, other])

    colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))

    plt.pie(plot_data, labels=plot_data.index, autopct='%1.1f%%', colors=colors, textprops={'fontsize': 12})
    plt.title('Distribution of Languages in Movies')
    plt.show()


def plot_emotion_language_distribution(df_languages, language_count):
    """
    Outputs top three emotions for each language using box plots.
    """
    # Define a color palette for the emotions
    emotion_palette = {
        'anger': 'red',
        'disgust': 'green',
        'sadness': 'blue',
        'fear': 'purple',
        'joy': 'yellow',
        'surprise': 'orange'
    }

    # We analyze the emotions of the top 14 languages
    top_languages = language_count[:14].index

    # For each language, get all the movies that belong to it and plot emotion distributions
    fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(20, 25))
    axes = axes.flatten()

    for i, lang in enumerate(top_languages):
        movies_in_lang = df_languages[df_languages['Languages'].apply(lambda x: lang in x)]
        distilbert_emotions = movies_in_lang['distilbert_emotions'].dropna()

        # Convert list of dicts to DataFrame for easier plotting
        distilbert_emotions_df = pd.DataFrame(list(distilbert_emotions))

        # Remove neutral emotion
        distilbert_emotions_df = distilbert_emotions_df.drop('neutral', axis=1)

        # Get top 3 emotions by mean value
        top_3_emotions_distilbert = distilbert_emotions_df.mean().nlargest(3).index

        # Plot boxplot for this language with only top 3 emotions
        sns.boxplot(data=distilbert_emotions_df[top_3_emotions_distilbert], ax=axes[i], palette=emotion_palette)
        axes[i].set_title(f'Top 3 Emotions Distribution for {lang}')
        axes[i].set_ylim(0, 1)

    plt.suptitle('Emotional Distributions for Most Popular Movie Languages', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_significant_language_per_emotion(regression_results, EMOTIONS):
    """
    Plots significant parameters for each emotion from regressions in Q4.
    """
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    axes = axes.flatten()

    # Find global min and max for consistent scale
    all_params = []
    for emotion in EMOTIONS:
        significant_params = regression_results[emotion]['significant_params']
        params_no_intercept = significant_params[1:]
        all_params.extend(params_no_intercept.values)
    global_min, global_max = min(all_params), max(all_params)

    for i, emotion in enumerate(EMOTIONS):
        if i < len(axes):
            significant_params = regression_results[emotion]['significant_params']
            
            # Don't include intercept
            params_no_intercept = significant_params[1:]  
            params_sorted = params_no_intercept.sort_values(ascending=True)

            y_labels = params_sorted.index.astype(str).str.extract(r'\[T\.(.*?)\]')[0]

            # Create bar plot
            sns.barplot(x=params_sorted.values, 
                    y=y_labels, 
                    ax=axes[i])
            
            axes[i].set_title(f'{emotion.capitalize()}\nR² = {regression_results[emotion]["r_squared"]:.3f}')
            axes[i].set_xlabel('Effect size (%)')
            axes[i].set_ylabel('Language')
            axes[i].set_xlim(global_min, global_max)  # consistent scale

    for j in range(len(EMOTIONS), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()


########### Q5 ###########
def plot_stddized_emotion_comparison(df_emotions, df_emotions_standardized):
    """
    Produces box plots that compare emotion intensities between standardized and unstandardized datasets. 
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6)) 
    long_emotions = pd.melt(df_emotions, var_name='Emotion', value_name='Emotion Intensity')
    long_emotions_standardized = pd.melt(df_emotions_standardized , var_name='Emotion', value_name='Emotion Intensity')
    ax1= sns.boxplot(x="Emotion", y="Emotion Intensity", ax=axes[0], data=long_emotions)
    ax1.set_title("Raw")
    ax2 = sns.boxplot(x="Emotion", y="Emotion Intensity", ax=axes[1],data=long_emotions_standardized)
    ax2.set_title("Standardized")


def plot_emotion_density_comparison(df_emotions, df_emotions_standardized):
    """

    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    axes = axes.flatten()
    for emotion in df_emotions.columns:
        ax1 = sns.kdeplot(data=df_emotions, x=emotion, ax=axes[0], label=emotion )
        ax1.set_title('Raw Emotions Density Plot')
        ax1.legend()
    for emotion in df_emotions_standardized.columns:
        ax2 = sns.kdeplot(data=df_emotions_standardized, x=emotion, ax=axes[1], label=emotion )
        ax2.set_title('Standardized Emotions Density Plot')
        ax2.legend()


def plot_ratings_distribution(df_emotions, df, EMOTIONS):
    '''
    Plots distributions of ratings.
    '''
    df_ratings = df_emotions.drop(columns=EMOTIONS)
    df_ratings["mean_ratings"] = df.loc[df_emotions.index, "mean_ratings"]

    sns.histplot(df_ratings['mean_ratings'], bins=10, kde=True, color='green')
    plt.title('Distribution of Movie Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.show()

    print(df_ratings["mean_ratings"].describe())


def plot_silhouette_scores(df_emotions_standardized):
    """
    Computes and plots silhouette score for different K.
    """
    silhouettes = []
    # Try multiple k
    for k in range(2, 21):
        # Cluster the data and assigne the labels
        labels = KMeans(n_clusters=k, random_state=10).fit_predict(df_emotions_standardized)
        # Get the Silhouette score
        score = silhouette_score(df_emotions_standardized, labels)
        silhouettes.append({"k": k, "score": score})
        
    # Convert to dataframe
    silhouettes = pd.DataFrame(silhouettes)

    # Plot the data
    plt.plot(silhouettes.k, silhouettes.score)
    plt.xlabel("K")
    plt.ylabel("Silhouette score")


def plot_sse(df_emotions_standardized, start=2, end=21):
    """
    Plots sum of squared errors for a range of K's.
    """
    sse = []
    for k in range(start, end):
        # Assign the labels to the clusters
        kmeans = KMeans(n_clusters=k, random_state=10).fit(df_emotions_standardized)
        sse.append({"k": k, "sse": kmeans.inertia_})

    sse = pd.DataFrame(sse)
    # Plot the data
    plt.plot(sse.k, sse.sse)
    plt.xlabel("K")
    plt.ylabel("Sum of Squared Errors")


def clustering_pca_visualizations(df_emotions_standardized, km_best_init, n_components=4):
    """
    Plots visualizations of clusters from PCA dimensions.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(df_emotions_standardized)

    COLUMNS = 3
    ROWS = 2
    fig, axs = plt.subplots(ROWS, COLUMNS, figsize=(10,8), sharey=True, sharex=True)
    combinations = [[1, 2],[1, 3],[1, 4],[2, 3],[2, 4],[3, 4]]

    for i, combination in enumerate(combinations):
        current_column = i % COLUMNS
        current_row = (i)//COLUMNS
        ax = axs[current_row, current_column]

        ax.scatter(X_pca[:,0], X_pca[:,1], c=km_best_init.labels_, alpha=0.6)
        ax.set_title(f"PCA {combination[0]} and PC {combination[1]}")
        plt.xlabel(f'Principal Component {combination[0]}')
        plt.ylabel(f'Principal Component {combination[1]}')
        cbar = plt.colorbar(ax.collections[0], ax=ax, label='Cluster')
        # Plot the centroids
        for c in km_best_init.cluster_centers_:
            ax.scatter(c[0], c[1], marker="+", color="red")
    plt.tight_layout()


def clustering_tsne_visualizations(df_emotions_standardized, km_best_init, n_components=2):
    """
    Plots visualizations of clusters from TSNE dimensions.
    """
    X_tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=10).fit_transform(df_emotions_standardized)

    COLUMNS = 3
    ROWS = 2
    fig, axs = plt.subplots(ROWS, COLUMNS, figsize=(10,8), sharey=True, sharex=True)
    combinations = [[1, 2],[1, 3],[1, 4],[2, 3],[2, 4],[3, 4]]

    for i, combination in enumerate(combinations):
        current_column = i % COLUMNS
        current_row = (i)//COLUMNS
        ax = axs[current_row, current_column]


        ax.scatter(X_tsne[:,0], X_tsne[:,1], c=km_best_init.labels_, alpha=0.6)
        ax.set_title(f"t-SNE {combination[0]} and t-SNE {combination[1]}")
        plt.xlabel(f't-SNE {combination[0]}')
        plt.ylabel(f't-SNE {combination[1]}')
        cbar = plt.colorbar(ax.collections[0], ax=ax, label='Cluster')
        # Plot the centroids
        for c in km_best_init.cluster_centers_:
            ax.scatter(c[0], c[1], marker="+", color="red")

    plt.tight_layout()


def plot_parallel_coordinates(df_cluster_emotions):
    plt.figure(figsize=(10, 5))
    parallel_coordinates(df_cluster_emotions, class_column='cluster', colormap=plt.get_cmap("Set2"))
    plt.title("Parallel Coordinates Plot of Clusters' Emotion Profiles")
    plt.ylabel("Emotion Value")
    plt.legend()
    plt.show()


def plot_radar_chart(data, title):
    """

    """
    categories = data.columns
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, row in data.iterrows():
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, label=f'Cluster {i}')
        ax.fill(angles, values, alpha=0.25)

    ax.set_title(title, size=15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.show()


def plot_dendrogram(cluster_df, linkage_matrix):
    """
    Plots dendrogram for hierarchial clustering.
    """
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, labels=cluster_df.index, orientation='top')
    plt.title('Hierarchical Clustering Dendrogram (Average Linkage, Cosine Distance) on Sample')
    plt.xlabel('Films')
    plt.ylabel('Cosine Distance')
    plt.show()


def plot_radar_chart_sub_plots(data, title, ax):
    categories = data.columns
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    # Plot each row of data
    for i, row in data.iterrows():
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, label=f'Cluster {i}')
        ax.fill(angles, values, alpha=0.25)

    # Set title and tick labels
    ax.set_title(title, size=15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    # Optionally adjust the radial limits
    # ax.set_ylim(0, some_max_value)
    ax.grid(True)


def plot_radar_per_epsilon(df, df_emotions, EMOTIONS, positive=False):
    ROWS = 3
    COLUMNS = 3
    fig, axs = plt.subplots(ROWS, COLUMNS, figsize=(15, 10), subplot_kw=dict(polar=True))

# Flatten axs for easier indexing if needed
    axs = axs.ravel()

# Example combinations or parameters for each subplot
    combinations = [0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07]

    for i, eps_val in enumerate(combinations):
        ax = axs[i]
        dbscan = DBSCAN(eps=eps_val, min_samples=2)
        dbscan_raw = dbscan.fit(df_emotions)
        df_cluster_emotions, cluster_ratings = calculate_cluster_ratings(df, df_emotions, dbscan_raw, EMOTIONS)
        merged_inner = pd.merge(df_cluster_emotions, cluster_ratings, on='cluster', how='inner')
        # Filter for the clusters you want to plot
        if positive == False:
            filtered_clusters_raw = merged_inner[(merged_inner['rating_movies_count'] > 3) & 
                                                (merged_inner['mean_ratings'] < 2.5)]
        else:
            filtered_clusters_raw = merged_inner[(merged_inner['rating_movies_count'] > 3) & 
                                                (merged_inner['mean_ratings'] > 3.5)]

        # Plot the radar chart on the given subplot axis
        plot_radar_chart_sub_plots(filtered_clusters_raw[EMOTIONS],
                                f"Cluster Emotional Profiles (eps={eps_val})",
                                ax)

    plt.tight_layout()
    plt.show()
