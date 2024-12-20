import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


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
