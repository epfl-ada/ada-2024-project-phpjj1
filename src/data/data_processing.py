import numpy as np
import pandas as pd
from utils.methods import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


################### Q1 ##################
def get_top_genres_df(df_with_plot, genre_count):
    relevant_genres = genre_count[genre_count > 2000].index
    temp = df_with_plot.explode('Genres')
    df_plot_genres = temp[temp['Genres'].isin(relevant_genres)].reset_index(drop=True)
    temp = pd.DataFrame(list(df_plot_genres['distilbert_emotions'].apply(conv_to_dict))).reset_index(drop=True)
    top_genres_df = pd.concat([df_plot_genres, temp], axis=1)
    return top_genres_df


def get_genre_emotion_mean_df(top_genres_df, emotions):
    temp = top_genres_df.groupby('Genres').agg({
        emotion: ['mean'] for emotion in emotions
    }).reset_index()
    temp['count'] = top_genres_df['Genres'].value_counts().sort_index().values
    temp.columns = ['Genres'] + emotions + ['count']
    genre_emotion_mean_df = temp.set_index('Genres')
    return genre_emotion_mean_df


################### Q2 ##################
def get_time_series_data(emotions_df, emotions):
    emotion_by_time = emotions_df.groupby('merge_year').agg({
        emotion: ['mean'] for emotion in emotions
    })
    emotion_by_time.columns = emotions
    return emotion_by_time


def get_movie_counts_by_time(emotions_df):
    movie_counts_by_time = emotions_df.groupby(['merge_year']).agg(
        counts=('merge_year', 'size')
    )
    return movie_counts_by_time


def get_timeseries_by_genre(emotions_df, genre_count, genres_emotions_mapping, emotions):
    columns_needed = ['Plot', 'Genres', 'merge_year']
    df_tone = emotions_df.dropna(subset=['Plot'])[columns_needed + emotions]

    relevant_genres = genre_count[genre_count>2000].index
    df_ex_gen = df_tone.explode('Genres')
    time_series_df = df_ex_gen[df_ex_gen['Genres'].isin(relevant_genres)].reset_index(drop=True)

    grouped_df = time_series_df.groupby(['Genres', 'merge_year'])[emotions].mean().reset_index()


    genre_timeseries_df = grouped_df[(grouped_df["Genres"].isin(genres_emotions_mapping.keys())) &
                            ((grouped_df["merge_year"] >= 1925) & (grouped_df["merge_year"] < 2012))]
    return genre_timeseries_df


################### Q3 ##################
def get_character_df(df, emotions):
    # Prepare the data for analysis
    # Filter for columns needed for the analysis
    character_df = df[['WikiID', 'merge_year', 'Genres', 'distilbert_emotions', 'ActorAge', 'ActorGender', 'ActorBirthDate', 'ActorGenderFlag']].copy()

    # Drop rows without emotions, age or gender because they are not helpful here
    print("Number of rows before droping: ", len(character_df))
    character_df = character_df.dropna(subset=['distilbert_emotions', 'ActorAge', 'ActorGenderFlag'])
    print("Number of rows after droping: ", len(character_df))

    # Parse relevant attributes from string to correct datatype
    character_df['ActorAge'] = character_df['ActorAge'].apply(str_to_list)
    character_df['ActorGender'] = character_df['ActorGender'].apply(str_to_list)
    character_df['ActorGenderFlag'] = character_df['ActorGenderFlag'].apply(str_to_list)

    # Check if transformations worked properly
    test_ages = character_df.iloc[0]['ActorAge']
    print(test_ages)
    test_gender = character_df.iloc[0]['ActorGender']
    print(test_gender)
    test_gender_flag = character_df.iloc[0]['ActorGenderFlag']
    print(test_gender_flag)
    print(f"Age entries: {len(test_ages)}, gender entries: {len(test_gender)}, gender flag entries: {len(test_gender_flag)}")
    # Create aggregations for gender and age of actors
    exploded_df = character_df.explode('ActorAge')
    exploded_df = exploded_df.explode('ActorGenderFlag')

    aggregated_df = exploded_df.groupby('WikiID').agg({
        "ActorAge": "mean",
        "ActorGenderFlag": "mean"
    })

    aggregated_df.rename(columns={
        'ActorAge': 'AgeAvg', 
        'ActorGenderFlag': 'GenderAvg'
    }, inplace=True)

    character_df = character_df.merge(aggregated_df, on='WikiID', how='left')

    # Parse emotions from string to dictionary
    character_df['distilbert_emotions'] = character_df['distilbert_emotions'].apply(conv_to_dict)

    # Check the result
    test_emotions = character_df.iloc[0]['distilbert_emotions']
    print(type(test_emotions))
    for emotion in emotions:
        character_df[emotion] = character_df['distilbert_emotions'].apply(lambda x: x[emotion])
    return character_df


################### Q4 ##################
def is_invalid_unicode(text: str):
    """Check if text contains invalid unicode characters"""
    try:
        text.encode('utf-8').decode('utf-8')
        return False
    except UnicodeError:
        return True


def clean_lang_list(languages: list[str]):
    """Remove invalid languages from a list of languages"""
    return [l for l in languages if not is_invalid_unicode(l)]


def prepare_data_for_analysis(df_to_prepare, EMOTIONS):
    """
    Prepare the data for statistical analysis by:
    1. Expanding the language lists into separate rows
    2. Creating emotion columns from the dictionary
    3. Filtering for languages with sufficient data
    """
    df_new = df_to_prepare.copy()

    # Get rid of rows without emotion data
    df_new = df_new[df_new['distilbert_emotions'].apply(lambda x: isinstance(x, dict) and x != {})]

    # Multiply all the emotions by 100 so we can work with percentages
    df_new['distilbert_emotions'] = df_new['distilbert_emotions'].apply(convert_to_percentage)

    # Explode the Languages column to create separate rows for each language
    df_new = df_new.explode('Languages').reset_index(drop=True)

    # Create separate columns for each emotion
    for emotion in EMOTIONS:
        df_new[emotion] = df_new['distilbert_emotions'].apply(lambda x: x.get(emotion, np.nan))

    # Filter for languages with at least 100 movies
    language_counts = df_new['Languages'].value_counts()
    valid_languages = language_counts[language_counts >= 100].index
    df_filtered = df_new[df_new['Languages'].isin(valid_languages)]
    return df_filtered


################### Q5 ##################
def standardize_emotions(df_emotions, EMOTIONS):
    standardizer = StandardScaler()
    x_standardized = standardizer.fit_transform(df_emotions)
    df_emotions_standardized = pd.DataFrame(x_standardized, columns=EMOTIONS)
    return df_emotions_standardized


def perform_pca(df_emotions, n_components=7):
    pca = PCA(n_components=7)
    pca.fit(df_emotions)

    explained_variance_ratio = pca.explained_variance_ratio_
    total_variance = sum(pca.explained_variance_ratio_)
    print(total_variance)
    print("Explained Variance Ratio per Principal Component: ", (explained_variance_ratio/total_variance))
