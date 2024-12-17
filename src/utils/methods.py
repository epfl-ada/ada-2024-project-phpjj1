# Includes methods that are used in the data pipeline
import ast
import pandas as pd

# Helper function to replace the dictionaries with lists
def str_to_list(str):
    return ast.literal_eval(str)


def conv_to_dict(val): 
    try:
        return ast.literal_eval(val) if pd.notna(val) else {}
    except (ValueError, SyntaxError):
        return {}


# Helper function to parse values of a given parameter to a list
def get_list(x):
    return list(x.values())


# def prepare_heatmap(emotions_df):
#     # Expand the genres into individual rows
#     emotions_df_expanded = emotions_df.explode('Genres')
#     
#     # Count the frequency of each genre
#     genre_counts = emotions_df_expanded['Genres'].value_counts()
#     
#     # Get the top 9 genres based on the frequency
#     top_genres = genre_counts.nlargest(9).index
#     
#     # Calculate the mean emotion for each genre
#     emotion_data = emotions_df_expanded.groupby('Genres')[['disgust', 'fear', 'anger', 'neutral', 'sadness', 'surprise', 'joy']].mean()
#     
#     # Group all other genres into 'Others'
#     emotion_data['Genre'] = emotion_data.index
#     emotion_data.loc[~emotion_data['Genre'].isin(top_genres), 'Genre'] = 'Others'
#     
#     # Aggregate the 'Others' genres and recalculate proportions
#     emotion_data_grouped = emotion_data.groupby('Genre').sum()
#     emotion_data_grouped = emotion_data_grouped.div(emotion_data_grouped.sum(axis=1), axis=0)
#     return emotion_data, emotion_data_grouped


# Converts a given date to datetime
def clean_date(date):
    try:
        return pd.to_datetime(date, errors='coerce')
    except:
        return pd.NaT
    

# Calculate age based on birth and movie release date, use mean if one of the dates is missing
def calculate_age(df, birth_date, movie_date):
    if pd.isna(birth_date):
        # Without birth date the age can't be calculate, replace age with mean
        return df['ActorAge'].mean()
    if pd.isna(movie_date):
        # Without movie release date the age can't be calculate, replace age with mean
        return df['ActorAge'].mean()
    age = movie_date.year - birth_date.year - ((movie_date.month, movie_date.day) < (birth_date.month, birth_date.day))
    return age


# Calculate age, if value is negative or too high return mean
def calculate_age_non_negative(df, birth_date, movie_date):
    if pd.isna(birth_date):
        # Without birth date the age can't be calculate, replace age with mean
        return df['ActorAge'].mean()
    if pd.isna(movie_date):
        # Without movie release date the age can't be calculate, replace age with mean
        return df['ActorAge'].mean()
    age = movie_date.year - birth_date.year - ((movie_date.month, movie_date.day) < (birth_date.month, birth_date.day))
    return age if 0 < age < 103 else df['ActorAge'].mean()
