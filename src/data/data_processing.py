import pandas as pd


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
