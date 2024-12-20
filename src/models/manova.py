from statsmodels.multivariate.manova import MANOVA


########## Q1 ##########
def manova_genre_emotion_mean_df(emotions_without_neutral, top_genres_df):
    top_genres_df['Genres'] = top_genres_df['Genres'].astype('category')
    manova = MANOVA.from_formula(f'{ " + ".join(emotions_without_neutral) } ~ Genres', data=top_genres_df)
    result = manova.mv_test()
    return result
