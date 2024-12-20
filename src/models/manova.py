from data.data_processing import *
from statsmodels.multivariate.manova import MANOVA


########## Q1 ##########
def manova_genre_emotion_mean_df(emotions_without_neutral, top_genres_df):
    """
    Performs MANOVA test on Genres vs Emotions.
    """
    top_genres_df['Genres'] = top_genres_df['Genres'].astype('category')
    manova = MANOVA.from_formula(f'{ " + ".join(emotions_without_neutral) } ~ Genres', data=top_genres_df)
    result = manova.mv_test()
    return result


########## Q4 ##########
def manova_emotion_language(df_languages, EMOTIONS):
    """
    Run MANOVA for a multivariate analysis of the emotions.
    """
    manova_data = prepare_data_for_analysis(df_languages, EMOTIONS)
    manova = MANOVA.from_formula('anger + disgust + fear + joy + sadness + surprise ~ Languages', data=manova_data)
    return manova.mv_test()
