import statsmodels.formula.api as smf


def character_emotion_regression(target, character_df):
    covariates = '~ disgust + fear + anger + sadness + surprise + joy'
    age_mod = smf.ols(formula=f'{target} {covariates}', data=character_df)
    return age_mod.fit()
