import statsmodels.formula.api as smf
import statsmodels.api as sm


############## Q3 ##############
def character_emotion_regression(target, character_df):
    """
    Performs regression between character traits vs emotions.
    """
    covariates = '~ disgust + fear + anger + sadness + surprise + joy'
    age_mod = smf.ols(formula=f'{target} {covariates}', data=character_df)
    return age_mod.fit()


############## Q4 ##############
def run_individual_regressions(df_reg, REGRESSION_MODEL, EMOTIONS):
    """
    Perform separate regression analyses for each emotion.
    """
    results = {}
    for emotion in EMOTIONS:
        model = sm.OLS.from_formula(f"{emotion} ~ {REGRESSION_MODEL}", data=df_reg).fit()
        results[emotion] = {
            'r_squared': model.rsquared,
            'significant_effects': model.pvalues[model.pvalues < 0.05],
            'significant_params': model.params[model.pvalues < 0.05],
            'model': model
        }
    return results
