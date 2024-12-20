# Includes methods that are used in the data pipeline
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

#Function for the box and jenkins procedure
import itertools
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# Helper function to replace the dictionaries with lists
def str_to_list(str):
    return ast.literal_eval(str)



# Helper function to replace the dictionaries with lists
def conv_to_dict(val): 
    try:
        return ast.literal_eval(val) if pd.notna(val) else {}
    except (ValueError, SyntaxError):
        return {}


# Helper function to parse values of a given parameter to a list
def get_list(x):
    return list(x.values())


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




def box_jenkins_procedure(data: pd.DataFrame, genre: str):
    tones = data.columns[1:]
    arima_results = {}
    warnings.filterwarnings("ignore")
    for tone in tones:
        print("##########################################\n",
              f"Performing Box-Jenkins Procedure for {tone} in {genre} Genre")
        series = data[tone]
        adf = adfuller(series)
        print(f"ADF Statistic: {adf[0]}")
        print(f"p-value: {adf[1]}")
        d = 0
        if adf[1] > 0.05:
            print("The series is not stationary and needs to be differenced.")
            d = 1
        
        p = range(1,4) # 1,2,3
        q = range(0,3) # 0,1,2
        best_aic = np.inf
        best_order = None
        best_model = None

        for p, q in itertools.product(p,q):
            model = ARIMA(series, order=(p,d,q))
            result = model.fit()
            if result.aic < best_aic:
                best_aic = result.aic
                best_order = (p,d,q)
                best_model = result
        print(f"Best ARIMA Order for {tone} in {genre} Genre: {best_order}")
        arima_results[tone] = best_model
    return arima_results


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
        #Save the plot in this directory proj/data/Q2_plots/acf_pacf/
        plt.savefig(f"../../data/Q2_plots/acf_pacf/{genre}_{tone}_acf_pacf.png")
        plt.show()

def forecast_series(data: pd.DataFrame, genre: str, results: dict):

    tones = data.columns[1:]  
    palette = sns.color_palette("bright", len(tones))  
    forecast_steps = 10
    future_years = [data["merge_year"].iloc[-1] + i for i in range(1, forecast_steps + 1)]
    
    forecast_df = data.copy()
    
    for tone, color in zip(tones, palette):
        if tone not in results:
            print(f"Warning: No ARIMA model found for {tone}. Skipping.")
            continue
        
        best_model = results[tone]  

        
        forecast = best_model.get_forecast(steps=forecast_steps)
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()

        
        new_rows = pd.DataFrame({
            "merge_year": future_years,
            tone: forecast_mean
        })
        forecast_df = pd.concat([forecast_df, new_rows], ignore_index=True)

      
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=data, x="merge_year", y=tone, label=f"Original {tone}", color=color)
        sns.lineplot(
            x=future_years,
            y=forecast_mean,
            label=f"Forecast {tone}",
            color=color,
            linestyle="--"
        )
        plt.fill_between(
            future_years,
            forecast_ci.iloc[:, 0],
            forecast_ci.iloc[:, 1],
            color=color,
            alpha=0.2,
            label=f"Confidence Interval ({tone})"
        )
        plt.title(f"{tone.capitalize()} Forecast for {genre} Genre")
        plt.xlabel("Year")
        plt.ylabel("Emotion Value")
        plt.legend(title="Tone", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.grid()
        plt.savefig(f"../../data/Q2_plots/forecast/{genre}_{tone}_forecast.png")
        plt.show()
    
    return forecast_df





    