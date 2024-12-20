import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from utils.plots import timeseries_plots


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
        plt.show()
    
    return forecast_df


def genre_timeseries_analysis(genre_timeseries_df, genres_emotions_mapping):
    for genre, emotions in genres_emotions_mapping.items():
        genre_data = genre_timeseries_df[genre_timeseries_df["Genres"] == genre]
        genre_indiv_timeseries_df = genre_data[["merge_year"] + emotions].dropna()
        timeseries_plots(genre_indiv_timeseries_df, genre)
        genre_timeseries_result = box_jenkins_procedure(genre_indiv_timeseries_df, genre)
        if genre == 'Action':
            print(genre_timeseries_result['anger'].summary())
        forecast_series(genre_indiv_timeseries_df, genre, genre_timeseries_result)
