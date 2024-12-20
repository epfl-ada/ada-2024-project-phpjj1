# Includes methods that are used in the data pipeline
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def str_to_list(str):
    """
    Helper function to replace the strings with lists
    """
    return ast.literal_eval(str)


def conv_to_dict(val): 
    """
    Helper function to replace the strings with dictionaries
    """
    try:
        return ast.literal_eval(val) if pd.notna(val) else {}
    except (ValueError, SyntaxError):
        return {}


def get_list(x):
    """
    Helper function to parse values of a given parameter to a list
    """
    return list(x.values())


def clean_date(date):
    """
    Converts a given date to datetime
    """
    try:
        return pd.to_datetime(date, errors='coerce')
    except:
        return pd.NaT
    

def calculate_age(df, birth_date, movie_date):
    """
    Calculate age based on birth and movie release date, use mean if one of the dates is missing
    """
    if pd.isna(birth_date):
        # Without birth date the age can't be calculate, replace age with mean
        return df['ActorAge'].mean()
    if pd.isna(movie_date):
        # Without movie release date the age can't be calculate, replace age with mean
        return df['ActorAge'].mean()
    age = movie_date.year - birth_date.year - ((movie_date.month, movie_date.day) < (birth_date.month, birth_date.day))
    return age


def calculate_age_non_negative(df, birth_date, movie_date):
    """
    Calculate age, if value is negative or too high return mean
    """
    if pd.isna(birth_date):
        # Without birth date the age can't be calculate, replace age with mean
        return df['ActorAge'].mean()
    if pd.isna(movie_date):
        # Without movie release date the age can't be calculate, replace age with mean
        return df['ActorAge'].mean()
    age = movie_date.year - birth_date.year - ((movie_date.month, movie_date.day) < (birth_date.month, birth_date.day))
    return age if 0 < age < 103 else df['ActorAge'].mean()


################# Q4 #################
def convert_to_percentage(emotions):
    """
    Convert emotional decimal values to percentages.
    """
    return {k: v * 100 for k, v in emotions.items()}
