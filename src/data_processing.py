import pandas as pd
import numpy as np
import os
import io
import requests
from sklearn.datasets import fetch_california_housing
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_salary_data(filepath=None):
    if filepath is None:
        filepath = os.path.join(BASE_DIR, 'data', 'ds_salaries.csv')
    try:
        df = pd.read_csv(filepath)
        df['job_title'] = df['job_title'].replace('ML Engineer', 'Machine Learning Engineer')
        targets = ['Data Scientist', 'Data Engineer', 'Data Analyst']
        df_filtered = df[df['job_title'].isin(targets)]
        arrays = [df_filtered[df_filtered['job_title'] == group]['salary_in_usd'].dropna().values for group in targets]
        return df_filtered, arrays, 'job_title', 'salary_in_usd'
    except FileNotFoundError:
        return None, None, None, None

def load_game_data(filepath=None):
    if filepath is None:
        filepath = os.path.join(BASE_DIR, 'data', 'vgsales.csv')
    try:
        df = pd.read_csv(filepath)
        targets = ['Action', 'Sports', 'Shooter', 'Role-Playing', 'Misc']
        df_filtered = df[df['Genre'].isin(targets)]
        arrays = [df_filtered[df_filtered['Genre'] == group]['Global_Sales'].dropna().values for group in targets]
        return df_filtered, arrays, 'Genre', 'Global_Sales'
    except FileNotFoundError:
        return None, None, None, None

def load_airbnb_data(filepath=None):
    if filepath is None:
        filepath = os.path.join(BASE_DIR, 'data', 'AB_NYC_2019.csv')
    try:
        df = pd.read_csv(filepath)
        df_shared = df[df['room_type'] == 'Shared room']
        targets = ['Queens', 'Bronx', 'Staten Island']
        df_filtered = df_shared[df_shared['neighbourhood_group'].isin(targets)]
        arrays = [df_filtered[df_filtered['neighbourhood_group'] == group]['price'].dropna().values for group in targets]
        return df_filtered, arrays, 'neighbourhood_group', 'price'
    except FileNotFoundError:
        return None, None, None, None

def load_california_1samp():
    california = fetch_california_housing()
    house_prices = california.target
    prices_standardized = (house_prices - np.mean(house_prices)) / np.std(house_prices)
    return prices_standardized

def load_california_spatial():
    california = fetch_california_housing(as_frame=True)
    df = california.frame
    high_income = df[df['MedInc'] > df['MedInc'].quantile(0.75)][['Longitude', 'Latitude']].values
    low_income = df[df['MedInc'] < df['MedInc'].quantile(0.25)][['Longitude', 'Latitude']].values
    
    np.random.seed(42)
    high_income_sample = high_income[np.random.choice(high_income.shape[0], 500, replace=False)]
    low_income_sample = low_income[np.random.choice(low_income.shape[0], 500, replace=False)]
    return high_income_sample, low_income_sample

def load_secom_data():
    url_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
    url_labels = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"
    response_data = requests.get(url_data, verify=False)
    response_labels = requests.get(url_labels, verify=False)
    
    df_secom = pd.read_csv(io.StringIO(response_data.text), sep=" ", header=None)
    df_labels = pd.read_csv(io.StringIO(response_labels.text), sep=" ", header=None, usecols=[0])
    df_secom['Result'] = df_labels[0]
    
    sensor_good = df_secom[df_secom['Result'] == -1][59].dropna().values
    sensor_bad = df_secom[df_secom['Result'] == 1][59].dropna().values
    return sensor_good, sensor_bad