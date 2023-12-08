# import libraries

import numpy as np
import pandas as pd
from typing import List
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


def random_forest_model(data_file: str, start_date: str, end_date: str,
                        n_estimators=5, test_size=0.20, random_state=10) -> None:
    """
    This function takes the clean dataset and trains a Random Forest model to fit the
    dataset.

    Arguments: this function expects the path as a string where the file is located, also a start and end date to train the model
    ---------

    Returns: None
    -------


    Example:
    -------
    data_file = r"C:\Users\frifi\OneDrive\Desktop\School Mat UHasselt\Third Semester\Project Data Science\Data sets\Outbound\Outbound_all.csv"
    start = '2023-01-25'
    end = '2023-06-30'
    random_forest_model(data_file, start, end)

    """
    # Read the data
    data = pd.read_csv(data_file)

    # Preprocessing
    df_train = data
    numeric_df_train = df_train.select_dtypes(include=['number'])
    correlation_matrix = numeric_df_train.corr()
    plt.figure(figsize=(20, 5))
    sns.heatmap(correlation_matrix, annot=True, cmap='inferno')
    plt.title("Correlation Plot")

    columns_to_remove = ['Workload(hrs)', 'Order Freq', 'quarter',
                         'Date', 'Actual number of workers', 'Unnamed: 0']

    # Model fitting
    X = df_train.drop(columns=columns_to_remove)
    y = df_train['Workload(hrs)']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    est_rf = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state)
    est_rf.fit(X_train, y_train)
    rf_pred = est_rf.predict(X_test)
    
    print("Training Score : ", est_rf.score(X_train, y_train))
    print("Validation Score : ", est_rf.score(X_test, y_test))
    print("Cross Validation Score : ", cross_val_score(
        est_rf, X_train, y_train, cv=5).mean())
    print("R2_Score : ", r2_score(rf_pred, y_test))
    print('MSE: ', mean_squared_error(rf_pred, y_test))

    # Predictions
    df_prediction = df_train
    rf_all = est_rf.predict(X)

    # Mean square error and confidence interval
    start_Date = start_date
    end_Date = end_date
    mask = (df_prediction['Date'] >= start_Date) & (
        df_prediction['Date'] <= end_Date)
    residuals = df_prediction['Workload(hrs)'] - rf_all
    mse = np.mean(residuals**2)
    df_prediction['mse'] = mse
    prediction_std = np.sqrt(mse)
    margin_of_error = prediction_std * 1.96
    lower_bound = rf_all - margin_of_error
    upper_bound = rf_all + margin_of_error
    df_prediction['lower_bound'] = lower_bound
    df_prediction['upper_bound'] = upper_bound

    # Visualization of feature importance
    feature_scores = pd.Series(
        est_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    sns.barplot(x=feature_scores, y=feature_scores.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.show()

    # Visualization of the prediction
    plt.figure(figsize=(12, 8))
    plt.plot(df_prediction['Date'],
             df_prediction['Workload(hrs)'], label='Actual')
    plt.plot(df_prediction[mask]['Date'], rf_all[mask], label='Predicted')

    plt.fill_between(df_prediction[mask]['Date'], df_prediction[mask]['lower_bound'],
                     df_prediction[mask]['upper_bound'], alpha=0.3, color='green', label='Confidence Interval')
    plt.xlabel('Date')
    plt.ylabel('Workload')
    plt.title('Overlaying Actual and Predicted Values in Outbound zone')
    plt.legend()
    plt.show()
