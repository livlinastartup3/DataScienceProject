import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


def plot_random_forest_model(df: pd.DataFrame, start_date: str, end_date: str,
                              n_estimators=5, random_state=42, test_size=0.30, confidence_level=0.95) -> None:
    
    """
    This function takes the clean dataset and trains a Random Forest model to fit the
    inbound dataset.

    Arguments: this function expects a dataframe NOT csv file as a string where the file is located, also a start and end date to train the model
    ---------

    Returns: None
    -------


    Example:
    -------
    data_file = r"C:\Users\frifi\OneDrive\Desktop\School Mat UHasselt\Third Semester\Project Data Science\Data sets\Outbound\Outbound_all.csv"
    start = '2023-01-25'
    end = '2023-06-30'
    plot_random_forest_model(data_file, start, end)

    """

    columns_to_remove = ['Workload(hrs)', 'year', 'Date', 'Actual number of workers', 'Frigo_Temp pallets',
                         'Truck Frequency', 'Trucks_in_Inbound', 'Trucks_in_Frigo', 'Total_trucks']
    # Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.strftime('%B').map(month_mapping)
    df['year'] = df['Date'].dt.year

    # Model fitting
    df_copy = df.copy()
    df_prediction = df_copy
    X = df_prediction.drop(columns=columns_to_remove)
    y = df_prediction['Workload(hrs)']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=10)
    est_rf = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state)
    est_rf.fit(X_train, y_train)
    rf_pred = est_rf.predict(X_test)

    # Accuracy metric
    train_score = est_rf.score(X_train, y_train)
    validation_score = est_rf.score(X_test, y_test)
    cross_validation_score = cross_val_score(
        est_rf, X_train, y_train, cv=5).mean()
    r2_Score = r2_score(rf_pred, y_test)
    MSE = np.sqrt(mean_squared_error(rf_pred, y_test))


    # Feature importance
    feature_scores = pd.Series(
        est_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    # Predictions
    rf_all = est_rf.predict(X)
    df_prediction["predicted_workload"] = rf_all
    mask = (df_prediction['Date'] >= start_date) & (
        df_prediction['Date'] <= end_date)

    # Confidence interval
    residuals = df_prediction['Workload(hrs)'] - rf_all
    mse = np.mean(residuals**2)
    df_prediction['mse'] = mse
    prediction_std = np.sqrt(mse)
    margin_of_error = prediction_std * 1.96
    lower_bound = rf_all - margin_of_error
    upper_bound = rf_all + margin_of_error
    df_prediction['lower_bound'] = lower_bound
    df_prediction['upper_bound'] = upper_bound

    # Visualization of the important features
    sns.barplot(x=feature_scores, y=feature_scores.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.show()

    # Visualization of the predictions
    plt.figure(figsize=(10, 8))
    plt.plot(df_prediction['Date'],
             df_prediction['Workload(hrs)'], label='Actual')
    plt.plot(df_prediction[mask]['Date'], rf_all[mask], label='Predicted')
    plt.fill_between(df_prediction[mask]['Date'], df_prediction[mask]['lower_bound'],
                     df_prediction[mask]['upper_bound'], alpha=0.5, color="green", label='Confidence Interval')
    plt.xlabel('Date')
    plt.ylabel('Workload(hrs)')
    plt.title('Overlaying Actual and Predicted Values in Inbound zone')
    plt.legend()
    plt.show()
