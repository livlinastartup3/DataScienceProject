# import libraries

import os
import pandas as pd
import numpy as np


def clean_and_combine(directory: str, bundle_path: str, item_overview_path: str) -> pd.DataFrame:
    # """
    # This function reads three datasets, merges them, and performs cleaning.

    # Parameters:
    # - directory (str): Path to the directory containing Orderlines dataset.
    # - bundle_path (str): Path to the Bundle dataset in xlsx format.
    # - item_overview_path (str): Path to the Item Overview dataset in xlsx format.

    # Returns:
    # - DataFrame: Combined and cleaned dataset.

    # Example:
    # - directory = r'C:\Users\frifi\OneDrive\Desktop\School Mat UHasselt\Third Semester\Project Data Science\Data sets\01 Sales Forecasts'
    # - bundle = r'C:\Users\frifi\OneDrive\Desktop\School Mat UHasselt\Third Semester\Project Data Science\Data sets\Bundle TE\Bundle item TE.xlsx'
    # - item_overview = r'C:\Users\frifi\OneDrive\Desktop\School Mat UHasselt\Third Semester\Project Data Science\Data sets\Bundle TE\Item overview TE (1).xlsx'

    # - combined_data = clean_and_combine(directory, bundle, item_overview)
    # """
    # Merge all the excel files
    dataframes = []

    for filename in os.listdir(directory):
        if filename.endswith((".xlsx", ".xls")):
            filepath = os.path.join(directory, filename)
            df = pd.read_excel(filepath)
            dataframes.append(df)

    # Validate and enforce consistent column names
    for df in dataframes:
        df.columns = map(str.strip, df.columns)

    # Merge the files
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Select only needed columns from the merged dataset
    columns_to_copy = ["Order_number", "Order_date", "Nb_of_colli",
                       "Nb_of_pallets", "Article_number", "Product_number", "Qty_actual"]
    real_df1 = merged_df[columns_to_copy]

    # Bundle Item data
    df2 = pd.read_excel(bundle_path)

    # Use merge to match and append values from df2 to real_df1
    real_df2 = real_df1.merge(
        df2, left_on='Product_number', right_on='ProductCode', how='left')

    # Drop specified columns
    drop_cols_df2 = ['Column1', 'Column2']  # Specify actual column names
    real_df2 = real_df2.drop(drop_cols_df2, axis=1)

    # Item Overview data
    df3 = pd.read_excel(item_overview_path)

    # Use merge to match and append values from df3 to real_df2
    real_df3 = real_df2.merge(
        df3, left_on='Product_number', right_on='Item No.', how='left')

    # Drop specified columns
    drop_cols_df3 = ['Column3', 'Column4']  # Specify actual column names
    real_df3 = real_df3.drop(drop_cols_df3, axis=1)

    return real_df3


def full_case_calculation(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function calculates the full case and single case of the 
    unique orders matched in the bundle and item overview dataset

    Argument:
    --------
    - df (str): dataframe with 'ChildQty', 'Qty_actual', 'Strategy Code'

    Returns:
    -------
    - DataFrame: with full case and single unit features

    Example:
    -------
    first_result1 = full_case_calculation(combined_data)
    """
    # Input parameter validation
    required_columns = ['ChildQty', 'Qty_actual', 'Strategy Code']
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Calculation for the full case
    df['Full_case'] = np.where(df['ChildQty'].notna() & df['Qty_actual'].notna(),
                               df['Qty_actual'] // df['ChildQty'], 0)

    # Calculation for the single unit
    df['Single_unit'] = np.where(df['ChildQty'].notna() & df['Qty_actual'].notna(),
                                 df['Qty_actual'] % df['ChildQty'], 0)

    # Set 'Single_unit' to 'Qty_actual' where 'Strategy Code' is NaN
    df['Single_unit'] = np.where(df['ChildQty'].isna(),
                                 df['Qty_actual'], df['Single_unit'])  # Create the 'Automated' column based on 'Strategy Code'
    # 1 if Strategy Code is "HB_OSR", "HB_OSR_TMB"
    # 0 otherwise
    df['Automated'] = df['Strategy Code'].isin(
        ["HB_OSR", "HB_OSR_TMB"]).astype(int)

    return df


def divide_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function splits the dataframe by the values in the column. The column
    "Automated" consists of binary values whereby; 0 - Manual and 1 - OSR

    Argument:
    --------
    - Dataframe: this dataframe has 'Automated' as a column

    Returns:
    -------
    - Dataframe: two (manual and OSR) dataframes

    Example:
    -------
    not_osr_data, osr_data = divide_and_clean(first_result1)
    """
    # Input parameter validation
    required_columns = ['Automated']
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Divide the data into Automated (OSR) and Manual (Not_OSR)
    not_osr = df[df["Automated"] == 0]
    osr = df[df["Automated"] == 1]

    return not_osr, osr


def process_osr_data(osr_data: pd.DataFrame) -> pd.DataFrame:
    """
    This function calculates the workload of the products processed in the OSR

    Argument:
    --------
    - Dataframe: this dataframe has 'Nb_of_colli', 'Nb_of_pallets',
    'Single_unit', 'Order_number', 'Order_date' as a column

    Returns:
    -------
    - Dataframe: with order date and time

    Example:
    -------
    new_OSR_data = process_osr_data(osr_data)
    """
    # Input parameter validation
    required_columns = ['Nb_of_colli', 'Nb_of_pallets',
                        'Single_unit', 'Order_number', 'Order_date']
    missing_columns = set(required_columns) - set(osr_data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Drop specified columns
    drop_cols = [4, 5, 6, 7, 8, 9, 10, 12]
    osr_data = osr_data.drop(osr_data.columns[drop_cols], axis=1)

    # Remove rows where both 'Nb_of_colli' and 'B' are 0
    osr_data = osr_data[(osr_data['Nb_of_colli'] != 0) |
                        (osr_data['Nb_of_pallets'] != 0)]

    # Aggregate by Order_number and Order_date
    osr_agg = osr_data.groupby(['Order_number', 'Order_date']).agg(
        Count=('Single_unit', 'count')).reset_index()

    # Drop the Order_number column
    osr_agg = osr_agg.drop(osr_agg.columns[[0]], axis=1)

    # Sum the counts by unique dates
    osr_time = osr_agg.groupby('Order_date')[
        'Count'].sum().reset_index(name='sum_count')
    osr_time['time'] = (osr_time['sum_count'] * 60) / 200 / 60

    return osr_time


def split_into_full_and_single(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function splits the dataframe into two: Full and Single case

    Argument:
    --------
    - Dataframe: this dataframe has 'full case' and 'single unit' as a column

    Returns:
    -------
    - Dataframe: with all the columns

    Example:
    -------
    new_Not_OSR_full, new_Not_OSR_single = split_into_full_and_single(not_osr_data)
    """
    # drop the single unit column
    not_osr_full = df.drop(df.columns[11], axis=1)

    # drop the full case column
    not_osr_single = df.drop(df.columns[10], axis=1)

    # drop unnecessary columns for full case
    drop_cols = [4, 5, 6, 7, 8, 9, 11]
    not_osr_full = not_osr_full.drop(not_osr_full.columns[drop_cols], axis=1)

    # remove rows with 0 pallets and 0 colli
    new_not_osr_full = not_osr_full[(not_osr_full['Nb_of_colli'] != 0) | (
        not_osr_full['Nb_of_pallets'] != 0)]

    # remove empty full cases
    not_osr_full = not_osr_full[not_osr_full['Full_case'] != 0]

    # drop irrelevant columns for single unit
    drop_cols = [4, 5, 6, 7, 8, 9, 11]
    not_osr_single = not_osr_single.drop(
        not_osr_single.columns[drop_cols], axis=1)

    # remove rows where both 'Nb_of_colli' and 'pallets' are 0
    new_not_osr_single = not_osr_single[(not_osr_single['Nb_of_colli'] != 0) | (
        not_osr_single['Nb_of_pallets'] != 0)]

    return new_not_osr_full, new_not_osr_single


def aggregate_counts_and_time(df_full: pd.DataFrame, df_single: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes two dataframe, aggregates and calculate the workload in the two dataframe

    Argument:
    --------
    - Dataframe: this dataframe has 'Order_number', 'Order_date', 'full case' and 'single unit' as a column

    Returns:
    -------
    - Dataframe: two single unit and full case dataframes

    Example:
    -------
    not_osr_time_full, not_osr_time_single = aggregate_counts_and_time(new_Not_OSR_full, new_Not_OSR_single)
    """
    # group by Order number and Order date, and count the number of single unit and full case
    agg_full = df_full.groupby(['Order_number', 'Order_date']).agg(
        Count=('Full_case', 'count')).reset_index()
    agg_single = df_single.groupby(['Order_number', 'Order_date']).agg(
        Count=('Single_unit', 'count')).reset_index()

    # drop the Order numbers
    agg1 = agg_full.drop(agg_full.columns[[0]], axis=1)
    agg2 = agg_single.drop(agg_single.columns[[0]], axis=1)

    # Sum the counts by unique dates
    time_full = agg1.groupby('Order_date')[
        'Count'].sum().reset_index(name='sum_count')
    time_single = agg2.groupby('Order_date')[
        'Count'].sum().reset_index(name='sum_count')

    # Calculate time
    time_single['time'] = (time_single['sum_count'] * 60) / 40 / 60
    time_full['time'] = (time_full['sum_count'] * 60) / 80 / 60

    return time_full, time_single


def merge_times(osr_time: pd.DataFrame, not_osr_time_full: pd.DataFrame, not_osr_time_single: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes three dataframe and merge them together.

    Argument:
    --------
    - Dataframe: OSR, Single unit manual and Full case manual dataframe. It has 'Order_date', 'time' as a columns.

    Returns:
    -------
    - Dataframe: a dataframe

    Example:
    -------
    combined_time_result = merge_times(new_OSR_data, not_osr_time_full, not_osr_time_single)
    """
    # Merge the dataframes based on the 'date' column
    combined_time = pd.merge(osr_time, not_osr_time_full,
                             on='Order_date', how='outer')

    # Fill NAN with 0
    combined_time = combined_time.fillna(0)

    # Sum the time column to get the total
    combined_time['sum_time'] = combined_time['time_x'] + \
        combined_time['time_y']

    # Drop irrelevant columns
    combined_time = combined_time.drop(
        combined_time.columns[[1, 2, 3, 4]], axis=1)

    # Merge with not_osr_time_single
    combined_time = pd.merge(
        combined_time, not_osr_time_single, on='Order_date', how='outer')

    # Fill NAN with 0
    combined_time = combined_time.fillna(0)

    # Sum the time column to get the total
    combined_time['Time'] = combined_time['sum_time'] + combined_time['time']

    # Drop irrelevant columns
    combined_time = combined_time.drop(
        combined_time.columns[[1, 2, 3]], axis=1)

    return combined_time


def extract_pallets_and_colli(combined_data: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes a dataframe and aggregates the number of pallets and colli by Order date.

    Argument:
    --------
    - Dataframe: It has 'Order_number', 'Order_date', 'Nb_of_colli', 'Nb_of_pallets' as a columns.

    Returns:
    -------
    - Dataframe: two (pallets and colli) dataframe

    Example:
    -------
    pallets_df, colli_df = extract_pallets_and_colli(new_data)
    """
    # Group date by Colli
    new_colli = combined_data.groupby(
        ['Order_number', 'Order_date', 'Nb_of_colli']).size().reset_index(name='count')
    unique_order_number_colli = new_colli.drop_duplicates(
        subset=['Order_number'])

    # Group date by Palette
    new_palette = combined_data.groupby(
        ['Order_number', 'Order_date', 'Nb_of_pallets']).size().reset_index(name='count')
    unique_order_number_pal = new_palette.drop_duplicates(
        subset=['Order_number'])

    # Drop unnecessary columns
    new_drop = [0, 3]
    unique_order_number_pal = unique_order_number_pal.drop(
        unique_order_number_pal.columns[new_drop], axis=1)
    unique_order_number_colli = unique_order_number_colli.drop(
        unique_order_number_colli.columns[new_drop], axis=1)

    # Group by the Order date and sum the number of pallets and colli
    unique_order_number_pal = unique_order_number_pal.groupby(
        'Order_date')['Nb_of_pallets'].sum().reset_index()
    unique_order_number_colli = unique_order_number_colli.groupby(
        'Order_date')['Nb_of_colli'].sum().reset_index()

    # Sort the dataframes by the order date
    unique_order_number_pal = unique_order_number_pal.sort_values(
        by='Order_date', ascending=True)
    unique_order_number_colli = unique_order_number_colli.sort_values(
        by='Order_date', ascending=True)

    return unique_order_number_pal, unique_order_number_colli


def calculate_workload(pallets_df, colli_df, combined_time_result):
    """
    This function takes three dataframe to calculate the Total number of pallets and workload.

    Argument:
    --------
    - Dataframe: It has 'Order_date', 'Nb_of_colli', 'Nb_of_pallets' as a columns.

    Returns:
    -------
    - Dataframe: a dataframe with new features like Total number of pallet and Workload(hrs)

    Example:
    -------
    workload_df = calculate_workload(pallets_df, colli_df, combined_time_result)
    """
    # Pallets time calculation
    pallets_time = pallets_df.copy()
    pallets_time['time'] = pallets_time["Nb_of_pallets"] * 0.516667

    # Combine OSR picking time and other activities time for pallets
    outbound_new = pd.merge(combined_time_result,
                            pallets_time, on='Order_date', how='outer')

    # Sum both time together
    outbound_new['time1'] = outbound_new['Time'] + outbound_new['time']
    outbound_new = outbound_new.drop(outbound_new.columns[[1, 3]], axis=1)
    outbound_new = outbound_new.fillna(0)

    # Colli time calculation
    colli_time = colli_df.copy()
    colli_time['time'] = colli_time["Nb_of_colli"] * 0.05

    # Merge pallets calculated time and calculated colli time
    outbound_new1 = pd.merge(outbound_new, colli_time,
                             on='Order_date', how='outer')
    outbound_new1 = outbound_new1.fillna(0)

    # Calculate the total number of workload
    outbound_new1['Workload(hrs)'] = outbound_new1['time'] + \
        outbound_new1['time1']
    outbound_new1 = outbound_new1.drop(outbound_new1.columns[[2, 4]], axis=1)

    # Calculate the total number of pallets including colli
    outbound_new1['Total_number_of_pallet'] = outbound_new1['Nb_of_pallets'] + \
        outbound_new1['Nb_of_colli']*0.02

    return outbound_new1


def add_columns_and_merge(workload_df: pd.DataFrame, combined_data: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes two dataframes and creates more features .

    Argument:
    --------
    - Dataframe: It has 'Order_date', 'Order_number' as a columns.

    Returns:
    -------
    - Dataframe: a dataframe with new features.

    Example:
    -------
    new_dataframe = add_columns_and_merge(workload_df, combined_data)
    """
    # Number of orders each day from orderline
    number_of_orders = combined_data.groupby(['Order_date', 'Order_number'])[
        'Order_number'].count().reset_index(name='Count')
    number_of_orders = number_of_orders.groupby(
        "Order_date")["Count"].sum().reset_index()

    # Merge order frequency with the outbound data
    outbound_new2 = pd.merge(workload_df, number_of_orders, on='Order_date')
    outbound_new2.rename(columns={'Count': 'Number of Orders',
                                  'Nb_of_pallets': 'Nb of pallets',
                                  'Nb_of_colli': 'Nb of colli',
                                  'Total_number_of_pallet': 'Tot Nb of pallet',
                                  'Order_date': 'Order Date'},
                         inplace=True)

    # Calculate the mean of the 'Number of Orders'
    mean_orders = outbound_new2['Number of Orders'].mean()

    # Create a new column 'order_freq' based on conditions
    outbound_new2['Order Freq'] = np.where(outbound_new2['Number of Orders'] > mean_orders, 'high',
                                           np.where((outbound_new2['Number of Orders'] >= (mean_orders)/2) & (outbound_new2['Number of Orders'] <= (mean_orders-200)), 'medium', 'low'))

    # Extract date-related information
    outbound_new2['day'] = outbound_new2['Order Date'].dt.day
    outbound_new2['month'] = outbound_new2['Order Date'].dt.month
    outbound_new2['week'] = outbound_new2['Order Date'].dt.isocalendar().week
    outbound_new2['quarter'] = outbound_new2['Order Date'].dt.quarter

    # Round the columns
    outbound_new2['Workload(hrs)'] = outbound_new2['Workload(hrs)'].round(2)
    outbound_new2['Tot Nb of pallet'] = outbound_new2['Tot Nb of pallet'].round(
        2)

    return outbound_new2
