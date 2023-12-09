# import libraries

import pandas as pd
import numpy as np


def process_receiving_lines(file_path: str) -> pd.DataFrame:
    
    """
    This function takes a csv file, clean and create new features.


    Arguments: A string of the csv file.
    ---------

    Returns: A dataframe with new features like 'Total number of pallets' and 'Workload(hrs)'
    -------

    Example:
    -------
    processed_inbound = process_receiving_lines(r'C:\Users\frifi\OneDrive\Desktop\School Mat UHasselt\Third Semester\data.csv')

    """
    # Read the receiving lines file
    Or_rec_lines = pd.read_csv(file_path)

    # Columns to drop
    columns_to_drop = ['Reference_number', 'Receiving_type_name', 'Batch_number', 'Article_number',
                       'Initiator_code', 'Qty_actual', 'Qty_original', 'Scheduled', 'Product_name_nl',
                       'Warehouse_name', 'Index', 'Receiving_number', 'Product_number', 'Transport_environment_name']

    # Remove unnecessary columns
    inbound = Or_rec_lines.drop(columns=columns_to_drop)

    # Keep a unique date and unique value for Frigo and kamertemperatuur pallets
    inbound.set_index('Delivery_date', inplace=True)
    inbound = inbound.groupby(inbound.index).first()
    inbound.reset_index(inplace=True)
    inbound.columns = ['Date', 'Room_Temp pallets', 'Frigo_Temp pallets']

    # Create a new column for the total number of pallets
    inbound['Total number of pallets'] = inbound['Room_Temp pallets'] + \
        inbound['Frigo_Temp pallets']

    # Calculate the workload column in the inbound
    inbound['Workload(hrs)'] = inbound['Total number of pallets'] * 0.107

    return inbound


def process_inbound_and_planning(receiving_lines_path:str, planning_path:str)-> pd.DataFrame:

    """
    This function takes one csv file and excel file, clean and create new features.

    Arguments: A string of one csv file and excel file.
    ---------

    Returns: A dataframe
    -------

    Example:
    -------

    merged_inbound_planning = process_inbound_and_planning(
    r'C:\Users\frifi\OneDrive\Desktop\School Mat UHasselt\Third Semester\Project Data Science\Receiving_lines.csv',
    r'C:\Users\frifi\OneDrive\Desktop\School Mat UHasselt\Third Semester\Project Data Science\Data sets\inbound22&23_version1.xlsx')

    """
    # Load the receiving lines dataset
    or_rec_lines = pd.read_csv(receiving_lines_path)

    # Removing unnecessary columns
    columns_to_drop = ['Reference_number', 'Receiving_type_name', 'Batch_number', 'Article_number',
                       'Initiator_code', 'Qty_actual', 'Qty_original', 'Scheduled', 'Product_name_nl',
                       'Warehouse_name', 'Index', 'Receiving_number', 'Product_number', 'Transport_environment_name']

    inbound = or_rec_lines.drop(columns=columns_to_drop)

    # Set 'Delivery_date' as the index
    inbound.set_index('Delivery_date', inplace=True)

    # Keep the first occurrence of each day
    inbound = inbound.groupby(inbound.index).first()

    # Reset the index to create a new column for 'Delivery_date'
    inbound.reset_index(inplace=True)

    # Rename the columns
    inbound.columns = ['Date', 'Room_Temp pallets', 'Frigo_Temp pallets']

    # Create a new column for the total number of pallets
    inbound['Total number of pallets'] = inbound['Room_Temp pallets'] + \
        inbound['Frigo_Temp pallets']

    # Adding the workload column with the calculation of the workload in the inbound
    inbound['Workload(hrs)'] = inbound['Total number of pallets'] * 0.107

    # Load the planning dataset for inbound
    plannings = pd.read_excel(planning_path)

    # Convert the 'Date' columns to datetime format in both datasets
    inbound['Date'] = pd.to_datetime(inbound['Date'])
    plannings['Date'] = pd.to_datetime(plannings['Date'])

    # Merges both the planning dataset and inbound datasets based on the 'Order Date' column
    merged_data = pd.merge(
        inbound, plannings[['Date', 'Total_Count']], on='Date', how='left')

    # Renaming the 'Total_Count' column to 'Actual number of workers'
    merged_data = merged_data.rename(
        columns={'Total_Count': 'Actual number of workers'})


    # Filled the missing value with 9, which is the mean of the number of workers planned each day at inbound
    merged_data['Actual number of workers'] = merged_data['Actual number of workers'].fillna(
        9)

    return merged_data


def process_inbound_timeslots(original_dataset_path: str, planning_dataset_path: str,
                               inbound_timeslots_path: str)-> pd.DataFrame:
    
    """
    This function takes a csv file, and two excel files to clean and create new features.

    Arguments: Three files are needed to use this function:
    ---------
        i. the receiving lines dataset in csv format
        ii. the planning dataset in excel format
        iii. the inbound timeslot in excel format

    Returns: A dataframe
    -------

    Example:
    -------
    result_dataset = process_inbound_timeslots(
    r'C:\Users\frifi\OneDrive\Desktop\School Mat UHasselt\Third Semester\Project Data Science\Data sets\Receiving_lines\Receiving_lines_ori 01072022-31072023.xlsx - Export.csv',
    r'C:\Users\frifi\OneDrive\Desktop\School Mat UHasselt\Third Semester\Project Data Science\Data sets\Personel_planning\inbound22&23_version1.xlsx',
    r'C:\Users\frifi\OneDrive\Desktop\School Mat UHasselt\Third Semester\Project Data Science\Data sets\Inbound_timeslots\Inbound tijdssloten.xlsx'
)

    """
    # Load the receiving lines dataset
    or_rec_lines = pd.read_csv(original_dataset_path)

    # Removing unnecessary columns
    columns_to_drop = ['Reference_number', 'Receiving_type_name', 'Batch_number', 'Article_number',
                       'Initiator_code', 'Qty_actual', 'Qty_original', 'Scheduled', 'Product_name_nl',
                       'Warehouse_name', 'Index', 'Receiving_number', 'Product_number', 'Transport_environment_name']

    inbound = or_rec_lines.drop(columns=columns_to_drop)

    # Set 'Delivery_date' as the index
    inbound.set_index('Delivery_date', inplace=True)

    # Keep the first occurrence of each day
    inbound = inbound.groupby(inbound.index).first()

    # Reset the index to create a new column for 'Delivery_date'
    inbound.reset_index(inplace=True)

    # Rename the columns
    inbound.columns = ['Date', 'Room_Temp pallets', 'Frigo_Temp pallets']

    # Create a new column for the total number of pallets
    inbound['Total number of pallets'] = inbound['Room_Temp pallets'] + \
        inbound['Frigo_Temp pallets']

    # Adding the workload column with the calculation of the workload in the inbound
    inbound['Workload(hrs)'] = inbound['Total number of pallets'] * 0.107

    # Load the planning dataset for inbound
    plannings = pd.read_excel(planning_dataset_path)

    # Convert the 'Date' columns to datetime format in both datasets
    inbound['Date'] = pd.to_datetime(inbound['Date'])
    plannings['Date'] = pd.to_datetime(plannings['Date'])

    # Merges both the planning dataset and inbound datasets based on the 'Order Date' column
    merged_data = pd.merge(
        inbound, plannings[['Date', 'Total_Count']], on='Date', how='left')

    # Renaming the 'Total_Count' column to 'Actual number of workers'
    merged_data = merged_data.rename(
        columns={'Total_Count': 'Actual number of workers'})


    # Filled the missing value with 9, which is the mean of the number of workers planned each day at inbound
    merged_data['Actual number of workers'] = merged_data['Actual number of workers'].fillna(
        9)

    # Load the inbound timeslots dataset
    df = pd.read_excel(inbound_timeslots_path)

    # Group by 'Date' and 'Task', then sum the counts for specified tasks for each day
    task_counts = df.groupby(
        ['From Dt', 'Group']).size().reset_index(name='Count')

    filtered_df = task_counts[task_counts['Group'].isin(
        ['Inbound', 'Inbound Frigo'])]

    # Group by 'Date' and sum the counts for specified tasks for each day
    task_counts = filtered_df.groupby(['From Dt', 'Group'])[
        'Count'].sum().reset_index()

    # Pivot the table to have tasks as columns and dates as rows
    pivot_table = task_counts.pivot_table(
        index='From Dt', columns='Group', values='Count', fill_value=0, aggfunc='sum').reset_index()

    # Sum the counts for 'Frigo', 'L&L', 'Onthaal', 'Inbound', 'No' for each day
    pivot_table['Total_Count'] = pivot_table[[
        'Inbound', 'Inbound Frigo']].sum(axis=1)

    # Select only the required columns in the final output
    final_output = pivot_table[['From Dt',
                                'Inbound', 'Inbound Frigo', 'Total_Count']]

    final_output['From Dt'] = pd.to_datetime(final_output['From Dt'])
    final_output = final_output.rename(columns={'Inbound': 'Trucks_in_Inbound',
                                                'Inbound Frigo': 'Trucks_in_Frigo',
                                                'Total_Count': 'Total_trucks',
                                                'From Dt': 'Date'})

    # Merging the information on the number of trucks with the original dataset
    df_merge = pd.merge(merged_data, final_output, on='Date', how='left')

    # Calculate the mean of the 'Total_trucks' column
    mean_total_trucks = df_merge['Total_trucks'].mean()

    # Create a new column 'Truck Frequency' based on the mean
    df_merge['Truck Frequency'] = np.where(df_merge['Total_trucks'] > mean_total_trucks, 'high',
                                           np.where((df_merge['Total_trucks'] >= 5) & (df_merge['Total_trucks'] <= 10), 'medium', 'low'))

    return df_merge
