import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import warnings

# machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# import datetime
warnings.filterwarnings('ignore')


# Set page configuration
st.set_page_config(
    page_title='Livlina Logistics Company',
    page_icon="🚚",
    layout='wide',
    initial_sidebar_state='expanded'
)

header_left, header_mid, header_right = st.columns([1, 6, 1])

# Header in the middle with logo
with header_mid:
    st.markdown(
        f'<style> .block-container{{padding-top: 1rem;}}</style>', unsafe_allow_html=True)

    image = Image.open("C:/Users/harve/Downloads/livlina_logistics_cover.jpeg")

    st.image(image, use_column_width=False)

    st.markdown('</p>', unsafe_allow_html=True)


# Load dataset
insert = st.file_uploader(":file_folder: ",
                          type=(["csv", "txt", "xlsx", "xls"]))
if insert is not None:
    filename = insert.name
    st.write(filename)
    data = pd.read_csv(filename)
else:
    # Inbound data
    inbound_df = pd.read_csv('Inbound timeslot.csv', low_memory=False)
    inbound_df['Order_date'] = pd.to_datetime(inbound_df['From Dt']).dt.date
    drop_col2 = [0, 2, 4, 5, 6, 7, 9, 12, 13, 14, 15,
                 16, 17, 18, 19, 23, 24, 25, 26, 28, 30]

    inbound_data = inbound_df.drop(inbound_df.columns[drop_col2], axis=1)
    # Rename dataframe
    inbound_data.rename(columns={
        'Group': 'Zone',
        'Order_date': 'Order Date',
        'temperature conditions': 'Temperature',
        'pharmaceutical company / customer': 'Customer',
        'transport company': 'Transport Company'
    }, inplace=True)

    # Outbound data
    orderline = pd.read_csv("All_Orderline.csv")

    col = [0, 2, 11, 12, 13, 14, 15, 18, 21, 23]
    outbound_data = orderline.iloc[:, col]

    outbound_data.rename(columns={
        'Order_number': 'Order Number',
        'Order_date': 'Order Date',
        'Nb_of_colli': 'Number of Colli',
        'Nb_of_pallets': 'Number of Pallets',
        'Consignee_type_name_en': 'Consignee Type',
        'Consignee_delivery_country_code': 'Consignee Delivery Country Code',
        'Consignee_name': 'Consignee Name',
        'Shipping_way_name': 'Shipping Way',
        'Product_name_nl': 'Product Name (NL)',
        'Qty_actual': 'Actual Quantity'
    }, inplace=True)

    outbound_data['Order Date'] = pd.to_datetime(
        outbound_data['Order Date']).dt.strftime('%Y-%m-%d')

# Set default values
view = "Overview"
zone = "Outbound"

# Home button
if st.sidebar.button("Home"):
    # Redirect to the main page or set the view variable accordingly
    view = "Overview"
    zone = "Outbound"
    st.write('<p style="font-weight:bold; font-size:24px; text-align:center;">Welcome, please select a zone you wish to explore and predict.</p>', unsafe_allow_html=True)


# Sidebar for zone selection
zone = st.sidebar.selectbox("Zone", ["Inbound", "Outbound"], index=1)

# Navigation menu for different views
view = st.sidebar.selectbox(
    "View", ["Overview", "Exploratory", "Prediction"], index=0)


# Function to load data
def load_data(zone):
    if zone == "Outbound":
        return outbound_data  # Outbound data
    else:
        return inbound_data  # Inbound_data


# Load data based on zone selection
selected_data = load_data(zone)

col1, col2 = st.columns((2))

# Assuming you have a selected_data variable
if selected_data is not None:

    # Check if "Order_date" column exists in selected_data
    if 'Order Date' in selected_data.columns:
        selected_data['Order Date'] = pd.to_datetime(
            selected_data['Order Date'])
        start_date = selected_data['Order Date'].min()
        end_date = selected_data['Order Date'].max()

        with col1:
            date1 = pd.to_datetime(st.date_input("Start Date", start_date))

        with col2:
            date2 = pd.to_datetime(st.date_input("End Date", end_date))

        selected_data = selected_data[
            (selected_data['Order Date'] >= date1) & (selected_data['Order Date'] <= date2)].copy()
    else:
        st.warning("Selected data does not have the 'Order_date' column.")
else:
    st.warning("No data selected.")


# Create columns for header
header_left, header_mid, header_right = st.columns([1, 6, 1])


# Overview View
if (view == "Overview" and zone == "Outbound"):
    with st.container():

        if selected_data is not None:
            st.write(f"### Overview of the {zone} Zone")

            # First Row: Metric Cards
            metric_columns = st.columns(4)

            # Metric 1: Average Number of Pallets
            with metric_columns[0]:
                sum_colli_pall = selected_data.groupby(["Order Date"]).agg(
                    {"Number of Pallets": "sum", "Number of Colli": "sum", "Actual Quantity": "sum"}).reset_index()
                avg_pall = round(sum_colli_pall["Number of Pallets"].mean())

                st.markdown(
                    f'<div style="background-color: #FFD700; padding: 20px; border-radius: 10px; text-align: center;">'
                    f'<h3 style="color: #000080;">Avg Pallets</h3>'
                    f'<p style="font-size: 24px;">{avg_pall}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Metric 2: Average Number of Colli
            with metric_columns[1]:

                avg_colli = round(sum_colli_pall["Number of Colli"].mean())

                st.markdown(
                    f'<div style="background-color: #800080; padding: 20px; border-radius: 10px; text-align: center;">'
                    f'<h3 style="color: #FFFFFF;">Avg Colli</h3>'
                    f'<p style="font-size: 24px; color: #FFFFFF;">{avg_colli}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Metric 3: Average Time(hr)
            with metric_columns[2]:

                avg_quantity = round(sum_colli_pall["Actual Quantity"].mean())

                st.markdown(
                    f'<div style="background-color: #00FA9A; padding: 20px; border-radius: 10px; text-align: center;">'
                    f'<h3 style="color: #000080;">Avg Quantity</h3>'
                    f'<p style="font-size: 24px;">{avg_quantity}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Metric 4: Average Number of Workers
            with metric_columns[3]:

                number_of_order = selected_data.groupby(
                    ["Order Date"])["Order Number"].count().reset_index()

                number_of_order.columns = ["Order Date", "Order Count"]

                avg_order = round(
                    number_of_order["Order Count"].mean())

                st.markdown(
                    f'<div style="background-color: #FFD700; padding: 20px; border-radius: 10px; text-align: center;">'
                    f'<h3 style="color: #2f4f4f;">Avg Orders</h3>'
                    f'<p style="font-size: 24px; color: #000080;">{avg_order}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Add spacing
            st.markdown("<br><br>", unsafe_allow_html=True)

            metric_columns = st.columns(2)

            # Plot of Workload by Order_dates
            with metric_columns[0]:
                st.write("### Distribution of Orders over Time")

                selected_data_sorted = selected_data.sort_values(
                    by='Order Date')

                number_of_order = selected_data_sorted.groupby(
                    ["Order Date"])["Order Number"].count().reset_index()

                number_of_order.columns = ["Order Date", "Order Count"]

                fig = px.line(number_of_order, x='Order Date', y='Order Count',
                              title='')

                st.plotly_chart(fig)

            # Plot of Order frequency
            with metric_columns[1]:
                st.write("### Order Frequency")

                selected_data_sorted = selected_data.sort_values(
                    by='Order Date')

                number_of_order = selected_data_sorted.groupby(
                    ["Order Date"])["Order Number"].count().reset_index()

                number_of_order.columns = ["Order Date", "Order Count"]

                number_of_order["Order Frequency"] = np.where(number_of_order["Order Count"] > 2300, 'high',
                                                              np.where((number_of_order["Order Count"] >= 1150) & (number_of_order["Order Count"] <= 2299), 'medium', 'low'))

                order_frequency_bar = px.bar(
                    number_of_order, x='Order Frequency', title='', labels={'Order Count': 'Order Count'})

                st.plotly_chart(order_frequency_bar)

            metric_columns = st.columns(2)

            with metric_columns[0]:
                st.write("### Distribution of Consignee Type")

                consignee = selected_data.groupby(["Consignee Type"])[
                    "Order Number"].count().reset_index()

                fig = px.pie(consignee, values="Order Number",
                             names="Consignee Type", hole=0.5)

                fig.update_traces(
                    text=consignee["Consignee Type"], textposition="outside")

                st.plotly_chart(fig, use_container_width=True)

            with metric_columns[1]:
                st.subheader("Distribution of Products")

                product_name = selected_data.groupby(["Product Name (NL)"])[
                    "Order Number"].count().reset_index()

                fig01 = px.treemap(product_name, path=["Product Name (NL)"], values="Order Number", hover_data=["Order Number"],
                                   color="Product Name (NL)")

                fig01.update_layout(width=600, height=450)

                st.plotly_chart(fig01, use_container_width=True)

            metric_columns = st.columns(2)

            with metric_columns[1]:
                st.subheader("Location Distribution")
                country_data = selected_data.groupby(["Consignee Delivery Country Code"])[
                    "Order Number"].count().reset_index()

                # Calculate percentage
                country_data['Order Percentage'] = round(country_data['Order Number'] /
                                                         country_data['Order Number'].sum() * 100)

                # Create a pie chart
                pie_chart = px.pie(country_data, names='Consignee Delivery Country Code', values='Order Number',
                                   title='',
                                   template="plotly_dark")

                pie_chart.update_traces(textposition="inside")

                # Display the pie chart
                st.plotly_chart(pie_chart, use_container_width=True)

            with metric_columns[0]:
                st.subheader("Shipping Way distribution")
                shipping_way = selected_data.groupby(
                    ["Shipping Way"])["Order Number"].count().reset_index()

                # Calculate the percentage
                shipping_way['Percentage'] = round(shipping_way['Order Number'] /
                                                   shipping_way['Order Number'].sum() * 100)

                # Create a horizontal bar plot with hover labels
                fig02 = px.bar(shipping_way, x='Percentage', y='Shipping Way', orientation='h',
                               text='Percentage', title='Order Distribution by Shipping Way',
                               labels={'Percentage': 'Order Percentage'})

                # Customize layout
                fig02.update_layout(title_text='',
                                    xaxis_title='Order Percentage',
                                    yaxis_title='Shipping Way',
                                    template='plotly_white')

                st.plotly_chart(fig02)


elif (view == "Overview" and zone == "Inbound"):
    if selected_data is not None:
        st.write(f"### Overview of the {zone} Zone")

        # First Row: Metric Cards
        metric_columns = st.columns(3)

        # Metric 1: Average Number of Pallets Inbound
        with metric_columns[0]:
            zone1 = round(selected_data.groupby('Zone')[
                          "Number Of Pallets"].mean().reset_index())

            # Print the value for the 'Outbound' zone
            avg_inbound = zone1.loc[zone1['Zone'] ==
                                    'Inbound', 'Number Of Pallets'].values[0].astype(int)

            st.markdown(
                f'<div style="background-color: #FFD700; padding: 20px; border-radius: 10px; text-align: center;">'
                f'<h3 style="color: #000080;">Avg Pallets Inbound</h3>'
                f'<p style="font-size: 24px;">{avg_inbound}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Metric 2: Average Number of Colli
        with metric_columns[1]:

            zone1 = round(selected_data.groupby('Zone')[
                          "Number Of Pallets"].mean().reset_index())

            avg_inbound_frigo = zone1.loc[zone1['Zone'] ==
                                          'Inbound Frigo', 'Number Of Pallets'].values[0].astype(int)

            st.markdown(
                f'<div style="background-color: #800080; padding: 20px; border-radius: 10px; text-align: center;">'
                f'<h3 style="color: #FFFFFF;">Avg Pallets InboundFrigo</h3>'
                f'<p style="font-size: 24px; color: #FFFFFF;">{avg_inbound_frigo}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Metric 3: Average Time(hr)
        with metric_columns[2]:
            zone1 = round(selected_data.groupby('Zone')[
                          "Number Of Pallets"].mean().reset_index())

            # Print the value for the 'Outbound' zone
            avg_outbound = zone1.loc[zone1['Zone'] ==
                                     'Outbound', 'Number Of Pallets'].values[0].astype(int)

            st.markdown(
                f'<div style="background-color: #00FA9A; padding: 20px; border-radius: 10px; text-align: center;">'
                f'<h3 style="color: #000080;">Avg Pallets Outbound</h3>'
                f'<p style="font-size: 24px;">{avg_outbound}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Second row

        # Add spacing
        st.markdown("<br><br>", unsafe_allow_html=True)

        metric_columns = st.columns(2)

        # Plot of Workload by Order_dates
        with metric_columns[0]:
            st.write("### Distribution of Orders over Time")

            selected_data_sorted = selected_data.sort_values(
                by='Order Date')
            number_of_order = selected_data_sorted.groupby(
                ["Order Date"])["Number Of Pallets"].count().reset_index()
            number_of_order.columns = ["Order Date", "Order Count"]

            fig03 = px.line(number_of_order, x='Order Date', y='Order Count',
                            title='')
            st.plotly_chart(fig03)

        # Plot of Order frequency
        with metric_columns[1]:
            st.write("### Order Frequency")

            selected_data_sorted = selected_data.sort_values(
                by='Order Date')
            number_of_order1 = selected_data_sorted.groupby(
                ["Order Date"])["Number Of Pallets"].count().reset_index()
            number_of_order1.columns = ["Order Date", "Order Count"]

            number_of_order1["Order Frequency"] = np.where(number_of_order1["Order Count"] > 13, 'high',
                                                           np.where((number_of_order1["Order Count"] >= 6.5) & (number_of_order1["Order Count"] <= 12.9), 'medium', 'low'))

            order_frequency_bar1 = px.bar(
                number_of_order1, x='Order Frequency', title='', labels={'Order Count': 'Order Count'})

            st.plotly_chart(order_frequency_bar1)

        # Third row

        metric_columns = st.columns(2)

        with metric_columns[0]:
            st.subheader("Daily Distribution")
            inbound_day = selected_data.groupby(
                ["Day"])["Order Date"].count().reset_index()

            days_order = ['Monday', 'Tuesday',
                          'Wednesday', 'Thursday', 'Friday']

            inbound_day['Day'] = pd.Categorical(
                inbound_day['Day'], categories=days_order, ordered=True)

            inbound_day.columns = ["Day", "Order Count"]

            # Sort the dataframe based on the custom order
            inbound_day = inbound_day.sort_values(by='Day')

            # Create a horizontal bar plot with hover labels
            fig04 = px.bar(inbound_day, x='Order Count', y='Day', orientation='h',
                           text='Order Count', title='')

            # Customize layout
            fig04.update_layout(title_text='',
                                xaxis_title='Frequency',
                                yaxis_title='Days',
                                template='plotly_white')

            st.plotly_chart(fig04)

        with metric_columns[1]:
            st.write("### Distribution of Partners")
            partner = selected_data.groupby(["Partner"])[
                "Order Date"].count().reset_index()
            partner.columns = ["Partner", "Order Count"]
            fig05 = px.pie(partner, values="Order Count",
                           names="Partner", hole=0.5)
            fig05.update_traces(
                text=partner["Partner"], textposition="outside")
            st.plotly_chart(fig05, use_container_width=True)

        # Fourth row

        metric_columns = st.columns(2)

        st.subheader("Booked Frequency")
        booked_name = selected_data.groupby(
            'Booked By')["Order Date"].count().reset_index()

        booked_name.columns = ["Booked By", "Order Count"]

        fig06 = px.treemap(booked_name, path=["Booked By"], values="Order Count", hover_data=["Order Count"],
                           color="Booked By")
        fig06.update_layout(width=600, height=450)
        st.plotly_chart(fig06, use_container_width=True)

# Exploratory View

elif (view == "Exploratory" and zone == "Outbound"):

    with st.container():
        st.write(f"### Exploratory view of the {zone} Zone")

        metric_columns = st.columns(2)

        outbound_data = pd.read_csv("outbound_data.csv")

        outbound_data.rename(columns={
            "Order_date": "Order Date",
            "T_Nb_of_pallets": "Nb of pallets",
            "Nb_of_colli": "Nb of colli",
            "Time(hrs)": "Workload(hrs)",
            "Total_Nb_Pallet": "Total Nb Pallet",
            "Number_of_orders": "Nb of orders",
            "Order_frequency": "Order Frequency"
        }, inplace=True)

        outbound_data["Number of workers"] = round(
            outbound_data['Workload(hrs)']/7.33333)
        outbound_data['Order Date'] = pd.to_datetime(
            outbound_data['Order Date'], format='%d-%m-%Y')

        selected_data1 = outbound_data.copy()

        ####################### NEW ##########################

        # Filter data based on date range
        selected_data1 = selected_data1[
            (selected_data1['Order Date'] >= date1) & (selected_data1['Order Date'] <= date2)].copy()

        ####################### NEW ##########################

        if selected_data1 is not None:

            # First Row: Metric Cards
            metric_columns = st.columns(4)

            # Metric 1: Average Number of Pallets
            with metric_columns[0]:
                avg_pallets = round(selected_data1['Total Nb Pallet'].mean())

                st.markdown(
                    f'<div style="background-color: #FFD700; padding: 20px; border-radius: 10px; text-align: center;">'
                    f'<h3 style="color: #000080;">Avg Pallets</h3>'
                    f'<p style="font-size: 24px;">{avg_pallets}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Metric 2: Average Number of Colli
            with metric_columns[1]:
                avg_colli = round(selected_data1['Nb of colli'].mean())

                st.markdown(
                    f'<div style="background-color: #800080; padding: 20px; border-radius: 10px; text-align: center;">'
                    f'<h3 style="color: #FFFFFF;">Avg Colli</h3>'
                    f'<p style="font-size: 24px; color: #FFFFFF;">{avg_colli}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Metric 3: Average Time(hr)
            with metric_columns[2]:
                avg_time = round(selected_data1['Workload(hrs)'].mean(), 2)

                st.markdown(
                    f'<div style="background-color: #00FA9A; padding: 20px; border-radius: 10px; text-align: center;">'
                    f'<h3 style="color: #000080;">Workload(hr)</h3>'
                    f'<p style="font-size: 24px;">{avg_time:.2f}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Metric 4: Average Number of Workers
            with metric_columns[3]:
                avg_wrk = round(selected_data1["Number of workers"].mean())

                st.markdown(
                    f'<div style="background-color: #FFD700; padding: 20px; border-radius: 10px; text-align: center;">'
                    f'<h3 style="color: #2f4f4f;">Avg Manpower</h3>'
                    f'<p style="font-size: 24px; color: #000080;">{avg_wrk}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Add spacing
            st.markdown("<br><br>", unsafe_allow_html=True)

            metric_columns = st.columns(2)

            # Plot of Workload by Order_dates
            with metric_columns[0]:
                st.write("### Distribution of Workload over Time")
                selected_data_sorted1 = selected_data1.sort_values(
                    by='Order Date')
                fig = px.line(selected_data_sorted1, x='Order Date', y='Workload(hrs)',
                              title='')
                st.plotly_chart(fig)

            # Plot of Order frequency
            with metric_columns[1]:
                st.write("### Distribution of Order Frequency")

                order_frequency_bar = px.bar(
                    selected_data1, x='Order Frequency', title='')
                st.plotly_chart(order_frequency_bar)

            metric_columns = st.columns(2)

            with st.expander("View Data of TimeSeries:"):
                st.write(
                    selected_data_sorted1.T.style.background_gradient(cmap="Blues"))
                csv = selected_data_sorted1.to_csv(index=False).encode("utf-8")
                st.download_button("Download Data", data=csv,
                                   file_name="Timeseries.csv", mime='text/csv')

            metric_columns = st.columns(2)

            with metric_columns[0]:
                st.subheader("Distribution of Colli")
                selected_data_sorted1 = selected_data1.sort_values(
                    by='Order Date')
                fig = px.scatter(selected_data_sorted1, x='Nb of colli', y='Workload(hrs)',
                                 title='')
                st.plotly_chart(fig)

            with metric_columns[1]:

                st.subheader("Distribution of Pallets")
                selected_data_sorted1 = selected_data1.sort_values(
                    by='Order Date')
                fig = px.scatter(selected_data_sorted1, x='Total Nb Pallet', y='Workload(hrs)',
                                 title='')
                st.plotly_chart(fig)

            metric_columns = st.columns(2)

            with metric_columns[0]:
                st.write("### Order Frequency Distribution")
                fig = px.pie(selected_data_sorted1, values="Nb of orders",
                             names="Order Frequency", hole=0.5)
                fig.update_traces(
                    text=selected_data_sorted1["Order Frequency"], textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

            with metric_columns[1]:
                st.write("### Subset of the Data")
                st.write(selected_data_sorted1.style.background_gradient(
                    cmap="rainbow"))
                csv = selected_data_sorted1.to_csv(index=False).encode('utf-8')
                st.download_button("Download Data", data=csv, file_name="Overview.csv", mime="text/csv",
                                   help="Click here to download the data as a CSV file")


# Exploratory View
elif (view == "Exploratory" and zone == "Inbound"):

    with st.container():
        st.write(f"### Exploratory view of the {zone} Zone")

        metric_columns = st.columns(2)

        data_livlina = pd.read_csv(
            'C:\\Users\\harve\\Downloads\\Data_Livlina.csv', low_memory=False)
        inbound_data = data_livlina.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

        # Reorder columns
        inbound_data = inbound_data[["Date", "Pallets(Frigo Temp)", "Pallets(Room Temp)", "Total number of pallets",
                                    "Workload(hrs)", "Trucks(Inbound)", "Trucks(Frigo)",
                                     "Total number of trucks",	"Truck Frequency", "Actual number of workers",]]
        # create number of workers
        inbound_data['Number of workers'] = round(
            inbound_data['Workload(hrs)']/7.3333)

        # Extract from the date

        inbound_data['Date'] = pd.to_datetime(inbound_data['Date'])

        # Extract components from the date
        inbound_data['Day'] = inbound_data['Date'].dt.day
        inbound_data['Month'] = inbound_data['Date'].dt.month
        inbound_data['Week'] = inbound_data['Date'].dt.isocalendar().week

        # Change the date format
        inbound_data['Date'] = pd.to_datetime(
            inbound_data['Date'], format='%m/%d/%Y')

        inbound_data['Date'] = inbound_data['Date'].dt.strftime('%Y/%m/%d')

        selected_data = inbound_data.copy()

        selected_data['Date'] = pd.to_datetime(
            selected_data['Date'], format='%Y-%m-%d')

        selected_data = selected_data[
            (selected_data['Date'] >= date1) & (selected_data['Date'] <= date2)].copy()

        if selected_data is not None:

            # First Row: Metric Cards
            metric_columns = st.columns(4)

            # Metric 1: Average Number of Pallets
            with metric_columns[0]:
                avg_pallets = round(
                    selected_data['Pallets(Frigo Temp)'].mean())

                st.markdown(
                    f'<div style="background-color: #FFD700; padding: 20px; border-radius: 10px; text-align: center;">'
                    f'<h3 style="color: #000080;">Avg Pallets(Frigo)</h3>'
                    f'<p style="font-size: 24px;">{avg_pallets}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Metric 2: Average Number of Colli
            with metric_columns[1]:
                avg_colli = round(
                    selected_data['Pallets(Room Temp)'].mean())

                st.markdown(
                    f'<div style="background-color: #800080; padding: 20px; border-radius: 10px; text-align: center;">'
                    f'<h3 style="color: #FFFFFF;">Avg Pallets(Room Temp)</h3>'
                    f'<p style="font-size: 24px; color: #FFFFFF;">{avg_colli}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Metric 3: Average Time(hr)
            with metric_columns[2]:
                avg_time = round(selected_data['Workload(hrs)'].mean(), 2)

                st.markdown(
                    f'<div style="background-color: #00FA9A; padding: 20px; border-radius: 10px; text-align: center;">'
                    f'<h3 style="color: #000080;">Workload(hr)</h3>'
                    f'<p style="font-size: 24px;">{avg_time:.2f}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Metric 4: Average Number of Workers
            with metric_columns[3]:
                avg_wrker = round(
                    selected_data["Number of workers"].mean())

                st.markdown(
                    f'<div style="background-color: #FFD700; padding: 20px; border-radius: 10px; text-align: center;">'
                    f'<h3 style="color: #2f4f4f;">Avg Manpower</h3>'
                    f'<p style="font-size: 24px; color: #000080;">{avg_wrker}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown("<br><br>", unsafe_allow_html=True)

            metric_columns = st.columns(2)

            # Plot of Workload by Order_dates
            with metric_columns[0]:
                st.write("### Distribution of Workload over Time")
                selected_data_sorted1 = selected_data.sort_values(
                    by='Date')
                fig = px.line(selected_data_sorted1, x='Date', y='Workload(hrs)',
                              title='')
                st.plotly_chart(fig)

            # Plot of Order frequency
            with metric_columns[1]:
                st.write("### Distribution of Truck Frequency")

                order_frequency_bar = px.bar(
                    selected_data, x='Truck Frequency', title='')
                st.plotly_chart(order_frequency_bar)

            metric_columns = st.columns(2)

            with st.expander("View Data of TimeSeries:"):
                st.write(
                    selected_data_sorted1.T.style.background_gradient(cmap="Blues"))
                csv = selected_data_sorted1.to_csv(index=False).encode("utf-8")
                st.download_button("Download Data", data=csv,
                                   file_name="Timeseries.csv", mime='text/csv')

            metric_columns = st.columns(2)

            with metric_columns[0]:
                st.subheader("Distribution of Pallets (Frigo)")
                selected_data_sorted1 = selected_data.sort_values(
                    by='Date')
                fig = px.scatter(selected_data_sorted1, x='Pallets(Frigo Temp)', y='Workload(hrs)',
                                 title='')
                st.plotly_chart(fig)

            with metric_columns[1]:

                st.subheader("Distribution of Pallets (Room Temp)")
                selected_data_sorted1 = selected_data.sort_values(
                    by='Date')
                fig = px.scatter(selected_data_sorted1, x='Pallets(Room Temp)', y='Workload(hrs)',
                                 title='')
                st.plotly_chart(fig)

            metric_columns = st.columns(2)

            with metric_columns[0]:
                st.write("### Truck Frequency Distribution")
                fig = px.pie(selected_data_sorted1, values="Total number of trucks",
                             names="Truck Frequency", hole=0.5)
                fig.update_traces(
                    text=selected_data_sorted1["Truck Frequency"], textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

            with metric_columns[1]:

                st.write("### Subset of the Data")

                selected_data_sorted1 = selected_data_sorted1.dropna()

                st.write(selected_data_sorted1.style.background_gradient(
                    cmap="viridis"))
                csv = selected_data_sorted1.to_csv(index=False).encode('utf-8')
                st.download_button("Download Data", data=csv, file_name="Overview.csv", mime="text/csv",
                                   help="Click here to download the data as a CSV file")


# Prediction View

elif (view == "Prediction" and zone == "Outbound"):

    # with st.container():
    st.write(f"### Prediction view of the {zone} Zone")

    metric_columns = st.columns(2)

    outbound_data = pd.read_csv("outbound_data.csv")

    outbound_data.rename(columns={
        "Order_date": "Order Date",
        "T_Nb_of_pallets": "Nb of pallets",
        "Nb_of_colli": "Nb of colli",
        "Time(hrs)": "Workload(hrs)",
        "Total_Nb_Pallet": "Total Nb Pallet",
        "Number_of_orders": "Nb of orders",
        "Order_frequency": "Order Frequency"
    }, inplace=True)

    outbound_data["Number of workers"] = round(
        outbound_data['Workload(hrs)']/7.33333)
    outbound_data['Order Date'] = pd.to_datetime(
        outbound_data['Order Date'], format='%d-%m-%Y')

    # Prediction steps

    sample = outbound_data.copy()

    # drop Order Date, Order Frequency and Workload
    X = sample.drop(["Order Date", "Order Frequency", "Workload(hrs)"], axis=1)

    # extract emission as the response
    y = sample["Workload(hrs)"]

    # Split 1
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42)

    # Fit Random Forest model

    reg = RandomForestRegressor(
        n_estimators=125, random_state=42)
    reg.fit(x_train, y_train)

    # Make predictions
    x_test['prediction'] = reg.predict(x_test)

    # Merge the predictions into the original dataframe
    feature_pred = sample.merge(
        x_test[['prediction']], how='left', left_index=True, right_index=True)

    feature_pred = feature_pred.fillna(0)

    selected_data1 = feature_pred.copy()

    ####################### NEW ##########################

    # Filter data based on date range
    selected_data11 = selected_data1[
        (selected_data1['Order Date'] >= date1) & (selected_data1['Order Date'] <= date2)].copy()

    ####################### NEW ##########################

    if selected_data11 is not None:

        # First Row: Metric Cards
        metric_columns = st.columns(4)

        # Metric 1: Average Number of Pallets
        with metric_columns[0]:
            avg_pallets = round(selected_data11['Total Nb Pallet'].mean())

            st.markdown(
                f'<div style="background-color: #ff00ff; padding: 20px; border-radius: 10px; text-align: center;">'
                f'<h3 style="color: #000080;">Avg Pallets</h3>'
                f'<p style="font-size: 24px;">{avg_pallets}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Metric 2: Average Number of Colli
        with metric_columns[1]:
            avg_colli = round(selected_data11['Nb of colli'].mean())

            st.markdown(
                f'<div style="background-color: #4b0082; padding: 20px; border-radius: 10px; text-align: center;">'
                f'<h3 style="color: #ff00ff;">Avg Colli</h3>'
                f'<p style="font-size: 24px; color: #FFFFFF;">{avg_colli}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Metric 3: Average Time(hr)
        with metric_columns[2]:
            avg_time = round(selected_data11['Workload(hrs)'].mean(), 2)

            st.markdown(
                f'<div style="background-color: #b22222; padding: 20px; border-radius: 10px; text-align: center;">'
                f'<h3 style="color: #000080;">Workload(hr)</h3>'
                f'<p style="font-size: 24px;">{avg_time:.2f}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Metric 4: Average Number of Workers
        with metric_columns[3]:
            avg_wrk = round(selected_data11["Number of workers"].mean())

            st.markdown(
                f'<div style="background-color: #FFFF00; padding: 20px; border-radius: 10px; text-align: center;">'
                f'<h3 style="color: #000080;">Avg Manpower</h3>'
                f'<p style="font-size: 24px; color: #000080;">{avg_wrk}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Add spacing
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Second row

        # Prediction plot

        # Plot of Workload by Order_dates

        st.write("### Workload Prediction over Time")

        selected_data_sorted1 = selected_data11.sort_values(
            by='Order Date')

        fig = px.line(selected_data_sorted1, x='Order Date', y=['Workload(hrs)', 'prediction'], title='',
                      labels={'value': 'Workload (Hours)'}, color_discrete_sequence=['blue', 'orange'])

        fig.update_traces(line=dict(width=2))

        fig.update_layout(width=1350, height=600)

        st.plotly_chart(fig)

        # Add spacing
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Third row

        with st.expander("View Data of TimeSeries:"):
            st.write(
                selected_data_sorted1.T.style.background_gradient(cmap="Blues"))
            csv = selected_data_sorted1.to_csv(index=False).encode("utf-8")
            st.download_button("Download Data", data=csv,
                               file_name="Timeseries.csv", mime='text/csv')

else:

    # with st.container():
    st.write(f"### Prediction view of the {zone} Zone")

    data_livlina = pd.read_csv(
        'C:\\Users\\harve\\Downloads\\Data_Livlina.csv', low_memory=False)
    inbound_data = data_livlina.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

    # Reorder columns
    inbound_data = inbound_data[["Date", "Pallets(Frigo Temp)", "Pallets(Room Temp)", "Total number of pallets",
                                "Workload(hrs)", "Trucks(Inbound)", "Trucks(Frigo)",
                                 "Total number of trucks",	"Truck Frequency", "Actual number of workers",]]
    # create number of workers
    inbound_data['Number of workers'] = round(
        inbound_data['Workload(hrs)']/7.3333)

    # Extract from the date

    inbound_data['Date'] = pd.to_datetime(inbound_data['Date'])

    # Extract components from the date
    inbound_data['Day'] = inbound_data['Date'].dt.day
    inbound_data['Month'] = inbound_data['Date'].dt.month
    inbound_data['Week'] = inbound_data['Date'].dt.isocalendar().week

    # Change the date format
    inbound_data['Date'] = pd.to_datetime(
        inbound_data['Date'], format='%m/%d/%Y')

    inbound_data['Date'] = inbound_data['Date'].dt.strftime('%Y/%m/%d')

    selected_data = inbound_data.copy()

    selected_data['Date'] = pd.to_datetime(
        selected_data['Date'], format='%Y-%m-%d')

    # Filter data based on date range
    selected_data = selected_data[
        (selected_data['Date'] >= date1) & (selected_data['Date'] <= date2)].copy()

    # Prediction steps

    sample = selected_data.copy()

    # drop Order Date, Order Frequency and Workload
    X = sample.drop(["Date", "Truck Frequency", "Workload(hrs)"], axis=1)

    # extract emission as the response
    y = sample["Workload(hrs)"]

    # Split 1
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42)

    # Fit Random Forest model

    reg = RandomForestRegressor(
        n_estimators=125, random_state=42)
    reg.fit(x_train, y_train)

    # Make predictions
    x_test['prediction'] = reg.predict(x_test)

    # Merge the predictions into the original dataframe
    feature_pred = sample.merge(
        x_test[['prediction']], how='left', left_index=True, right_index=True)

    feature_pred = feature_pred.fillna(0)

    selected_data1 = feature_pred.copy()

    ####################### NEW ##########################

    # Filter data based on date range
    selected_data11 = selected_data1[
        (selected_data1['Date'] >= date1) & (selected_data1['Date'] <= date2)].copy()

    ####################### NEW ##########################

    if selected_data11 is not None:

        # First Row: Metric Cards
        metric_columns = st.columns(4)

        # Metric 1: Average Number of Pallets
        with metric_columns[0]:
            avg_pallets = round(
                selected_data11['Total number of pallets'].mean())

            st.markdown(
                f'<div style="background-color: #87cefa; padding: 20px; border-radius: 10px; text-align: center;">'
                f'<h3 style="color: #000080;">Avg Pallets</h3>'
                f'<p style="font-size: 24px;">{avg_pallets}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Metric 2: Average Number of Trucks
        with metric_columns[1]:
            avg_truck = round(selected_data11['Total number of trucks'].mean())

            st.markdown(
                f'<div style="background-color: #FFB6C1; padding: 20px; border-radius: 10px; text-align: center;">'
                f'<h3 style="color: #000080;">Avg Trucks</h3>'
                f'<p style="font-size: 24px; color: #FFFFFF;">{avg_truck}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Metric 3: Average Time(hr)
        with metric_columns[2]:
            avg_time = round(selected_data11['Workload(hrs)'].mean(), 2)

            st.markdown(
                f'<div style="background-color: #6495ed; padding: 20px; border-radius: 10px; text-align: center;">'
                f'<h3 style="color: #000080;">Workload(hr)</h3>'
                f'<p style="font-size: 24px;">{avg_time:.2f}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Metric 4: Average Number of Workers
        with metric_columns[3]:
            avg_wrk = round(selected_data11["Number of workers"].mean())

            st.markdown(
                f'<div style="background-color: #dda0dd; padding: 20px; border-radius: 10px; text-align: center;">'
                f'<h3 style="color: #000080;">Avg Manpower</h3>'
                f'<p style="font-size: 24px; color: #FFFFFF;">{avg_wrk}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Add spacing
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Second row

        # Prediction plot

        # Plot of Workload by Order_dates

        st.write("### Workload Prediction over Time")

        selected_data_sorted1 = selected_data11.sort_values(
            by='Date')

        fig = px.line(selected_data_sorted1, x='Date', y=['Workload(hrs)', 'prediction'], title='',
                      labels={'value': 'Workload (Hours)'}, color_discrete_sequence=['blue', 'orange'])

        fig.update_traces(line=dict(width=2))

        fig.update_layout(width=1350, height=600)

        st.plotly_chart(fig)

        # Add spacing
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Third row

        with st.expander("View Data of TimeSeries:"):
            st.write(
                selected_data_sorted1.T.style.background_gradient(cmap="Blues"))
            csv = selected_data_sorted1.to_csv(index=False).encode("utf-8")
            st.download_button("Download Data", data=csv,
                               file_name="Timeseries.csv", mime='text/csv')
