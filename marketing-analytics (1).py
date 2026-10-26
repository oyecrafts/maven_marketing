# Generated from: marketing-analytics (1).ipynb
# Converted at: 2026-10-26T18:49:18.427Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# **Load Packages**


import plotly.io as pio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import plotly.graph_objects as go
import squarify
from sklearn.datasets import load_iris
import folium
import geopandas as gpd
import plotly.express as px
from geopy.geocoders import Nominatim

# **Read dataset into a dataframe**


marketing = pd.read_csv('marketing-data-csv')

# **Explore the data**


print(marketing.head())
column_names = marketing.columns
print(column_names)


# **Perform Data Cleaning and Manipulation**


#Handle the null values 
null_values = marketing.isnull().any(axis=1)
print(null_values)

print(marketing.dtypes)

# convert dates to datetime
invalid_dates = marketing[marketing['Dt_Customer'].isna()]

marketing['Dt_Customer'] = pd.to_datetime(
    marketing['Dt_Customer'], errors='coerce')

print(marketing.describe())

marketing['Year_Birth'] = marketing['Year_Birth'].astype(float)

print(marketing.dtypes)

# Extract month and year of enrollment
marketing['Enrollment_Month'] = marketing['Dt_Customer'].dt.month
marketing['Enrollment_Year'] = marketing['Dt_Customer'].dt.year
marketing['Enrollmnt_Day']=marketing['Dt_Customer'].dt.day
month_names = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'}


# Removing leading and trailing whitespaces from column names
marketing.columns = marketing.columns.str.strip()


# Checking if 'Income' column exists after stripping whitespaces
if 'Income' in marketing.columns:
    print(marketing['Income'].head())
else:
    print("Column 'Income' not found after stripping whitespaces.")


default_size = 0 
marketing['Income'] = marketing['Income'].fillna(default_size)


# handling the outliers
# income
marketing.drop(index=527,inplace=True)
income_threshold=105471
marketing = marketing[marketing['Income'] <= income_threshold]
print(marketing['Income'].describe())
marketing['Income'].max()

# year
marketing.loc[marketing['Year_Birth'] == 1893, 'Year_Birth'] = 1993
marketing.loc[marketing['Year_Birth'] == 1899, 'Year_Birth'] = 1999
marketing = marketing[marketing['Year_Birth'] != 1900]

max_Year = marketing['Year_Birth'].max()
print(max_Year)

# age was derived from year of enrollment and birth year
marketing['Age'] = marketing['Enrollment_Year']-marketing['Year_Birth']

print(marketing['Age'])

max_Age = marketing['Age'].max()

print(max_Age)

min_age = marketing['Age'].min()
print(min_age)


# AGE
age_bins = [min_age, 30, 40, 50, 60, max_Age]
age_labels = ['<30', '30-39', '40-49', '50-59', '60+']

marketing['Age_group'] = pd.cut(
    marketing['Age'], bins=age_bins, labels=age_labels, right=False)

print(marketing[['Age', 'Age_group']])

age_group_counts = marketing['Age_group'].value_counts()


colors= ['lightblue', 'lightgreen','lightcoral', 'lightsalmon', 'lightyellow']

# creating a new table for total purchases made by customers
marketing['Total_Purchases'] = marketing[['MntWines', 'MntFruits', 'MntMeatProducts', 
                                            'MntFishProducts', 'MntSweetProducts', 
                                            'MntGoldProds']].sum(axis=1)





# **Data Visualization**


# Bar chart of age(GROUPED)
fig = go.Figure(go.Bar(
    x=age_group_counts.index,
    y=age_group_counts.values,
    marker_color='lightcoral'))

fig.update_layout(
    title='Distribution of Age Groups',
    xaxis_title='Age Group',
    yaxis_title='Number of Individuals',
    xaxis=dict(tickangle=45),
    yaxis=dict(gridcolor='lightgrey'),
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False)


#Education
education_counts=marketing['Education'].value_counts()
fig = go.Figure(go.Pie(
    labels=education_counts.index,
    values=education_counts.values,
    hole=0.5,
    marker=dict(colors=['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightyellow'])
))

fig.update_layout(
    title='Distribution of Education Levels',
    showlegend=True
)


# Age by education
fig = go.Figure()

for edu_level, data in marketing.groupby('Education'):
    fig.add_trace(go.Box(
        y=data['Age'],
        name=edu_level
    ))

fig.update_layout(
    title='Box Plot of Age by Education Level',
    xaxis_title='Education Level',
    yaxis_title='Age',
    xaxis=dict(tickangle=45),
    showlegend=True
)


# marital status
marriage_count=marketing['Marital_Status'].value_counts()

color_map = {
    'together': 'lightgreen',
    'single': 'skyblue',
    'married': 'lightcoral',
    'divorced': 'lightsalmon',
    'widow': 'lightyellow'
}


fig = px.treemap(names=marriage_count.index, parents=[""] * len(marriage_count), values=marriage_count.values, 
                 color=marriage_count.index, color_discrete_map=color_map)

fig.update_layout(title='Treemap of Marital Status',
    hovermode='closest')


# Create histogram for income
fig = go.Figure(go.Histogram(x=marketing['Income'],
    marker_color='Lightgreen',
    opacity=0.7))

fig.update_layout(
    title='Histogram of Income',
    xaxis_title='Income',
    yaxis_title='Frequency',
    bargap=0.05,
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False)


fig = go.Figure(go.Histogram(x=marketing['Recency'], marker_color='skyblue', opacity=0.7))
              
fig.update_layout(title="Distribution of Customer Recency",
                  xaxis_title='Recency',
                  yaxis_title='Number of Individuals',
                  bargap=0.1,
                  plot_bgcolor='rgba(0,0,0,0)')


fig = go.Figure(go.Histogram(x=marketing['MntWines'], marker_color='darkgreen', opacity=0.7))

fig.update_layout(title="Wine Expenditure Over 2 Years",
                  xaxis_title='MntWines',
                  bargap=0.2,
                  plot_bgcolor='rgba(0,0,0,0)')


fig = go.Figure(go.Histogram(x=marketing['MntFruits'], marker_color='lightcoral', opacity=0.7))
fig.update_layout(title='Fruit Expenditure Over 2 Years',
                  xaxis_title='MntFruits', bargap=0.2,
                  plot_bgcolor='rgba(0,0,0,0)')


# Meat Products
fig = go.Figure(go.Histogram(x=marketing['MntMeatProducts'], marker_color='lightgreen', opacity=0.7))
fig.update_layout(title='Meat Expenditure Over 2 Years',
                  xaxis_title='MntMeatProducts',
                  bargap=0.3,
                  plot_bgcolor='rgba(0,0,0,0)')

# Fish Products
fig = go.Figure(go.Histogram(x=marketing['MntFishProducts'], marker_color='skyblue', opacity=0.7))
fig.update_layout(title='Fish Expenditure Over 2 Years',
                  xaxis_title='MntFishProducts',
                  bargap=0.2,
                  plot_bgcolor='rgba(0,0,0,0)')

# Sweet Products
fig = go.Figure(go.Histogram(x=marketing['MntSweetProducts'], opacity=0.7, marker_color='skyblue'))
fig.update_layout(title='Sweets Expenditure Over the Years',
                  xaxis_title='MntSweetProducts',
                  bargap=0.2,
                  plot_bgcolor='rgba(0,0,0,0)')


# Gold Products
fig = go.Figure(go.Histogram(x=marketing['MntGoldProds'], opacity=0.7, marker_color='lightgreen'))
fig.update_layout(title='Gold Expenditure Over the Years',
                  xaxis_title='MntGoldProds',
                  bargap=0.2,
                  plot_bgcolor='rgba(0,0,0,0)')


#Deals made
fig = go.Figure(go.Histogram(x=marketing['NumDealsPurchases'], opacity=0.7, marker_color='orange'))
fig.update_layout(title='Discounted Purchases Distribution',
                    xaxis_title='NumDealsPurchases',
                    bargap=0.2,
                    plot_bgcolor='rgba(0,0,0,0)')

# Web Purchases
fig = go.Figure(go.Histogram(x=marketing['NumWebPurchases'], opacity=0.7, marker_color='lightcoral'))
fig.update_layout(title='Distribution of Web Purchases',
                    xaxis_title='NumWebPurchases',
                    bargap=0.2,
                    plot_bgcolor='rgba(0,0,0,0)')


# Catalog Purchases
fig = go.Figure(go.Histogram(x=marketing['NumCatalogPurchases'], opacity=0.7, marker_color='orange'))
fig.update_layout(title='Distribution of Catalog Purchases',
                    xaxis_title='NumCatalogPurchases',
                    bargap=0.2,
                    plot_bgcolor='rgba(0,0,0,0)')

# Store Purchases
fig = go.Figure(go.Histogram(x=marketing['NumStorePurchases'], opacity=0.7, marker_color='lightcoral'))
fig.update_layout(title='Distribution of Store Purchases',
                    xaxis_title='NumStorePurchases',
                    bargap=0.2,
                    plot_bgcolor='rgba(0,0,0,0)')

# Web Visits
fig = go.Figure(go.Histogram(x=marketing['NumWebVisitsMonth'], opacity=0.7, marker_color='orange'))
fig.update_layout(title='Distribution of Monthly Web Visits',
                    xaxis_title='NumWebVisitsMonth',
                    bargap=0.2,
                    plot_bgcolor='rgba(0,0,0,0)')



# Campaign counts
# Campaign Acceptance Rates
campaign_counts = marketing[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].sum()
bar_trace = go.Bar(
    x=campaign_counts.index,
    y=campaign_counts.values,
    marker=dict(color=['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightyellow'])
)
layout = go.Layout(
    title='Acceptance Rates of Campaign Offers',
    xaxis=dict(title='Campaign'),
    yaxis=dict(title='Number of Responses')
)
fig = go.Figure(data=[bar_trace], layout=layout)


# Household Children Distribution
kids_count = marketing['Kidhome'].value_counts()
fig = go.Figure(go.Pie(labels=kids_count.index, values=kids_count.values,
                         marker=dict(colors=['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightyellow'])))
fig.update_layout(title='Household Children Distribution', showlegend=True)


# Household Teenager Distribution
teen_count = marketing['Teenhome'].value_counts()
fig = go.Figure(data=[go.Pie(labels=teen_count.index, values=teen_count.values, hole=0.5,
                               marker=dict(colors=colors))])
fig.update_layout(title='Household Teenager Distribution', showlegend=True)




# Education vs. Product Spending
product_categories = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

education_spending = marketing.groupby('Education')[product_categories].sum().reset_index()

education_spending_melted = pd.melt(education_spending, id_vars='Education', var_name='Product', value_name='Amount')

fig = px.bar(education_spending_melted, x='Education', y='Amount', color='Product',
               labels={'Education': 'Education Level', 'Amount': 'Amount Spent', 'Product': 'Product Category'},
               title='Amount Spent on Different Product Categories by Education Level')



# Marital Status vs. Product Spending
marriage_spending = marketing.groupby('Marital_Status')[product_categories].sum().reset_index()

marriage_spending_melted = pd.melt(marriage_spending, id_vars='Marital_Status', var_name='Product', value_name='Amount')

fig = px.bar(marriage_spending_melted, x='Marital_Status', y='Amount', color='Product',
               labels={'Marital_Status': 'Marital Status', 'Amount': 'Amount Spent', 'Product': 'Product Category'},
               title='Amount Spent on Different Product Categories by Marital Status')



# Correlation Heatmap
# Customer Complaints Distribution
complain_counts = marketing['Complain'].value_counts()

fig = go.Figure(go.Pie(labels=complain_counts.index, values=complain_counts.values,
                          hole=0.5, marker=dict(colors=['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightyellow'])))
fig.update_layout(title='Distribution of Customer Complaints',
                    showlegend=False)






# **Hypothesis Testing**



# Hypothesis
from scipy.stats import ttest_ind
# Set the significance level (alpha)
alpha = 0.05

# 1. Income and Total Money Spent:
print(marketing['Income'].mean())

#mean income is 51695
income_threshold=51695

high_income = marketing[marketing['Income'] > income_threshold]['Total_Purchases']
low_income = marketing[marketing['Income'] <= income_threshold]['Total_Purchases']

t_statistic, p_value = ttest_ind(high_income, 
                                 low_income, equal_var=False)



# Interpret the results
if p_value < alpha:
    if high_income.mean() > low_income.mean():
        print("Reject the null hypothesis. Households with high income spend more compared to households with low income.")
    else:
        print("Reject the null hypothesis. Households with low income spend more compared to households with high income.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in spending between income groups.")


# total income and number of kids
kidhome_group_1_2 = marketing[marketing['Kidhome'].isin([1, 2])]['Total_Purchases']

kidhome_group_0 = marketing[marketing['Kidhome'] == 0]['Total_Purchases']

t_statistic, p_value = ttest_ind(kidhome_group_1_2, kidhome_group_0, equal_var=False)


alpha = 0.05


if p_value < alpha:
    if kidhome_group_1_2.mean() > kidhome_group_0.mean():
        print('Reject the null hypothesis. People with children spend more money than households without children.')
    else:
        print('Reject the null hypothesis. People with children spend less money than households without children.')
else:
    print('Fail to reject the null hypothesis. There is no significant difference in money spent by households with and without children.')


# Recency and total money spent
from scipy.stats import f_oneway

# Categorize Recency into groups
bins = [-1, 30, 60, 91]  # Define bins for Low, Medium, and High Recency
labels = ['Low Recency', 'Medium Recency', 'High Recency']
marketing['Recency_Category'] = pd.cut(marketing['Recency'], bins=bins, labels=labels)

# Calculate mean income for each recency category
mean_income_by_recency = marketing.groupby('Recency_Category')['Income'].mean()

# Perform ANOVA test to compare mean income among recency categories
f_statistic, p_value = f_oneway(marketing[marketing['Recency_Category'] == 'Low Recency']['Income'],
                                 marketing[marketing['Recency_Category'] == 'Medium Recency']['Income'],
                                 marketing[marketing['Recency_Category'] == 'High Recency']['Income'])

# Set significance level
alpha = 0.05

# Interpret the results
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in mean income among recency categories.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in mean income among recency categories.")

# Print mean income for each recency category
print(mean_income_by_recency)





# **Linear Regression**


# linear regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
marketing['Enrollment_Month_num'] = marketing['Dt_Customer'].dt.strftime('%m')


selected_features = ['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 
                     'MntWines', 'MntFruits', 'MntMeatProducts', 
                     'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
                     'NumDealsPurchases', 'NumWebPurchases', 
                     'NumCatalogPurchases', 'NumStorePurchases', 
                     'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 
                     'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 
                     'Response', 'Complain', 'Enrollment_Month_num', 
                     'Enrollment_Year', 'Enrollmnt_Day', 'Age', 
                     'Total_Purchases']

# Handle categorical variables if needed (e.g., one-hot encoding)
# For linear regression, handle categorical features appropriately before model training.

# Split data into features (X) and target variable (y)
X = marketing[selected_features]
y = marketing['Recency']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (optional but can be beneficial for linear regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Instantiate and train the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
sum_y_test = sum(y_test)
sum_y_pred = sum(y_pred)

# Print the sum of y_test and y_pred
print(f"Sum of y_test values: {sum_y_test}")
print(f"Sum of y_pred values: {sum_y_pred}")

figlinear = go.Figure()

# Add scatter plot for actual versus predicted values
figlinear.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Actual vs. Predicted', 
                                marker=dict(color='purple', size=10, opacity=0.8)))

# Add a diagonal line for perfect predictions (optional)
max_val = max(y_test.max(), y_pred.max())
figlinear.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', 
                                name='Perfect Predictions', line=dict(color='gold', dash='dash')))

# Update layout with title and axis labels
figlinear.update_layout(
    title='Actual vs. Predicted Recency',
    xaxis_title='Actual Recency',
    yaxis_title='Predicted Recency',
    plot_bgcolor='#111111',  # Set background color
    paper_bgcolor='#111111',  # Set plot area background color
    font=dict(color='white'),  # Set font color to white
    showlegend=True,
    legend=dict(x=0.8, y=0.1)  # Position legend
)
pio.write_html(figlinear, 'linear_regression.html', auto_open=False)


# Most Significant Factors Related to Campaign Success



X = marketing[['Income', 'MntWines', 'MntMeatProducts', 'MntSweetProducts', 'NumCatalogPurchases','NumWebPurchases','NumDealsPurchases']]
y_variables = ['Response', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2']


coefficients_dict = {}
r2_dict = {}

for target_variable in y_variables:

    y = marketing[target_variable]
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    model = LinearRegression()
    model.fit(X_train, y_train)
    

    y_pred = model.predict(X_test)
    

    r2 = r2_score(y_test, y_pred)
    print("R-squared for {}: {}".format(target_variable, r2))
    

    coefficients = pd.DataFrame({'Variable': X.columns, 'Coefficient': model.coef_})
    coefficients_dict[target_variable] = coefficients
    r2_dict[target_variable] = r2
    
    
coefficients_dataset = pd.DataFrame()
for target_variable, coefficients_df in coefficients_dict.items():
    coefficients_df['Target'] = target_variable  # Add target variable column
    coefficients_dataset = pd.concat([coefficients_dataset, coefficients_df], ignore_index=True)
  
    
    
    

    fig = px.bar(coefficients, x='Variable', y='Coefficient', color='Coefficient',
                 color_continuous_scale='RdBu', labels={'Coefficient': 'Effect on {}'.format(target_variable)},
                 title='Effect of Variables on {}'.format(target_variable))
    fig.update_layout(
        plot_bgcolor='#111111', 
        paper_bgcolor='#111111',  
        font_color='#ffffff'
    )
    fig.show()
    pio.write_html(fig, 'linear_regression_{}.html'.format(target_variable), auto_open=True)
    




# Significant Factors Related to Web Purchases



# web purchases coefficient
features = ['Income', 'MntWines', 'MntMeatProducts', 'MntSweetProducts', 
            'NumCatalogPurchases','Age', 'NumDealsPurchases','NumWebVisitsMonth','Kidhome','Teenhome']
target = 'NumWebPurchases'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


coefficients = model.coef_
variables = X.columns


coefficients_df = pd.DataFrame({'Variable': variables, 'Coefficient': coefficients})


fig = go.Figure(data=[go.Bar(x=coefficients_df['Variable'], y=coefficients_df['Coefficient'])])
fig.update_layout(
    title="Significant Factors Related to Web Purchases",
    xaxis_title="Variable",
    yaxis_title="Coefficient",
    plot_bgcolor='#111111',
    paper_bgcolor='#111111',
    font=dict(color='white')
)
fig.show()

pio.write_html(fig, 'linear_2.html'.format(target_variable), auto_open=True)


# average customer Profile



from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
relevant_columns = ['Age', 'Income', 'Education', 'Marital_Status', 'Kidhome', 'Teenhome', 
                    'Total_Purchases', 'Recency', 'MntWines', 'MntMeatProducts', 
                    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 
                    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
                    'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 
                    'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 
                    'Response', 'Complain', 'Country']


customer_data = marketing[relevant_columns].copy()

label_encoder = LabelEncoder()
customer_data['Education'] = label_encoder.fit_transform(customer_data['Education'])
customer_data['Marital_Status'] = label_encoder.fit_transform(customer_data['Marital_Status'])
customer_data['Country'] = label_encoder.fit_transform(customer_data['Country'])


average_profile = customer_data.mean()


print("Average Customer Profile:")
print(average_profile)


relevant_categorical_columns = ['Education', 'Marital_Status', 'Country']
category_modes = {}

for col in relevant_categorical_columns:
    mode_value = marketing[col].mode()[0] 
    category_modes[col] = mode_value


print("\nMode for each categorical variable:")
print(category_modes)



# Best products
product_columns = ['MntWines', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
product_spending = marketing[product_columns].sum()


best_performing_products = product_spending.sort_values(ascending=False)


fig = px.bar(x=best_performing_products.index, y=best_performing_products.values,
             labels={'x': 'Product Category', 'y': 'Total Spending'},
             title='Best Performing Products by Total Spending',
             color=best_performing_products.index, color_discrete_sequence=['purple']*len(best_performing_products),
             template='plotly_dark')


fig.update_layout(plot_bgcolor='#111111', paper_bgcolor='#111111', font_color='white')


fig.show()
pio.write_html(fig,'best_performing.html',auto_open=True)



# underperforming channels
total_web_purchases = marketing['NumWebPurchases'].sum()
total_catalog_purchases = marketing['NumCatalogPurchases'].sum()
total_store_purchases = marketing['NumStorePurchases'].sum()


avg_web_purchases = marketing['NumWebPurchases'].mean()
avg_catalog_purchases = marketing['NumCatalogPurchases'].mean()
avg_store_purchases = marketing['NumStorePurchases'].mean()


print(f"Total Web Purchases: {total_web_purchases}")
print(f"Total Catalog Purchases: {total_catalog_purchases}")
print(f"Total Store Purchases: {total_store_purchases}")

print(f"Avg Web Purchases per Customer: {avg_web_purchases}")
print(f"Avg Catalog Purchases per Customer: {avg_catalog_purchases}")
print(f"Avg Store Purchases per Customer: {avg_store_purchases}")

# **Dashboard Vizzes**
# 


import pandas as pd
import folium
from geopy.geocoders import Nominatim

country_recency = marketing.groupby('Country')['Recency'].mean().reset_index()
country_purchases = marketing.groupby('Country')['Total_Purchases'].sum().reset_index()
country_recency = country_recency.merge(country_purchases, on='Country', how='left')
geolocator = Nominatim(user_agent="my_geocoder")

# Function to get latitude and longitude for each country
def get_lat_lon(country):
    try:
        location = geolocator.geocode(country)
        if location:
            return (location.latitude, location.longitude)
    except Exception as e:
        print(f"Error geocoding {country}: {e}")
    return (None, None)

# Apply the function to get latitude and longitude for each country
country_recency['Latitude'], country_recency['Longitude'] = zip(*country_recency['Country'].apply(get_lat_lon))

# Drop rows with missing latitude or longitude
country_recency = country_recency.dropna(subset=['Latitude', 'Longitude'])

# Create the map with dark background
m = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB dark_matter')

# Add markers to the map
for index, row in country_recency.iterrows():
    popup_text = f"Country: {row['Country']}<br>Recency: {round(row['Recency'], 2)}<br>Total Purchases: {row['Total_Purchases']}"
    folium.Marker([row['Latitude'], row['Longitude']], 
                  popup=popup_text,
                  icon=folium.Icon(color='purple')).add_to(m)

# Save the map to an HTML file
m.save('recency_and_purchases_map.html')

import warnings


axis_labels = {
    'Recency': 'Recency',
    'Income': 'Income',
    'NumWebPurchases': 'Web Purchases',
    'NumCatalogPurchases': 'Catalog Purchases'
}

# Create scatter plot matrix using Plotly Express
fig_scatter_matrix = px.scatter_matrix(
    marketing,
    dimensions=['Recency', 'Income', 'NumWebPurchases', 'NumCatalogPurchases'],
    color='NumWebPurchases',
    title='Scatter Plot Matrix of Numerical Features',
    labels=axis_labels
)

# Update layout of the scatter plot matrix figure
fig_scatter_matrix.update_layout(
    plot_bgcolor='#111111',     # Set plot background color to black
    paper_bgcolor='#111111',    # Set paper background color to black
    font_color='white',         # Set font color to white
    title=dict(font=dict(color='white'))  # Set title text color to white
)


# Correlation Heatmap
correlation_matrix = marketing[['Recency', 'Income', 'NumWebPurchases', 'NumCatalogPurchases']].corr()

figb_heatmap = px.imshow(correlation_matrix, 
                         x=['Recency', 'Income', 'NumWebPurchases', 'NumCatalogPurchases'],
                         y=['Recency', 'Income', 'NumWebPurchases', 'NumCatalogPurchases'],
                         color_continuous_scale='Viridis',
                         title='Correlation Heatmap')

figb_heatmap.update_layout(
    plot_bgcolor='#111111',       # Set plot background color to black
    paper_bgcolor='#111111',      # Set paper background color to black
    font_color='white',         # Set font color to white
    title=dict(text='Correlation Heatmap', font=dict(color='white')),  # Set title text and color
    coloraxis_colorbar=dict(title='Correlation', tickfont=dict(color='white')),  # Set colorbar title and tick color
    xaxis=dict(tickfont=dict(color='white')),  # Set x-axis tick color
    yaxis=dict(tickfont=dict(color='white'))   # Set y-axis tick color
)



# campaign sucess rate
campaign_success_rates = pd.Series({
    'AcceptedCmp1': 6.434316353887399,
    'AcceptedCmp2': 1.3404825737265416,
    'AcceptedCmp3': 7.283288650580875,
    'AcceptedCmp4': 7.462019660411082,
    'AcceptedCmp5': 7.283288650580875,
    'Response': 14.924039320822164})

    
figc_bar_chart = px.bar(campaign_success_rates, x=campaign_success_rates.index, y=campaign_success_rates.values,
                        labels={'x': 'Campaign', 'y': 'Success Rate (%)'},
                        title='Success Rates of Different Marketing Campaigns')

figc_bar_chart.update_layout(
    title_font_color='purple',  # Set title text color to purple
    plot_bgcolor='#111111',       # Set plot background color to black
    paper_bgcolor='#111111',      # Set paper background color to black
    font_color='white',         # Set font color to white
    title=dict(font=dict(color='white'))  # Set title text color to white
)

figc_bar_chart.update_traces(marker_color='purple', opacity=0.8)  # Set bar color to purple

pio.write_html(figc_bar_chart, 'campaign_success_rates_bar_chart.html', auto_open=True)


# sucess pie chart
figd_pie_chart = px.pie(campaign_success_rates, values=campaign_success_rates.values, names=campaign_success_rates.index,
                        title='Success Rates of Different Marketing Campaigns')

figd_pie_chart.update_layout(
    plot_bgcolor='#111111',       # Set plot background color to black
    paper_bgcolor='#111111',      # Set paper background color to black
    font_color='white',         # Set font color to white
    title=dict(font=dict(color='white')),  # Set title text color to white
)

purple_colors = ['#8c6bb1', '#8e7cc3', '#7281a3', '#6a51a3', '#5a189a']

figd_pie_chart.update_traces(marker=dict(colors=purple_colors))



# Line Plot for trend of campaign success rates over time

marketing['Month_Year'] = pd.to_datetime(marketing['Dt_Customer']).dt.strftime('%m-%Y')
marketing['Month_Year'] = pd.to_datetime(marketing['Month_Year'], format='%m-%Y')
campaign_data = marketing[['Month_Year', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']]

campaign_data = campaign_data.melt(id_vars='Month_Year', var_name='Campaign', value_name='Success_Rate')

campaign_data = campaign_data.dropna()

campaign_data = campaign_data.groupby(['Campaign', 'Month_Year'], as_index=False).mean()

fig1_line_plot = px.line(campaign_data, x='Month_Year', y='Success_Rate', color='Campaign',
                        title='Trend of Campaign Success Rates Over Time',
                        labels={'Month_Year': 'Month/Year', 'Success_Rate': 'Success Rate (%)'},
                        color_discrete_sequence=px.colors.qualitative.Pastel)  # Use a qualitative color palette

fig1_line_plot.update_layout(
    plot_bgcolor='#111111',      
    paper_bgcolor='#111111',     
    font_color='white',       
    title=dict(font=dict(color='white')),  
    xaxis=dict(title='Month/Year', color='white'),  
    yaxis=dict(title='Success Rate (%)', color='white') 
)


numeric_columns = ['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

marketing[numeric_columns] = marketing[numeric_columns].apply(pd.to_numeric, errors='coerce')

avg_education = marketing.groupby('Education')[numeric_columns].mean().reset_index()

avg_marital_status = marketing.groupby('Marital_Status')[numeric_columns].mean().reset_index()

avg_overall = marketing[numeric_columns].mean().reset_index()
avg_overall.columns = ['Feature', 'Average']

fig2 = px.bar(avg_education, x='Education', y=numeric_columns,
              title='Average Values by Education Level',
              labels={'value': 'Average Value', 'variable': 'Product Category'},
              barmode='stack',opacity=0.8)


fig2.update_layout(
    plot_bgcolor='#111111',       
    paper_bgcolor='#111111',      
    font_color='white',         
    title=dict(font=dict(color='white')),  
    xaxis=dict(title='Education Level', color='white'), 
    yaxis=dict(title='Average Value', color='white')
)

fig2.update_traces(marker_color='purple')  

pio.write_html(fig2, 'avg_by_education.html', auto_open=False)

figii = px.bar(avg_marital_status.reset_index(), x='Marital_Status', y=numeric_columns,
               title='Average Values by Marital Status',
               labels={'value': 'Average Value', 'variable': 'Product Category'},opacity=0.8)


figii.update_layout(
    plot_bgcolor='#111111',     
    paper_bgcolor='#111111',     
    font_color='white',      
    title=dict(font=dict(color='white')), 
    xaxis=dict(title='Marital Status', color='white'), 
    yaxis=dict(title='Average Value', color='white')   
)

figii.update_traces(marker_color='purple')  

#  radar chart for overall average values across different features
data_dict = {
    'Feature': ['MntFruits', 'MntSweetProducts', 'MntFishProducts', 'MntGoldProds', 'MntMeatProducts', 'MntWines', 'Income'],
    'Average': [26.31680071, 27.08489723, 37.55227882, 44.04468275, 167.0875782, 304.1966041, 51419.39366]
}

# Create DataFrame from the data dictionary
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.preprocessing import StandardScaler

numerical_columns = ['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 
                     'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

# Calculate the average values of selected numerical columns
average_values = marketing[numerical_columns].mean()

# Create a DataFrame from the average values
average_data = pd.DataFrame({
    'Feature': average_values.index, 
    'Average': average_values.values   
})

# Extract features and average values
features = average_data['Feature']
averages = average_data['Average']

# Normalize the averages using z-score normalization
scaler = StandardScaler()
averages_normalized = scaler.fit_transform(averages.values.reshape(-1, 1)).flatten()

# Create a radar chart using Plotly
fig4 = go.Figure()

fig4.add_trace(go.Scatterpolar(
    r=averages_normalized,
    theta=features,
    fill='toself',
    name='Average',
    line=dict(color='purple')
))


fig4.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[-2, 2]  
        )
    ),
    showlegend=True,
    title='Radar Chart of Normalized Average Values',  
    plot_bgcolor='#111111', 
    paper_bgcolor='#111111', 
    font_color='white'     
)

#Total spending
spending_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

# Group by product categories and sum up the spending
product_spending = marketing[spending_columns].sum()

product_spending_df = pd.DataFrame({
    'Product_Category': spending_columns,
    'Total_Amount_Spent': product_spending.values
})

figv = px.bar(product_spending_df, x='Product_Category', y='Total_Amount_Spent', 
              title='Total Amount Spent on Product Categories',
              labels={'Product_Category': 'Product Category', 'Total_Amount_Spent': 'Total Amount Spent'},opacity=0.8)


figv.update_traces(marker_color='purple')  


figv.update_layout(
    plot_bgcolor='#111111',    
    paper_bgcolor='#111111',    
    font_color='white',      
    title_font_color='white',
    xaxis=dict(tickfont_color='white'), 
    yaxis=dict(tickfont_color='white')   
)



#webvisits
marketing['NumWebVisitsMonth'] = pd.to_numeric(marketing['NumWebVisitsMonth'], errors='coerce')
marketing['NumWebPurchases'] = pd.to_numeric(marketing['NumWebPurchases'], errors='coerce')

web_data = marketing[['Month_Year', 'NumWebVisitsMonth', 'NumWebPurchases']]

# Group by 'Month_Year' and calculate monthly averages
web_data = web_data.groupby('Month_Year').mean().reset_index()

# Create the line plot using Plotly Express
fig_web_chart5 = px.line(web_data, x='Month_Year', y=['NumWebVisitsMonth', 'NumWebPurchases'],
                        title='Trend of Website Visits and Purchases Over Time',
                        labels={'Month_Year': 'Month/Year', 'value': 'Count'},
                        color_discrete_sequence=['purple', 'deepskyblue'],  # Specify line colors
                        line_shape='spline')  # Adjust plot size


fig_web_chart5.update_layout(
    plot_bgcolor='#111111',       # Set plot background color to black
    paper_bgcolor='#111111',      # Set paper background color to black
    font_color='white',         # Set font color to white
    title=dict(font=dict(color='white')),  # Set title text color to white
    xaxis=dict(title='Month/Year', color='white'),  # Set x-axis label color to white
    yaxis=dict(title='Count', color='white')  # Set y-axis label color to white
)





#  purchases
web_purchases = marketing['NumWebPurchases'].sum()
catalog_purchases = marketing['NumCatalogPurchases'].sum()
store_purchases = marketing['NumStorePurchases'].sum()

channel_data = pd.DataFrame({
    'Channel': ['Web', 'Catalog', 'Store'],
    'Total Purchases': [web_purchases, catalog_purchases, store_purchases]
})

fig6 = px.bar(channel_data, x='Channel', y='Total Purchases', 
             title='Distribution of Purchases by Channel',
             labels={'Channel': 'Purchase Channel', 'Total Purchases': 'Total Purchases'},opacity=0.8)

fig6.update_traces(marker_color='purple', opacity=0.8)  
fig6.update_layout(template='plotly_dark')  

pio.write_html(fig6, 'channel_purchases_distribution.html', auto_open=False)



# null Values
fig7 = px.bar(x=null_values.index, y=null_values.values, 
              labels={'x': 'Feature', 'y': 'Number of Null Values'},
              title='Distribution of Null Values Across Features')


fig7.update_layout(
    title_font_color='white',    
    plot_bgcolor='#111111',       
    paper_bgcolor='#111111',       
    font_color='white'   
)

fig7.update_traces(marker_color='purple', opacity=0.8) 


# num and web
numerical_cols = ['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumWebPurchases']


fig_heatmap10 = px.imshow(correlation_matrix,
                        labels=dict(x='Numerical Features', y='Numerical Features', color='Correlation'),
                        title='Correlation Between Web Purchases and Other Numerical Features')


fig_heatmap10.update_layout(
    title='Correlation Heatmap',
    title_font_color='white',     
    plot_bgcolor='#111111',          
    paper_bgcolor='#111111',        
    font_color='white',            
    xaxis=dict(tickfont=dict(color='white')), 
    yaxis=dict(tickfont=dict(color='white')),  
    width=500,                     
    height=600,                   
)



# web and features
correlation_matrix = marketing[numerical_cols].corr()

# multivariate_scatter_plot
fig_multivariate_plot = px.scatter(marketing, x='Age', y='Total_Purchases', color='Education', 
                 size='Income', hover_data=['Marital_Status', 'Recency'],
                 title='Customer Analysis', labels={'Age': 'Customer Age', 'Total_Purchases': 'Total Purchases'},
                 template='plotly_dark') 
fig_multivariate_plot.show()


#general Heatmap
numeric_columns = marketing.select_dtypes(include=['number']).drop(['Year_Birth', 'ID'], axis=1)

# Calculate correlation matrix
correlation_matrix = numeric_columns.corr()

# Create heatmap using go.Heatmap
figh = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.index,
    colorbar=dict(
        tickfont=dict(color='white') 
    )
))

# Customize layout for the heatmap
figh.update_layout(
    title='Correlation Heatmap of Marketing Dataset',
    title_font_color='white',       
    plot_bgcolor='black',          
    paper_bgcolor='black',        
    font_color='white',             
    xaxis=dict(tickfont=dict(color='white')), 
    yaxis=dict(tickfont=dict(color='white')),  
    height=600,                    
    width=1200,                    
    margin=dict(l=50, r=50, t=80, b=50) 
)

# income recency and date
fig3d = px.scatter_3d(marketing, x='Income', y='Recency', z='Enrollment_Month', color='Income',title='Customer Behavior in 3D: Income, Recency, and Enrollment')

fig3d.update_layout(
    scene=dict(
        xaxis=dict(title='Income'),
        yaxis=dict(title='Recency'),
        zaxis=dict(title='Enrollment Month')
    ),
    font=dict(color='white'),  
    paper_bgcolor='#111111'      
)


caption_text = "July experienced a significant drop in income,<br>accompanied by the lowest recency of 4,974."


fig3d.add_annotation(
    x=0.9,   
    y=0.95,  
    xref="paper",
    yref="paper",
    text=caption_text,
    showarrow=False,
    align="left",
    font=dict(color='white', size=12),
    bgcolor='purple',
    bordercolor='black',
    borderwidth=1,
    borderpad=10,
    opacity=0.8
)


# Country Map based on recency and purchases
import folium
from geopy.geocoders import Nominatim
from folium.features import CustomIcon




# Scatter Matrix
fig_scatter_matrix = px.scatter_matrix(marketing,
                                       dimensions=['Recency', 'Income', 'NumWebPurchases', 'NumCatalogPurchases'],
                                       color='NumWebPurchases',
                                       title='Scatter Plot Matrix of Numerical Features',
                                       labels=axis_labels)

fig_scatter_matrix.update_layout(
    plot_bgcolor='#111111',       # Set plot background color to black
    paper_bgcolor='#111111',      # Set paper background color to black
    font_color='white',         # Set font color to white
    title=dict(font=dict(color='white'))  # Set title text color to white
)


# Correlation Heatmap
correlation_matrix = marketing[['Recency', 'Income', 'NumWebPurchases', 'NumCatalogPurchases']].corr()

figb_heatmap = px.imshow(correlation_matrix, 
                         x=['Recency', 'Income', 'NumWebPurchases', 'NumCatalogPurchases'],
                         y=['Recency', 'Income', 'NumWebPurchases', 'NumCatalogPurchases'],
                         color_continuous_scale='Viridis',
                         title='Correlation Heatmap')

figb_heatmap.update_layout(
    plot_bgcolor='#111111',       # Set plot background color to black
    paper_bgcolor='#111111',      # Set paper background color to black
    font_color='white',         # Set font color to white
    title=dict(text='Correlation Heatmap', font=dict(color='white')),  # Set title text and color
    coloraxis_colorbar=dict(title='Correlation', tickfont=dict(color='white')),  # Set colorbar title and tick color
    xaxis=dict(tickfont=dict(color='white')),  # Set x-axis tick color
    yaxis=dict(tickfont=dict(color='white'))   # Set y-axis tick color
)

# campaign sucess rate
campaign_success_rates = pd.Series({
    'AcceptedCmp1': 6.434316353887399,
    'AcceptedCmp2': 1.3404825737265416,
    'AcceptedCmp3': 7.283288650580875,
    'AcceptedCmp4': 7.462019660411082,
    'AcceptedCmp5': 7.283288650580875,
    'Response': 14.924039320822164})

    
figc_bar_chart = px.bar(campaign_success_rates, x=campaign_success_rates.index, y=campaign_success_rates.values,
                        labels={'x': 'Campaign', 'y': 'Success Rate (%)'},
                        title='Success Rates of Different Marketing Campaigns')

figc_bar_chart.update_layout(
    title_font_color='purple',  # Set title text color to purple
    plot_bgcolor='#111111',       # Set plot background color to black
    paper_bgcolor='#111111',      # Set paper background color to black
    font_color='white',         # Set font color to white
    title=dict(font=dict(color='white'))  # Set title text color to white
)

figc_bar_chart.update_traces(marker_color='purple', opacity=0.8)  # Set bar color to purple

# sucess pie chart
figd_pie_chart = px.pie(campaign_success_rates, values=campaign_success_rates.values, names=campaign_success_rates.index,
                        title='Success Rates of Different Marketing Campaigns')

figd_pie_chart.update_layout(
    plot_bgcolor='#111111',       # Set plot background color to black
    paper_bgcolor='#111111',      # Set paper background color to black
    font_color='white',         # Set font color to white
    title=dict(font=dict(color='white')),  # Set title text color to white
)

purple_colors = ['#8c6bb1', '#8e7cc3', '#7281a3', '#6a51a3', '#5a189a']

figd_pie_chart.update_traces(marker=dict(colors=purple_colors))



# Line Plot for trend of campaign success rates over time

marketing['Month_Year'] = pd.to_datetime(marketing['Dt_Customer']).dt.strftime('%m-%Y')
marketing['Month_Year'] = pd.to_datetime(marketing['Month_Year'], format='%m-%Y')
campaign_data = marketing[['Month_Year', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']]

campaign_data = campaign_data.melt(id_vars='Month_Year', var_name='Campaign', value_name='Success_Rate')

campaign_data = campaign_data.dropna()

campaign_data = campaign_data.groupby(['Campaign', 'Month_Year'], as_index=False).mean()

fig1_line_plot = px.line(campaign_data, x='Month_Year', y='Success_Rate', color='Campaign',
                        title='Trend of Campaign Success Rates Over Time',
                        labels={'Month_Year': 'Month/Year', 'Success_Rate': 'Success Rate (%)'},
                        color_discrete_sequence=px.colors.qualitative.Pastel)  # Use a qualitative color palette

fig1_line_plot.update_layout(
    plot_bgcolor='#111111',      
    paper_bgcolor='#111111',     
    font_color='white',       
    title=dict(font=dict(color='white')),  
    xaxis=dict(title='Month/Year', color='white'),  
    yaxis=dict(title='Success Rate (%)', color='white') 
)


numeric_columns = ['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

marketing[numeric_columns] = marketing[numeric_columns].apply(pd.to_numeric, errors='coerce')

avg_education = marketing.groupby('Education')[numeric_columns].mean().reset_index()

avg_marital_status = marketing.groupby('Marital_Status')[numeric_columns].mean().reset_index()

avg_overall = marketing[numeric_columns].mean().reset_index()
avg_overall.columns = ['Feature', 'Average']

fig2 = px.bar(avg_education, x='Education', y=numeric_columns,
              title='Average Values by Education Level',
              labels={'value': 'Average Value', 'variable': 'Product Category'},
              barmode='stack',opacity=0.8)


fig2.update_layout(
    plot_bgcolor='#111111',       
    paper_bgcolor='#111111',      
    font_color='white',         
    title=dict(font=dict(color='white')),  
    xaxis=dict(title='Education Level', color='white'), 
    yaxis=dict(title='Average Value', color='white')
)

fig2.update_traces(marker_color='purple')  

figii = px.bar(avg_marital_status.reset_index(), x='Marital_Status', y=numeric_columns,
               title='Average Values by Marital Status',
               labels={'value': 'Average Value', 'variable': 'Product Category'},opacity=0.8)


figii.update_layout(
    plot_bgcolor='#111111',     
    paper_bgcolor='#111111',     
    font_color='white',      
    title=dict(font=dict(color='white')), 
    xaxis=dict(title='Marital Status', color='white'), 
    yaxis=dict(title='Average Value', color='white')   
)

figii.update_traces(marker_color='purple')  

#  radar chart for overall average values across different features
data_dict = {
    'Feature': ['MntFruits', 'MntSweetProducts', 'MntFishProducts', 'MntGoldProds', 'MntMeatProducts', 'MntWines', 'Income'],
    'Average': [26.31680071, 27.08489723, 37.55227882, 44.04468275, 167.0875782, 304.1966041, 51419.39366]
}

# Create DataFrame from the data dictionary
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.preprocessing import StandardScaler

numerical_columns = ['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 
                     'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

# Calculate the average values of selected numerical columns
average_values = marketing[numerical_columns].mean()

# Create a DataFrame from the average values
average_data = pd.DataFrame({
    'Feature': average_values.index, 
    'Average': average_values.values   
})

# Extract features and average values
features = average_data['Feature']
averages = average_data['Average']

# Normalize the averages using z-score normalization
scaler = StandardScaler()
averages_normalized = scaler.fit_transform(averages.values.reshape(-1, 1)).flatten()

# Create a radar chart using Plotly
fig4 = go.Figure()

fig4.add_trace(go.Scatterpolar(
    r=averages_normalized,
    theta=features,
    fill='toself',
    name='Average',
    line=dict(color='purple')
))


fig4.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[-2, 2]  
        )
    ),
    showlegend=True,
    title='Radar Chart of Normalized Average Values',  
    plot_bgcolor='#111111', 
    paper_bgcolor='#111111', 
    font_color='white'     
)

#Total spending
spending_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

# Group by product categories and sum up the spending
product_spending = marketing[spending_columns].sum()

product_spending_df = pd.DataFrame({
    'Product_Category': spending_columns,
    'Total_Amount_Spent': product_spending.values
})

figv = px.bar(product_spending_df, x='Product_Category', y='Total_Amount_Spent', 
              title='Total Amount Spent on Product Categories',
              labels={'Product_Category': 'Product Category', 'Total_Amount_Spent': 'Total Amount Spent'},opacity=0.8)


figv.update_traces(marker_color='purple')  


figv.update_layout(
    plot_bgcolor='#111111',    
    paper_bgcolor='#111111',    
    font_color='white',      
    title_font_color='white',
    xaxis=dict(tickfont_color='white'), 
    yaxis=dict(tickfont_color='white')   
)


#webvisits
marketing['NumWebVisitsMonth'] = pd.to_numeric(marketing['NumWebVisitsMonth'], errors='coerce')
marketing['NumWebPurchases'] = pd.to_numeric(marketing['NumWebPurchases'], errors='coerce')

web_data = marketing[['Month_Year', 'NumWebVisitsMonth', 'NumWebPurchases']]

# Group by 'Month_Year' and calculate monthly averages
web_data = web_data.groupby('Month_Year').mean().reset_index()

# Create the line plot using Plotly Express
fig_web_chart5 = px.line(web_data, x='Month_Year', y=['NumWebVisitsMonth', 'NumWebPurchases'],
                        title='Trend of Website Visits and Purchases Over Time',
                        labels={'Month_Year': 'Month/Year', 'value': 'Count'},
                        color_discrete_sequence=['purple', 'deepskyblue'],  # Specify line colors
                        line_shape='spline')  # Adjust plot size


fig_web_chart5.update_layout(
    plot_bgcolor='#111111',       # Set plot background color to black
    paper_bgcolor='#111111',      # Set paper background color to black
    font_color='white',         # Set font color to white
    title=dict(font=dict(color='white')),  # Set title text color to white
    xaxis=dict(title='Month/Year', color='white'),  # Set x-axis label color to white
    yaxis=dict(title='Count', color='white')  # Set y-axis label color to white
)






#  purchases
web_purchases = marketing['NumWebPurchases'].sum()
catalog_purchases = marketing['NumCatalogPurchases'].sum()
store_purchases = marketing['NumStorePurchases'].sum()

channel_data = pd.DataFrame({
    'Channel': ['Web', 'Catalog', 'Store'],
    'Total Purchases': [web_purchases, catalog_purchases, store_purchases]
})

fig6 = px.bar(channel_data, x='Channel', y='Total Purchases', 
             title='Distribution of Purchases by Channel',
             labels={'Channel': 'Purchase Channel', 'Total Purchases': 'Total Purchases'},opacity=0.8)

fig6.update_traces(marker_color='purple', opacity=0.8)  
fig6.update_layout(template='plotly_dark')  


# null Values
fig7 = px.bar(x=null_values.index, y=null_values.values, 
              labels={'x': 'Feature', 'y': 'Number of Null Values'},
              title='Distribution of Null Values Across Features')


fig7.update_layout(
    title_font_color='white',    
    plot_bgcolor='#111111',       
    paper_bgcolor='#111111',       
    font_color='white'   
)

fig7.update_traces(marker_color='purple', opacity=0.8) 


# num and web
numerical_cols = ['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumWebPurchases']


fig_heatmap10 = px.imshow(correlation_matrix,
                        labels=dict(x='Numerical Features', y='Numerical Features', color='Correlation'),
                        title='Correlation Between Web Purchases and Other Numerical Features')


fig_heatmap10.update_layout(
    title='Correlation Heatmap',
    title_font_color='white',     
    plot_bgcolor='#111111',          
    paper_bgcolor='#111111',        
    font_color='white',            
    xaxis=dict(tickfont=dict(color='white')), 
    yaxis=dict(tickfont=dict(color='white')),  
    width=500,                     
    height=600,                   
)



# web and features
correlation_matrix = marketing[numerical_cols].corr()

# multivariate_scatter_plot
fig_multivariate_plot = px.scatter(marketing, x='Age', y='Total_Purchases', color='Education', 
                 size='Income', hover_data=['Marital_Status', 'Recency'],
                 title='Customer Analysis', labels={'Age': 'Customer Age', 'Total_Purchases': 'Total Purchases'},
                 template='plotly_dark') 
fig_multivariate_plot.show()


#general Heatmap
numeric_columns = marketing.select_dtypes(include=['number']).drop(['Year_Birth', 'ID'], axis=1)

# Calculate correlation matrix
correlation_matrix = numeric_columns.corr()

# Create heatmap using go.Heatmap
figh = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.index,
    colorbar=dict(
        tickfont=dict(color='white') 
    )
))

# Customize layout for the heatmap
figh.update_layout(
    title='Correlation Heatmap of Marketing Dataset',
    title_font_color='white',       
    plot_bgcolor='black',          
    paper_bgcolor='black',        
    font_color='white',             
    xaxis=dict(tickfont=dict(color='white')), 
    yaxis=dict(tickfont=dict(color='white')),  
    height=600,                    
    width=1200,                    
    margin=dict(l=50, r=50, t=80, b=50) 
)

# income recency and date
fig3d = px.scatter_3d(marketing, x='Income', y='Recency', z='Enrollment_Month', color='Income',title='Customer Behavior in 3D: Income, Recency, and Enrollment')

fig3d.update_layout(
    scene=dict(
        xaxis=dict(title='Income'),
        yaxis=dict(title='Recency'),
        zaxis=dict(title='Enrollment Month')
    ),
    font=dict(color='white'),  
    paper_bgcolor='#111111'      
)


caption_text = "July experienced a significant drop in income,<br>accompanied by the lowest recency of 4,974."


fig3d.add_annotation(
    x=0.9,   
    y=0.95,  
    xref="paper",
    yref="paper",
    text=caption_text,
    showarrow=False,
    align="left",
    font=dict(color='white', size=12),
    bgcolor='purple',
    bordercolor='black',
    borderwidth=1,
    borderpad=10,
    opacity=0.8
)

fig3d.show()

#Indicators
from plotly.subplots import make_subplots
total_recency = marketing['Recency'].sum()
total_income = marketing['Income'].sum()
total_purchases = marketing['Total_Purchases'].sum()
total_web_visits = marketing['NumWebVisitsMonth'].sum()
total_web_purchases=marketing['NumWebPurchases'].sum()
total_individuals=marketing['ID'].count()



total_recency = marketing['Recency'].sum()


fig_indicator = go.Figure(go.Indicator(
    mode="number",
    value=total_recency,
    number={"font": {"color": "white", "size": 40}},
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Total Recency", 'font': {'color': 'white'}},

    gauge={'shape': 'angular', 'bgcolor': 'purple', 'bordercolor': 'purple', 'borderwidth': 2},
))

fig_indicator.update_layout(
    width=150,
    height=100,
    margin=dict(l=10, r=10, t=10, b=10),
    template="plotly_dark",
    plot_bgcolor='purple',  
)


pio.write_html(fig_indicator, "indicator.html", auto_open=False)


# total income
fig_indicator2 = go.Figure(go.Indicator(
    mode="number",
    value=total_income,
    number={"font": {"color": "white", "size": 40}},
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Total Income", 'font': {'color': 'white'}},
    gauge={'shape': 'angular', 'bgcolor': 'purple', 'bordercolor': 'purple', 'borderwidth': 2},
))

fig_indicator2.update_layout(
    width=150,
    height=100,
    margin=dict(l=10, r=10, t=10, b=10),
    template="plotly_dark",
    plot_bgcolor='rgba(0,0,0,0)',  )


pio.write_html(fig_indicator2, "indicator2.html", auto_open=False)




# total purchases
fig_indicator3 = go.Figure(go.Indicator(
    mode="number",
    value=total_purchases,
    number={"font": {"color": "white", "size": 40}},
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Total Purchases", 'font': {'color': 'white'}},
    gauge={'shape': 'angular', 'bgcolor': 'purple', 'bordercolor': 'purple', 'borderwidth': 2},
))

fig_indicator3.update_layout(
    width=150,
    height=100,
    margin=dict(l=10, r=10, t=10, b=10),
    template="plotly_dark",
    plot_bgcolor='black'
)

pio.write_html(fig_indicator3, "indicator3.html", auto_open=False)



# total_web_visits
fig_indicator4 = go.Figure(go.Indicator(
    mode="number",
    value=total_web_visits,
    number={"font": {"color": "white", "size": 40}},
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Total Web Visits", 'font': {'color': 'white'}},
    gauge={'shape': 'angular', 'bgcolor': 'purple', 'bordercolor': 'purple', 'borderwidth': 2},
))

fig_indicator4.update_layout(
    width=150,
    height=100,
    margin=dict(l=10, r=10, t=10, b=10),
    template="plotly_dark",
    plot_bgcolor='rgba(0,0,0,0)'
)

pio.write_html(fig_indicator4, "indicator4.html", auto_open=False)


#total web purchases
fig_indicator5 = go.Figure(go.Indicator(
    mode="number",
    value=total_web_purchases,
    number={"font": {"color": "white", "size": 40}},
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Total Web Purchases", 'font': {'color': 'white'}},
    gauge={'shape': 'angular', 'bgcolor': 'purple', 'bordercolor': 'purple', 'borderwidth': 2},
))

fig_indicator5.update_layout(
    width=150,
    height=100,
    margin=dict(l=10, r=10, t=10, b=10),
    template="plotly_dark",
    plot_bgcolor='rgba(0,0,0,0)'
)

pio.write_html(fig_indicator5, "indicator5.html", auto_open=False)

#Number of Unique Customers
fig_indicator6 = go.Figure(go.Indicator(
    mode="number",
    value=total_individuals,
    number={"font": {"color": "white", "size": 40}},
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Total Unique Customers", 'font': {'color': 'white'}},
    gauge={'shape': 'angular', 'bgcolor': 'purple', 'bordercolor': 'purple', 'borderwidth': 2},
))


fig_indicator6.update_layout(
    width=150,
    height=100,
    margin=dict(l=10, r=10, t=10, b=10),
    template="plotly_dark",
    plot_bgcolor='rgba(0,0,0,0)' 
)

pio.write_html(fig_indicator6, "indicator6.html", auto_open=False)







# **Dashboard**


pip install dash


import dash
from dash import html
from dash import dcc

from dash.dependencies import Input, Output


# Initialize the Dash app
app = dash.Dash(__name__)

# Define the style for the dashboard
app.layout = html.Div(
    style={'backgroundColor': '#111111', 'padding': '20px', 'display': 'flex', 'flexDirection': 'column'},
    children=[
        html.H1(
            children='Sales and Marketing Insights',
            style={'textAlign': 'center', 'color': '#ffffff'}
        ),
        html.Div(
            style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'},
            children=[
                html.Div(
                    style={'marginRight': '60px'},
                    children=[
                        dcc.Graph(figure=fig_indicator, config={'displayModeBar': False})
                    ]
                ),
                html.Div(
                    style={'marginRight': '60px'},
                    children=[
                        dcc.Graph(figure=fig_indicator2, config={'displayModeBar': False})
                    ]
                ),
                html.Div(
                    style={'marginRight': '60px'},
                    children=[
                        dcc.Graph(figure=fig_indicator3, config={'displayModeBar': False})
                    ]
                ),
                html.Div(
                    style={'marginRight': '50px'},
                    children=[
                        dcc.Graph(figure=fig_indicator4, config={'displayModeBar': False})
                    ]
                ),
                html.Div(
                    style={'marginRight': '50px'},
                    children=[
                        dcc.Graph(figure=fig_indicator5, config={'displayModeBar': False})
                    ]
                ),
                html.Div(
                    children=[
                        dcc.Graph(figure=fig_indicator6, config={'displayModeBar': False})
                    ]
                )
            ]
        ),
     
        html.Div(
            style={'marginBottom': '20px'},
            children=[
                html.Iframe(srcDoc=open('recency_and_purchases_map.html', 'r').read(),
                            width='100%', height='400px')
            ]
        ),
        
        html.Div(
            style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px','border': '2px solid white'},
            children=[
                html.Div(
                    style={'width': '60%', 'marginRight': '10px'},
                    children=[
                        dcc.Graph(figure=fig4, config={'displayModeBar': False})
                    ]
                ),
                html.Div(
                    style={'width': '60%'},
                    children=[
                        dcc.Graph(figure=figlinear, config={'displayModeBar': False})
                    ]
                )
            ]
        ),

        html.Div(
            style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '30px','border': '2px solid white'},
            children=[
                html.Div(
                    style={'width': '100%', 'height': '450px','padding': '30px'},
                    children=[
                        dcc.Graph(figure=fig3d, config={'displayModeBar': False})
                    ]
                )
            ]
        ),
        html.Div(
            style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px','border': '2px solid white'},
            children=[
                html.Div(
                    style={'width': '60%', 'marginRight': '10px','borderRight': '2px solid white'},
                    children=[
                        dcc.Graph(figure=fig2, config={'displayModeBar': False})
                    ]
                ),
                html.Div(
                    style={'width': '60%'},
                    children=[
                        dcc.Graph(figure=fig1_line_plot, config={'displayModeBar': False})
                    ]
                )
            ]
        ),   
                html.Div(
                    style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px','border': '2px solid white'},
                    children=[
                        html.Div(
                            style={'width': '100%', 'height': '450px','padding': '20px'},
                            children=[
                                dcc.Graph(figure=fig_multivariate_plot, config={'displayModeBar': False})
                            ]
                        )
                    ]
                ),
                html.Div(
                    style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px','border': '2px solid white'},
                    children=[
                        html.Div(
                            style={'width': '60%', 'marginRight': '10px','borderRight': '2px solid white'},
                            children=[
                                dcc.Graph(figure=fig7, config={'displayModeBar': False})
                            ]
                        ),
                        html.Div(
                            style={'width': '60%'},
                            children=[
                                dcc.Graph(figure=fig6, config={'displayModeBar': False})
                            ]
                        )
                    ]
                ),  
                        html.Div(
                            style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px','border': '2px solid white'},
                            children=[
                                html.Div(
                                    style={'width': '100%', 'height': '450px','padding': '20px'},
                                    children=[
                                        dcc.Graph(figure=fig_scatter_matrix, config={'displayModeBar': False})
                                    ]
                                )
                            ]
                        ), html.Div(
                      style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px','border': '2px solid white'},
                      children=[
                          html.Div(
                              style={'width': '60%', 'marginRight': '10px','borderRight': '2px solid white'},
                              children=[
                                  dcc.Graph(figure=figd_pie_chart, config={'displayModeBar': False})
                              ]
                          ),
                          html.Div(
                              style={'width': '60%'},
                              children=[
                                  dcc.Graph(figure=figii, config={'displayModeBar': False})
                              ]
                          )
                      ]
                  ),html.Div(
                        style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px','border': '2px solid white'},
                        children=[
                            html.Div(
                                style={'width': '60%', 'marginRight': '10px','borderRight': '2px solid white'},
                                children=[
                                    dcc.Graph(figure=fig_web_chart5, config={'displayModeBar': False})
                                ]
                            ),
                            html.Div(
                                style={'width': '60%'},
                                children=[
                                    dcc.Graph(figure=figc_bar_chart, config={'displayModeBar': False})
                                ]
                            )
                        ]
                    ), html.Div(
            style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px','border': '2px solid white'},
            children=[
                html.Div(
                    style={'width': '100%'},
                    children=[
                        dcc.Graph(figure=figh, config={'displayModeBar': False})
                    ]
                                   )
                               ]
                           )
                       ])
    

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8051)