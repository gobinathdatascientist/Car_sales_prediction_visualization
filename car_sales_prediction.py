import numpy as np
import pandas as pd
from dateutil.utils import today
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns

from fontTools.misc.cython import returns
from scipy.stats import describe
from select import select

# Load the data
file = pd.read_csv('used_cars_data.csv')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
data = pd.DataFrame(file)

# Analyze the data
print(data.head(20))
data.info()
print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

# Check for duplication
print(f'Number of unique values in each column:\n{data.nunique()}')
print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

# Missing value calculations
print(f'Missing value details:\n{data.isnull().sum()}')
print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

# Calculate the percentage of missing values
pov = ((data.isnull().sum()) / len(data)) * 100
print(f'Percentage of missing values:\n{pov}')
print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

# Data reduction (Remove unwanted column)
if 'S.No.' in data.columns:
    data.drop(['S.No.'], axis=1, inplace=True)

data.info()
print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

# Feature Engineering (Creating Features)
# Creating new column 'car_age' based on 'Year'
data['car_age'] = date.today().year - data['Year']
print(data.head())
print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

# Split the "Brand" & "Model" from 'Name' column
data['brand'] = data['Name'].str.split().str.get(0)
data['model'] = data['Name'].str.split().str.get(1)

# Drop the 'Name' column after splitting
if 'Name' in data.columns:
    data.drop(['Name'], axis=1, inplace=True)
print(data.head())
print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

# Check number of unique brands
print(f'Number of unique brands: {data.brand.unique()}')
print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

# Replacing brand names without using inplace
data['brand'] = data['brand'].replace({'ISUZU': 'Isuzu', 'Mini': 'Mini Cooper', 'Land': 'Land Rover'})
# Print unique brand names
print(data.brand.unique())
print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

# EDA
print(data.describe())

print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

categorical_column=data.select_dtypes(include=['object','category'])
print(categorical_column.head())

print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')


# Select numerical columns
num_cols = data.select_dtypes(include=['float64', 'int64']).columns

# Loop through each numerical column
for col in num_cols:
    print(f'Column: {col}')
    print(f'Skewness: {round(data[col].skew(), 2)}')

    # Create a figure with two subplots (histogram and boxplot)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'Distribution and Boxplot for {col}', fontsize=16)

    # Histogram: to visualize the distribution
    sns.histplot(data[col], ax=axes[0], kde=True, color='skyblue', bins=30)  # Added KDE for better visual understanding
    axes[0].set_title(f'Histogram of {col}')
    axes[0].set_ylabel('Frequency')

    # Boxplot: to visualize outliers
    sns.boxplot(x=data[col], ax=axes[1], color='lightgreen')
    axes[1].set_title(f'Boxplot of {col}')

    # Optional: Add gridlines for better visual clarity
    axes[0].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to include suptitle
    plt.show()


fig, axes = plt.subplots(3, 2, figsize=(18, 18))
fig.suptitle('Horizontal Bar plot for all categorical variables in the dataset')

sns.countplot(ax=axes[0, 0], y='Fuel_Type', data=data, color='blue', order=data['Fuel_Type'].value_counts().index)
sns.countplot(ax=axes[0, 1], y='Transmission', data=data, color='blue', order=data['Transmission'].value_counts().index)
sns.countplot(ax=axes[1, 0], y='Owner_Type', data=data, color='blue', order=data['Owner_Type'].value_counts().index)
sns.countplot(ax=axes[1, 1], y='Location', data=data, color='blue', order=data['Location'].value_counts().index)
sns.countplot(ax=axes[2, 0], y='brand', data=data, color='blue', order=data['brand'].value_counts().index[:20])  # Top 20 Brands
sns.countplot(ax=axes[2, 1], y='model', data=data, color='blue', order=data['model'].value_counts().index[:20])  # Top 20 Models

plt.tight_layout()
plt.show()

fuel_transmission = pd.crosstab(data['Fuel_Type'], data['Transmission'])
fuel_transmission.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='coolwarm')
plt.title('Stacked Bar Plot: Fuel Type vs Transmission')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(pd.crosstab(data['Fuel_Type'], data['Transmission']), annot=True, fmt='d', cmap='Blues')
plt.title('Heatmap: Fuel Type vs Transmission')
plt.show()

plt.figure(figsize=(8, 8))
data['Fuel_Type'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'orange'])
plt.title('Fuel Type Distribution')
plt.ylabel('')
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(x='Fuel_Type', y='Price', data=data)  # Assuming 'Price' is a numerical feature
plt.title('Violin plot: Fuel Type vs Price')
plt.show()

g = sns.FacetGrid(data, col='Fuel_Type', col_wrap=3, height=4)
g.map(sns.countplot, 'Transmission', color='blue', order=data['Transmission'].value_counts().index)
g.set_titles("{col_name}")
plt.show()

avg_price = data.groupby(['Fuel_Type', 'Transmission']).Price.mean().reset_index()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=avg_price, x='Fuel_Type', y='Transmission', size='Price', hue='Price', sizes=(100, 1000), legend=False, palette='coolwarm')
plt.title('Bubble Chart: Fuel Type vs Transmission by Average Price')
plt.show()


# Set up figure size and visual theme for clarity
plt.figure(figsize=(13, 17))
sns.set_style("whitegrid")  # Improved background for readability

# Improved pairplot with additional customization
pairplot = sns.pairplot(data=data.drop(['Kilometers_Driven', 'Price'], axis=1),
                        hue='Fuel_Type',  # Color by 'Fuel_Type' or any relevant categorical variable
                        palette='Set2',   # Set color palette for better visualization
                        diag_kind='kde',  # Use KDE for diagonal plots instead of histograms
                        plot_kws={'alpha': 0.6, 's': 50})  # Customize scatter plot markers for visibility

# Additional title and tweaks
pairplot.fig.suptitle('Pairplot for Numerical Variables (Excluding Kilometers_Driven and Price)',y=1.02, fontsize=16)

plt.show()