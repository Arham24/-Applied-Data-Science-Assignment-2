import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def read_data(filename):
    """Reads a dataframe in Worldbank format and returns two dataframes: one with years as columns and one with
    countries as columns."""
    df = pd.read_csv(filename, header=2)

    # Remove any leading/trailing whitespaces from column names
    df.columns = df.columns.str.strip()
    
    # Drop the extra column(s)
    df = df.drop(columns=['Unnamed: 66'])
    dtf = df
    ## NaN values with the mean of the respective column
    dtf = dtf.fillna(df.mean(numeric_only=True))
#     print(dtf.info())
    
    # Set the index to 'Country Name' and 'Indicator Name'
    df = df.set_index(['Country Name', 'Indicator Name'])
    
    # Fill NaN values with the mean of the respective column
    df = df.fillna(df.mean(numeric_only=True))

    # Transpose the dataframe to get years as columns
    df_years = df.T
    
    # Reset the index to get years as a column
    df_years = df_years.reset_index()
    df_years = df_years.rename(columns={'index': 'Year'})
    df_years = df_years.set_index('Year')
    df_years = df_years.T
    
#     display(df_years)
    print(df_years.info())
    
    # Get countries as columns
    df_countries =  df_years.T
#     display(df_countries)
    print(df_countries.info())

    
    return dtf, df_years, df_countries


df, df_years, df_countries = read_data('climate_change.csv')


# Select the columns of interest for exploration of data
## The below is describing the statistical summary along with Correlation Matrix for selected indicators for 2016 till 2021.
cols = ['Country Name', 'Indicator Name', '2016', '2017', '2018', '2019', '2020', '2021']
df_selected = df[cols]

# Filter the data to include only selected indicators
indicators_of_interest = ['Population, total', 'GDP per capita (current US$)', 'CO2 emissions (kg per PPP $ of GDP)' , 'CO2 emissions from solid fuel consumption (% of total)', 'Renewable energy consumption (% of total final energy consumption)', 'Electric power consumption (kWh per capita)']
df_filtered = df_selected[df_selected['Indicator Name'].isin(indicators_of_interest)]

# Calculate the mean and standard deviation for each indicator for each country
summary_statistics = df_filtered.groupby(['Country Name', 'Indicator Name']).agg(['mean', 'median', 'min', 'max'])

# Print the summary statistics
print("Summary Statistics of Selected Indicators: ")
print(display(summary_statistics))

# Calculate the correlation matrix between the indicators
corr_matrix = df_filtered.pivot_table(index='Country Name', columns='Indicator Name')

print("Summary of Correlation Matrix: ")
# Print the correlation matrix
print(display(corr_matrix))


#### Exploration of data for finding correlation ####

#### Followings are the important points that are used when we check the correlation between two viariables: 
"""
A correlation coefficient of 1 indicates a perfect positive correlation --> as one variable increases, the other variable increases proportionally.
A correlation coefficient of -1 indicates a perfect negative correlation --> ne variable increases, the other variable decreases proportionally.
A correlation coefficient of 0 indicates --> no correlation between the variables.
A correlation coefficient between -1 and 0 indicates a negative correlation --> one variable increases, the other variable tends to decrease, but not proportionally.
A correlation coefficient between 0 and 1 indicates a positive correlation --> one variable increases, the other variable tends to increase, but not proportionally. 
"""

### Finding Correlation among all indicators ###
cols = ['Country Name', 'Indicator Name'] + [str(i) for i in range(2016, 2021, 6)]
df_6years = df[cols]

# Calculate correlation matrix
corr_matrix = df_6years.pivot_table(index='Country Name', columns='Indicator Name')
# Plot the correlation matrix
fig, ax = plt.subplots(figsize=(20, 20))
im = ax.imshow(corr_matrix.corr(), cmap='coolwarm')

# Add labels, title and colorbar, and show the plot
ax.set_xticks(np.arange(len(corr_matrix.columns)))
ax.set_yticks(np.arange(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=-90, ha='right', fontsize=17)
ax.set_yticklabels(corr_matrix.columns, fontsize=8)
ax.set_title('Correlation between Indicators', fontsize=20)
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom", fontsize=17)
plt.show()


# Selecting all indicators and finding their relationship with Population growth (annual %) indicator
cols = ['Country Name', 'Indicator Name'] + [str(i) for i in range(2020, 2021, 1)]
df_oneyear = df[cols]

# Get the indicator values for 'Population growth (annual %)'
pop_growth = df_oneyear[df_oneyear['Indicator Name'] == 'Population growth (annual %)'].set_index('Country Name')
pop_growth = pop_growth.rename(columns={'2020':'Population growth (annual %)'})

# Calculate correlation matrix
corr_matrix = df_oneyear.pivot_table(index='Country Name', columns='Indicator Name')

# Merge the population growth data with the correlation matrix
corr_pop = pd.merge(corr_matrix, pop_growth, left_index=True, right_index=True)

# Calculate correlations between each indicator and population growth
corr_with_pop_growth = corr_pop.corr()['Population growth (annual %)']

# Plot the correlation with population growth for each indicator
fig, ax = plt.subplots(figsize=(20, 20))
corr_with_pop_growth.drop('Population growth (annual %)').plot(kind='barh', ax=ax)
ax.set_title('Correlation with Population growth (annual %)', fontsize=14)
ax.set_xlabel('Correlation', fontsize=12)
ax.set_ylabel('Indicator', fontsize=17)
plt.show()



#################### Plots of different Indicators are created below  ########################

### 1st Plot ###

# # Select top 10 countries and indicators of interest i.e CO2 emissions (kt)
co2_emissions = df_years[df_years.index.get_level_values('Indicator Name') == 'CO2 emissions (kt)']
# Get top 10 countries with highest CO2 emissions in 2021
top10_countries = co2_emissions['2021'].sort_values(ascending=False).head(10).index.get_level_values('Country Name')
# Extract data for top 10 countries from 2014 to 2021
co2_top10 = co2_emissions.loc[(top10_countries, slice(None)), '2014':'2021']

# Now Plot the data with a line chart
co2_top10.T.plot(kind='line')
plt.title('CO2 Emissions for Top 10 Countries from 2014 to 2021')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (kt)')
plt.show()

#### 2nd Plot ###

# Select top 10 countries and indicators of interest i.e Access to Electricity (2018)
access_to_electricity = df[df['Indicator Name'] == 'Access to electricity (% of population)']
# Get top 10 Countries
top_countries = access_to_electricity.sort_values('2020', ascending=False)[:10]
# Set the index to 'Country Code'
top_countries = top_countries.set_index('Country Code')
# Select the columns with years 2016 to 2021
years = ['2016', '2017', '2018', '2019', '2020']
top_countries = top_countries[years]

# Plot a line chart
top_countries.T.plot(figsize=(10, 5))
plt.title('Top 10 Countries with Highest Access to Electricity (2016-2020)')
plt.xlabel('Year')
plt.ylabel('% of Population with Access to Electricity')
plt.legend(title='Country Code', loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()


#### 3rd Plot ####
# Filter the dataframe for Access to electricity indicator
access_to_electricity = df[df['Indicator Name'] == 'Access to electricity (% of population)']
# Group by country and calculate the mean across all years
mean_access = access_to_electricity.groupby('Country Name').mean()
# Select the top 10 countries with the highest mean access to electricity
top_countries = mean_access.nlargest(10, '2021')

# Transpose the dataframe and plot a line chart
top_countries.T.plot(figsize=(10, 5))

## Plot line chart
plt.title('Top Countries with Highest Access to Electricity (1960-2020)')
plt.xlabel('Year')
plt.ylabel('% of Population with Access to Electricity')
plt.legend(title='Country', loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()


### 4th Plot ####
# Select top 5 countries and indicators of interest i.e Total greenhouse gas emissions (kt of CO2 equivalent) (2019)
greenhouse_gas = df_years[df_years.index.get_level_values('Indicator Name') == 'Total greenhouse gas emissions (kt of CO2 equivalent)']
## Now Find out Top 5 Countries
top_countries = greenhouse_gas.sort_values('2019', ascending=False)[:5]

## Plot the bar chart particularly for 2019
ax = top_countries.plot(kind='bar', x='Country Code', y='2019', figsize=(10, 5))
plt.title('Top 5 Countries with Highest Total Greenhouse Gas Emissions in 2019')
plt.xlabel('Country')
plt.ylabel('% of Greenhouse Gas Emissions')
# Adding labels to the bars
for i, v in enumerate(top_countries['2019']):
    ax.text(i, v + 0.1, str(round(v, 1)), ha='center')
plt.show()



### 5th Plot ###
# Select top 5 countries and indicators of interest i.e Total greenhouse gas emissions (kt of CO2 equivalent)
greenhouse_gas = df_years[df_years.index.get_level_values('Indicator Name') == 'Total greenhouse gas emissions (kt of CO2 equivalent)']
## Find out Top 5 Countries
top_countries = greenhouse_gas.sort_values('2019', ascending=False)[:10]
top_countries = top_countries[['2016', '2017', '2018', '2019', '2020', '2021']]
top_countries = top_countries.T
top_countries.plot(kind='line', figsize=(10, 5))

# Define a Plot over data
plt.title('Top Countries with Highest Total greenhouse gas emissions')
plt.xlabel('Year')
plt.ylabel('Greenhouse Gas Emissions (kt of CO2 equivalent)')
plt.legend(title='Country', loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.show()


### 6th Plot ###
# Select top 5 countries and indicators of interest i.e CO2 emissions from liquid fuel consumption (kt)
co2_emissions_liquid = df_years[df_years.index.get_level_values('Indicator Name') == 'CO2 emissions from liquid fuel consumption (kt)']
co2_emissions_liquid = co2_emissions_liquid.dropna()
# Get Top Countries
top_5countries = co2_emissions_liquid.groupby('Country Code').mean().sort_values('2021', ascending=False)[:5]
top_5countries = top_5countries.T

# Plot the date on a line chart
top_5countries.plot(kind='line', figsize=(10, 5))
plt.title('Top 5 Countries with CO2 emissions from liquid fuel consumption')
plt.xlabel('Year')
plt.ylabel('CO2 emissions from liquid fuel consumption (kt)')
plt.legend(title='Country', loc='upper right', bbox_to_anchor=(1.0, 1.0))
plt.show()

### 7th Plot ###
# Filter the data and select the indicator i.e. electricity production from natural gas sources
co2_emissions = df_years[df_years.index.get_level_values('Indicator Name') == 'Electricity production from natural gas sources (% of total)']
# Get top 5 countries with highest CO2 emissions
top5_countries = co2_emissions['2021'].sort_values(ascending=False).head(5).index.get_level_values('Country Name')
# Extract data for top 5 countries from 2017 to 2021
co2_top5 = co2_emissions.loc[(top5_countries, slice(None)), '2017':'2021']

# Now Define Plot over data
co2_top5.T.plot(kind='line', figsize=(10, 5))
plt.title('Top 5 Countries with Highest Electricity Production from Natural Gas Sources by Years')
plt.xlabel('Year')
plt.ylabel('Electricity production from natural gas sources (% of total)')
plt.legend(title='Country', loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.show()

### 8th Plot ###
# Filter the data for electricity production from natural gas sources for creating a pie chart
co2_emissions = df_years[df_years.index.get_level_values('Indicator Name') == 'Electricity production from natural gas sources (% of total)']
# Get top 5 countries with highest electricity production from natural gas sources
top5_countries = co2_emissions['2021'].sort_values(ascending=False).head(5).index.get_level_values('Country Name')
# Extract data for top 5 countries from 2017 to 2021
co2_top5 = co2_emissions.loc[(top5_countries, slice(None)), '2017':'2021']
co2_top5_mean = co2_top5.mean(axis=1)

# Plot the data as a pie chart
plt.figure(figsize=(8, 8))
plt.pie(co2_top5_mean, labels=top5_countries, autopct='%1.1f%%', startangle=90, counterclock=False)
plt.title('Top 5 Countries with Highest Electricity Production from Natural Gas Sources', fontsize=14)
plt.axis('equal')
plt.legend(title='Country', loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.show()

### 9th Plot ###
## Filter the data for Renewable energy consumption (% of total final energy consumption)
renewable_energy = df[df['Indicator Name'] == 'Renewable energy consumption (% of total final energy consumption)']
world_data = renewable_energy[renewable_energy['Country Name'] == 'World']
world_data = world_data.transpose()
world_data = world_data[4:]
world_data.columns = ['Renewable Energy Consumption']
# Plot the data over a line chart
world_data.plot(kind='line', figsize=(10, 5))
plt.title('Global Renewable Energy Consumption (1960-2020)')
plt.xlabel('Year')
plt.ylabel('% of Total Final Energy Consumption')
plt.show()