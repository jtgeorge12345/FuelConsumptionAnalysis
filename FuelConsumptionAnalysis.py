#Fuel Consumption Analysis
#Data Source: https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64

import pandas as pd
pd.set_option("display.max_columns", 13)

# read in CSV file
data = pd.read_csv("Canada Fuel Consumption Ratings 2000-2014 2.csv")
print(data.head())

data.dropna(axis=1, inplace=True, thresh=2000)
print(data.head())

print(data.describe())

print(data.loc[data['CYLINDERS'] == 16])

#Put data into a Pandas Dataframe

#Use learning techniques to predict fuel consumption based on other attributes

#Cluster analysis and create plots of clusters

#compare clusters to actual car categories

#Which cars improved the most over 15 years?

#Which cars are improving the most consistently?

#Which cars are getting worse?

#Can cost data be added to this dataset?
