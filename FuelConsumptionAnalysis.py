#Fuel Consumption Analysis
#Data Source: https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64

import pandas as pd
import csv
import matplotlib
print(matplotlib.get_backend())
matplotlib.use('Agg')
print(matplotlib.get_backend())

pd.set_option("display.max_columns", 13)

#################### read in CSV file#############################
def importData():
    data = pd.read_csv("Canada Fuel Consumption Ratings 2000-2014 2.csv")
    data.dropna(axis=1, inplace=True, thresh=2000)
    return(data)

data = importData()
header = data.columns
print("All Columns:", header)
print(data.describe())
""" Observations: There are about 14,000 cars in this dataset
Average combined MPG is 27.3, high is 78, low is 7.5.Mean year is 2007.6,
which implies slightly more newer cars than older cars (7 would be expected
given even distribution of 2000 through 2014). Looking at engine size, cylinders,
all fuel consumptions, and c02 emissions, there appear to be some outliers on
both the high and low ends, but the majority of cars are grouped fairly
tightly around the averages. It would be interesting to see histograms of each
distribution for more detail, so that's what we'll do."""


""" Before histograms, I'm curious as to what car has 16 cylinders"""

interestingPoints = data.loc[data['CYLINDERS'] == 16]
interestingPoints.assign(Comment="16 Cylinders!")
#Exporting some interesting observations to a csv file
interestingPoints.to_csv("InterestingPoints.csv")
"""It turns out the 2010-2012 Bugatti Veyron had 16 cylindars. I wouldn't be
surprised to see this car show up as an outlier later on in this analysis"""


"""Histograms"""
data.hist()
############# Note: This is not working with Linux Subsystem for Windows.
# Implement fix noted here:https://www.reddit.com/r/Python/comments/595j6v/matplotlib_in_the_new_linux_subsystem_of_windows/
# Failing that, switch over to windows and resolve dependency issues.


#Put data into a Pandas Dataframe

#Use learning techniques to predict fuel consumption based on other attributes

#Cluster analysis and create plots of clusters

#compare clusters to actual car categories

#Which cars improved the most over 15 years?

#Which cars are improving the most consistently?

#Which cars are getting worse?

#Can cost data be added to this dataset?
