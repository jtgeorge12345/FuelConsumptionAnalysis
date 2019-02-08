#Fuel Consumption Analysis
#Data Source: https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64

import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

pd.set_option("display.max_columns", 13)

#################### read in CSV file#############################
def importData():
    data = pd.read_csv("Canada Fuel Consumption Ratings 2000-2014 2.csv") #TODO: Specify data types here
    data.dropna(axis=1, inplace=True, thresh=2000)
    data['VEHICLE CLASS'] = data['VEHICLE CLASS'].astype("category")
    data['MAKE'] = data['MAKE'].astype("category")
    data['MODEL'] = data['MODEL'].astype("category")
    data['TRANSMISSION'] = data['TRANSMISSION'].astype("category")
    data['FUEL'] = data['FUEL'].astype("category")


    return(data)

data = importData()

def printDescriptiveStats(data):
    header = data.columns
    print("All Columns:", header)
    print(data.describe())
    print(data.dtypes)
    categoricals = data.select_dtypes(include="category")
    for item in categoricals:
        print("Categorial Attribute:", item)
        print(data[item].describe())

printDescriptiveStats(data)
print("===========================")

""" Observations: There are about 14,000 cars in this dataset
Average combined MPG is 27.3, high is 78, low is 7.5.Mean year is 2007.6,
which implies slightly more newer cars than older cars (7 would be expected
given even distribution of 2000 through 2014). Looking at engine size, cylinders,
all fuel consumptions, and c02 emissions, there appear to be some outliers on
both the high and low ends, but the majority of cars are grouped fairly
tightly around the averages. It would be interesting to see histograms of each
distribution for more detail, so that's what we'll do."""

#How many makes do we have?

""" Before histograms, I'm curious as to what car has 16 cylinders"""

# interestingPoints = data.loc[data['CYLINDERS'] == 16]
# interestingPoints.assign(Comment="16 Cylinders!")
# """Exporting some interesting observations to a csv file"""
# interestingPoints.to_csv("InterestingPoints.csv")

"""It turns out the 2010-2012 Bugatti Veyron had 16 cylindars. I wouldn't be
surprised to see this car show up as an outlier later on in this analysis"""

"""Histograms"""
def makeHistograms(data):
    histogram = data.hist()
    plt.savefig('BasicHistogram')

# makeHistograms(data)
""" "Year" histogram is distorted by the bin sizes, as is "CYLINDERS" """
# histogram = data.hist(column="YEAR", bins=15)
# plt.savefig('YearsHistogram')
# histogram = data.hist(column="CYLINDERS", bins=16)
# plt.savefig('CylindersHistogram')

"""A low-effort way of seeing correlations and seeing kde's equivalent to the
above histograms would be to make a scatter matrix: """
def makeScatterMatrix(data):
    scatter_matrix = pd.plotting.scatter_matrix(data, diagonal='kde')
    for ax in scatter_matrix.ravel():
        ax.set_xlabel(ax.get_xlabel(), fontsize = 5, rotation = 90)
        ax.set_ylabel(ax.get_ylabel(), fontsize = 5, rotation = 0)
    plt.savefig("scatter_matrix")
    print("Scatter Matrix Saved")
#makeScatterMatrix(data)

""" From this we can see that there are some pretty obvious correlations
between fuel consumption, cylindars, engine size, and CO2 emissions. Years
don't appear to correlate with anything in particular in this view, but it will
be interesting to see if a closer look will reveal correlations. My expectation
is that in general, CO2 emissions should fall over time when we look more
specifically at certain models"""




#print(data.loc[data["CYLINDERS"] == 5].groupby("MAKE").count()["MODEL"])
#print data.loc[data['CYLINDERS'] == 5]

# print (data.groupby(["CYLINDERS", "MAKE"]).count())
""" There are 16 models with 2 cylinders, 28 with 3, 481 with 5, a few with 10,
12, and 6, but most have either 4, 6, or 8. Turns out there was nothing wrong
with the histogram, I just  didn't know you could have cars with odd numbers of
cylinders. Does having an odd number of cylinders have a disproportionate effect
 on fuel consumption? May be a question for later. Here's a look at cylindar
 count by make:"""

table = pd.pivot_table(data, index = ["MAKE"], columns = "CYLINDERS", aggfunc = len)
#print(table)
"""     """

#Put data into a Pandas Dataframe

#Use learning techniques to predict fuel consumption based on other attributes

#Cluster analysis and create plots of clusters

#compare clusters to actual car categories

#Which cars improved the most over 15 years?

#Which cars are improving the most consistently?

#Which cars are getting worse?

#Can cost data be added to this dataset?
