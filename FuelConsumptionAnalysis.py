#Fuel Consumption Analysis
#Data Source: https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64

import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sklearn
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn_util import *


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


def printDescriptiveStats(data):
    header = data.columns
    print("All Columns:", header)
    print(data.describe())
    print(data.dtypes)
    categoricals = data.select_dtypes(include="category")
    for item in categoricals:
        print("Categorial Attribute:", item)
        print(data[item].describe())



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
def YearsHistogram(data):
    histogram = data.hist(column="YEAR", bins=15)
    plt.savefig('YearsHistogram')
    histogram = data.hist(column="CYLINDERS", bins=16)
    plt.savefig('CylindersHistogram')

"""A low-effort way of seeing correlations and seeing kde's equivalent to the
above histograms would be to make a scatter matrix: """
def makeScatterMatrix(data, name=""):
    scatter_matrix = pd.plotting.scatter_matrix(data, diagonal='kde')
    for ax in scatter_matrix.ravel():
        ax.set_xlabel(ax.get_xlabel(), fontsize = 5, rotation = 90)
        ax.set_ylabel(ax.get_ylabel(), fontsize = 5, rotation = 0)
    plt.savefig("scatter_matrix"+name)
    print("Scatter Matrix Saved" + name)

""" From this we can see that there are some pretty obvious correlations
between fuel consumption, cylindars, engine size, and CO2 emissions. Years
don't appear to correlate with anything in particular in this view, but it will
be interesting to see if a closer look will reveal correlations. My expectation
is that in general, CO2 emissions should fall over time when we look more
specifically at certain models"""

def cylindarHistogram(data):
    print(data['CYLINDERS'].value_counts())

""" There are 16 models with 2 cylinders, 28 with 3, 481 with 5, a few with 10,
12, and 6, but most have either 4, 6, or 8. Turns out there was nothing wrong
with the histogram, I just  didn't know you could have cars with odd numbers of
cylinders. Does having an odd number of cylinders have a disproportionate effect
 on fuel consumption? May be a question for later. Here's a look at cylindar
 count by make:"""


""" Linear Regression"""
def linReg(data):
    #Todo: Need to re-do this with categorical variables encoded correctly
    reducedData = data[["TRANSMISSION", "FUEL", "CYLINDERS", "ENGINE SIZE", "COMB (mpg)", "CO2 EMISSIONS"]]

    #print(reducedData.head())
    #print(data["TRANSMISSION"].value_counts())

    #These statements produce warnings, however the columns are coded as intended
    #so the warnings can be ignored.
    reducedData["TRANSMISSION"], trans_map = reducedData["TRANSMISSION"].factorize()
    reducedData["FUEL"], fuel_map = reducedData["FUEL"].factorize()

    #makeScatterMatrix(reducedData, "_Reduced")

    regressionModel = sklearn.linear_model.LinearRegression()
    print(regressionModel)
    response = reducedData.pop("CO2 EMISSIONS")

    regressionModel.fit(reducedData[:10000], response[:10000])

    print("coefficients:", regressionModel.coef_)
    print("intercepts:", regressionModel.intercept_)

    score = regressionModel.score(reducedData[10000:], response[10000:])

    print("score:", score)


    predictions = regressionModel.predict(reducedData[10000:10010])

    """Checking a few of the results. They don't look too far off. However, athough
    I tried to eliminate obviously correlated variables, it's hard to ensure there's
    no multicolinearity problem without examining variance inflation factors, which
    sckit does not easily support at this time. Also, though my R squared value came
    out to .836, it would be better to look at R squared-adusted. There are also
    problems with how the categorical variables are coded, which I will not get into here
    """

    """ Need to resove the issues with how variables are coded. The numbers assigned
    are being fed into the linear regression as being sequential, which is not relevant.
    There should be additional columns, 1 for each variable."""

    for i in range(len(predictions)):
        print("predicted:", predictions[i], "Actual:", response[10000 + i] )



def prepDataForClassifiers(data):
    response = data["VEHICLE CLASS"]
    data = data.drop(columns=["MAKE","MODEL", "VEHICLE CLASS"])
    prepped_data = pd.get_dummies(data)
    return(prepped_data, response)


def crossValidate(data, response, clf):

    result = cross_val_predict(clf, data, response, cv=5)
    labels = response.unique()
    cm = confusion_matrix(response, result, labels=labels)

    print(response.value_counts())
    for line in cm:
        print(list(line))

    plot_confusion_matrix(cm, labels, normalize=True, filename="KNN")

def pivot(data, c1, c2):

    table = pd.pivot_table(data, columns=[c1], fill_value=0, index=[c2], aggfunc="count")
    print(table)


#Put data into a Pandas Dataframe

#Use learning techniques to predict fuel consumption based on other attributes

#Cluster analysis and create plots of clusters

#compare clusters to actual car categories

#Which cars improved the most over 15 years?

#Which cars are improving the most consistently?

#Which cars are getting worse?

#Can cost data be added to this dataset?
