#sklearn regression

"""
As I was doing this I learned that sklearn does not have the proper tools for
multivariable regression, ANOVA, VIF, P-values, etc, so this analysis should
really be done in R, which is not in my scope at the moment. There is also a
multicolinearity problem in this, categorical data is not handled correctly,
and this is generally a bad analysis all around.

The only reason I'm keeping it is to shame myself into re-doing it in R later
"""


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
