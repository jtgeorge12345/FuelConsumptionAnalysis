#main.py

from FuelConsumptionAnalysis import *
data = importData()
printDescriptiveStats(data)
print("===========================")
#makeScatterMatrix(data)
#cylindarHistogram(data)
#pivot(data, "CYLINDERS", "TRANSMISSION")

data, response = prepDataForClassifiers(data)
knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=2, algorithm="auto")
# crossValidate(data, response, knn)

#jg_cross_validate_metrics(data, response, knn, shortname="knn")

clfs = getClassifiers()

svcLinearResults(data, response)
svcGammaResults(data, response)


for key in clfs.keys():

    print("trying:", key)

    try:
        jg_cross_validate_metrics(data, response, clfs[key], shortname=key)
        print(key, "succeeded")
    except:
        print(key,"failed")
