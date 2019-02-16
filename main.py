#main.py

from FuelConsumptionAnalysis import *
data = importData()
printDescriptiveStats(data)
print("===========================")
#makeScatterMatrix(data)
#cylindarHistogram(data)
pivot(data, "CYLINDERS", "TRANSMISSION")

# data, response = prepDataForClassifiers(data)
# knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=2, algorithm="auto")
# crossValidate(data, response, knn)
