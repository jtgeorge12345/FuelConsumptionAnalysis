#main.py

from FuelConsumptionAnalysis import *
from cluster_analysis import *

#data = importData()
# printDescriptiveStats(data)
#print("===========================")
#makeScatterMatrix(data)
#cylindarHistogram(data)
#pivot(data, "CYLINDERS", "TRANSMISSION")


#################################### Classification ##########################
# data, response = prepDataForClassifiers(data)
#
# svcLinearResults(data, response)
# svcGammaResults(data, response)
#
# clfs = getClassifiers()
# for key in clfs.keys():
#
#     print("trying:", key)
#
#     try:
#         jg_cross_validate_metrics(data, response, clfs[key], shortname=key)
#         print(key, "succeeded")
#     except:
#         print(key,"failed")

#################################### Clustering ##########################


data = importData()
cluster_prepped_data = prepDataForClustering(data)

jg_cluster(cluster_prepped_data)
