#cluster_analysis.py

from sklearn.cluster import KMeans
import pandas as pd
import numpy
import matplotlib.pyplot as plt

def prepDataForClustering(data):
    data = data.drop(columns=["MAKE","MODEL", "VEHICLE CLASS"])
    prepped_data = pd.get_dummies(data)
    return(prepped_data)


def jg_cluster(data):

    print("prepped for clustering:", data.columns)


    data_norm = (data - data.mean()) / (data.max() - data.min())
    data = data_norm
    kmeans = KMeans(n_clusters=10, random_state=0).fit(data)

    print(kmeans.cluster_centers_)
    print(type(kmeans.labels_))


    fig = plt.figure()
    axes = fig.add_axes()
    ax1 = fig.add_subplot(1,1,1)
    ax1.scatter(data["ENGINE SIZE"], data["CYLINDERS"], c=kmeans.labels_)
    plt.show()
