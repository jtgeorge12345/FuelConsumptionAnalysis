#sklearn_util.py
"""
This function borrowed from Scikit-Learn documentation:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
"""

"""
Another great method of comparing classifiers - Not implemented in this project
but may be in the future. The challenge with this dataset is that there are more
than 2 labels and many more than 2 dimensions"
"""

import matplotlib.pyplot as plt
import numpy as np
import itertools
from matplotlib.pyplot import figure

def plot_confusion_matrix(cm, classes, filename="",
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # """Note: [JG] Modified code to create both normalized and non-normalized plots"""
    # if (normalize == False):
    #     plot_confusion_matrix(cm, classes, normalize=True, filename=filename+"_normalized")


    """Commenting out print statements"""
    # # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')
    #
    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("./classifier_plots/CM_" + filename)
    plt.close()
