import numpy as np
import pandas as pd
import scipy.sparse as sparse
from datetime import datetime
from pandas import DataFrame
from pandas import Series
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import  warnings



def k_neighbors(mtx, k, number):

    warnings.filterwarnings("ignore")

    # Using PCA for dimensionality reduction
    start = datetime.now()
    pca = PCA(copy=True)
    pca_mtx = pca.fit_transform(mtx)
    end = datetime.now()



    # Build the KNN model
    start = datetime.now()
    neigh = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', metric="cosine", n_jobs=1)
    neigh.fit(pca_mtx)
    end = datetime.now()

    # Obtain the k nearst neighbors for each use
    start = datetime.now()
    distance, neighbor = neigh.kneighbors(pca_mtx)

    new_neighbor = []
    for i in range(len(neighbor)):
        temp = []
        find = 0
        count = 1
        for j in range(len(neighbor[i])):
            if i != neighbor[i][j]:
                temp.append(neighbor[i][j])
                count += 1
            else:
                find = 1
            if find == 0 and count == len(neighbor[i]):
                break
        #print(i,temp)
        new_neighbor.append(temp)
    return new_neighbor

                

