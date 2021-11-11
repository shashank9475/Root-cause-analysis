import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from vectorizer_v7 import tfidf_solution
from custom_extractor_v7 import custom_main
from sklearn.preprocessing import normalize
from find_clusterNumber import knee
from Yake_v7 import Yake_main
 
    
def kmeans_clustering(tfidf_matrix, noOfClusters):
    
    kmeans = KMeans(n_clusters = noOfClusters, init='k-means++', max_iter=500, n_init=25)
    model = kmeans.fit(tfidf_matrix)     # training on all documents except last one(will be used for testing)
    predict = kmeans.predict(tfidf_matrix)
    #predict = kmeans.fit_predict(tfidf_matrix)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
  
    return model, labels, predict

def plot(tfidf_matrix, predict):
    # PCA reduction
    sklearn_pca = PCA(n_components = 2)
    Y_sklearn = sklearn_pca.fit_transform(tfidf_matrix)
    
    #2D Plot
    plt.figure()
    myPlot = plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=predict, s=50, cmap='viridis')
    plt.colorbar(myPlot)
    plt.title('K-means Clustering plot in 2D')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    
def plot2(tfidf_matrix, predict):
    # PCA reduction
    sklearn_pca = PCA(n_components = 2)
    Y_sklearn = sklearn_pca.fit_transform(tfidf_matrix)
    
    #2D Plot
    plt.figure()
    myPlot = plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], s=50, cmap='viridis')
    plt.colorbar(myPlot)
    plt.title('All datapoints in 2 dimensions')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    #plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6) 
    


def cluster_solution(df, df_withoutClean):
    tfidf_mat, tfidf_pipe_matrix_euclidean = tfidf_solution(df)

    X_Norm = normalize(tfidf_pipe_matrix_euclidean)
    noOfClusters = knee(X_Norm)
    
    model, label, y_kmeans = kmeans_clustering(X_Norm, noOfClusters)

    #to add cluster column in df1 dataframe    
    df["cluster_solution"] = pd.Series(y_kmeans, index=df.index)
    df_withoutClean["cluster_solution"] = pd.Series(y_kmeans, index=df_withoutClean.index)
    
    
    # list of clusters
    list = []
    for i in range(noOfClusters):
        #list.append(df1[(df1['cluster_solution'] == i)])
        list.append(df_withoutClean[(df_withoutClean['cluster_solution'] == i)])
    
    list_yake = []
    for i in list:
        tex = Yake_main(i)
        list_yake.append(tex)
        
    return list, df, df_withoutClean, list_yake
