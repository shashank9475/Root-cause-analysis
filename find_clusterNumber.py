from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from kneed import KneeLocator
import math


from vectorizer_v7 import tfidf_solution


def kmeans(X_Norm, upperLimit):
    
    km_scores= []
    km_silhouette = []
    for i in range(2,upperLimit):
        km = KMeans(n_clusters=i, random_state=1, init='k-means++', max_iter=100, n_init=25)
        preds = km.fit_predict(X_Norm)
        
        #print("Score for number of cluster(s) {}: {}".format(i,km.score(df_scale)))
        km_scores.append(-km.score(X_Norm))
        
        silhouette = silhouette_score(X_Norm, preds)
        km_silhouette.append(silhouette)
        print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))
    
    return km_scores, km_silhouette

def Elbow_knee(km_scores, upperLimit):
    
    plt.figure()
    plt.title("The elbow method for determining number of clusters\n",fontsize=16)
    plt.scatter(x=[i for i in range(2,upperLimit)],y=km_scores,s=150,edgecolor='k')
    plt.grid(True)
    plt.xlabel("Number of clusters",fontsize=14)
    plt.ylabel("K-means score",fontsize=15)
    #plt.xticks([i for i in range(2,30)],fontsize=14)
    plt.yticks(fontsize=15)
    plt.show()
    
def silhouette_plot(km_silhouette, upperLimit):
    plt.figure(figsize=(7,4))
    plt.title("Silhouette coefficient plot",fontsize=16)
    plt.scatter(x=[i for i in range(2,upperLimit)],y=km_silhouette,s=150,edgecolor='k')
    plt.grid(True)
    plt.xlabel("Number of clusters",fontsize=14)
    plt.ylabel("Silhouette score",fontsize=15)
    #plt.xticks([i for i in range(2,30)],fontsize=14)
    plt.yticks(fontsize=15)
    plt.show()


def knee(X_Norm):
    sum_squared_dist = []
    upperLimit = math.floor(math.sqrt(len(X_Norm)))
    #print(upperLimit)
    if len(X_Norm) < 10:
        sum_squared_dist = []
        for i in range(1, upperLimit+1):
            kmeans = KMeans(n_clusters=i, random_state=1, init='k-means++', max_iter=100, n_init=25)
            preds = kmeans.fit_predict(X_Norm)
            sum_squared_dist.append(kmeans.inertia_)
            
    else:
        sum_squared_dist = []
        for i in range(1, upperLimit+5):
            kmeans = KMeans(n_clusters=i, random_state=1, init='k-means++', max_iter=100, n_init=25)
            preds = kmeans.fit_predict(X_Norm)
            sum_squared_dist.append(kmeans.inertia_)
            
    #print(sum_squared_dist)

    
    x = range(1, len(sum_squared_dist)+1)
    
    kn = KneeLocator(x, sum_squared_dist, curve='convex', direction='decreasing')
    #print(kn.knee)
    
    #plt.xlabel('number of clusters k')
    #plt.ylabel('Sum of squared distances')
    #plt.plot(x, sum_squared_dist, 'bx-')
    #plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
       
    return kn.knee
   
def main():
    df1 = pd.read_pickle('dataset/pickle/dataset.pkl')

    tfidf_mat, tfidf_pipe_matrix_euclidean = tfidf_solution(df1)

    X_Norm = normalize(tfidf_pipe_matrix_euclidean)
    upperLimit = math.floor(math.sqrt(len(X_Norm)))
    km_scores, km_silhouette = kmeans(X_Norm, upperLimit)
    silhouette_plot(km_silhouette, upperLimit)
    #Elbow_knee(km_scores, upperLimit)
    #silhouette(km_silhouette, upperLimit)
    li = knee(X_Norm)
    
if __name__ == "__main__":
    main()