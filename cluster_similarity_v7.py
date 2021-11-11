import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from kmean_v7 import cluster_solution
from Yake_v7 import Yake_main
#from similarity_testing import bert_similarity
#from sentence_transformers import SentenceTransformer
import operator


def normalize(d, target=1.0):
   raw = sum(d.values())
   factor = target/raw
   normalized_avg = {key:value*factor for key,value in d.items()}
   return normalized_avg

# calculating average score for each cluster. 
def cluster_avg_score(cluster_list, text, model):
    lis = {}
    for i in range(len(cluster_list)):
        final_list = similarity(cluster_list[i], text, model)
        avg_score = sum(final_list) / len(final_list)
        lis[i] = avg_score
    return lis


# calculating the avg score for each cluster and then normalized the avg list and populating only those clusters whose 
#normalized avg score is greater than average of normalized average list.
def similar_cluster(df, cluster_list, text, model):
    lis = cluster_avg_score(cluster_list, text, model)
    print("Average score of each cluster: ", lis)
    normalized_avg = normalize(lis)
    print("Normalized score of each cluster: ", normalized_avg)
    sort_items = sorted(normalized_avg.items(), key=lambda x: x[1], reverse=True)
    filtered_clusterList = []
    list_yake = []
    list_cluster = []
    for i in sort_items:
        if i[1] > (1.0 / len(normalized_avg)):
            cluster = cluster_list[i[0]]["cluster_solution"].unique()
            filtered_clusterList.append(cluster)
            list_cluster.append(cluster_list[i[0]])
            tex = Yake_main(cluster_list[i[0]])
            list_yake.append(tex)

    #Now since we got the similar cluster, we now check which other individual datasample having higher similarity than all the 
    #cluster similarity and also doesn't belong to any of the similar clusters filtered above.
    print("filtered_clusterList: ", filtered_clusterList)
    dic = individual_similarity(df, cluster_list, text, model)
    individual_datasample = []
    list_yake_indi = []
    for k, v in dic.items():
        print(df[["Index", "cluster_solution"]].iloc[k])
        if (df["cluster_solution"].iloc[k] not in filtered_clusterList):
            print("inside")
            print(df[["Index","cluster_solution"]].iloc[k])
            dataframe = pd.DataFrame(df.iloc[k,:])
            yak = Yake_main(dataframe.T)
            individual_datasample.append(dataframe.T)
            list_yake_indi.append(yak)
    return list_cluster, list_yake, individual_datasample, list_yake_indi
    

# checking similarity of each datasample for both failure type and tech explanation columns and taking the highest of the two and 
#then returning the final list containing similarity score of each datasample with the inputed text.
def similarity(df_withoutClean, text, model):
    li_typeFailure = bert_similarity_individual(df_withoutClean["Type of failure"], text, model)
    li_techExplain = bert_similarity_individual(df_withoutClean["Technical explanation"], text, model)
    print("similarity based on type of failure: ", li_typeFailure)
    print("similarity based on technical explanation: ", li_techExplain)
    
    final_list = []
    for i in range(len(li_typeFailure)):
        if li_typeFailure[i] > li_techExplain[i]:
            final_list.append(li_typeFailure[i])
        else:
            final_list.append(li_techExplain[i])
    return final_list

    

def bert_similarity_individual(df, text, model):
    #li = []
    sentences = list(df.values)
    sentences.append(text)
    #print(sentences)

    
    sentence_embeddings = model.encode(sentences)
    sentence_embeddings.shape
    
    li= cosine_similarity([sentence_embeddings[-1]],sentence_embeddings[:-1])
    return li[0]


def individual_similarity(df, cluster_list, text, model):   
    individualScore_list = similarity(df, text, model)
    print("Individual score: ", individualScore_list)
    individualScore_dic = {}
    for i in range(len(individualScore_list)):
        individualScore_dic[i] = individualScore_list[i]
    avgScore_dic = cluster_avg_score(cluster_list, text, model)
    max_value = max(avgScore_dic.items(), key=operator.itemgetter(1))[1]
    dic = {}
    for key1, value1 in individualScore_dic.items():
        #for key, value in avgScore_dic.items():
        if max_value < individualScore_dic.get(key1):
            #print(df["cluster_solution"].iloc[key1])
            #print(df["cluster_solution"].iloc[key])
            dic[key1] = value1
    return dic


def cluster_similarity_main(text, model):
    
    df = pd.read_pickle('dataset/pickle/dataset.pkl')
    df_withoutClean = pd.read_pickle('dataset/pickle/dataset_without_cleaning.pkl')
    cluster_list, df1, df1_withoutClean, yake = cluster_solution(df, df_withoutClean)
    
    list_cluster, list_yake, individual_datasample, list_yake_indi = similar_cluster(df, cluster_list, text, model)

    return list_cluster, list_yake, individual_datasample, list_yake_indi
