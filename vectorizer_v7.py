import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_solution(dataframe):
    vec_tfidf = TfidfVectorizer(stop_words='english', lowercase=True, token_pattern=r"(?u)\b\w[\w-]*\w\b", min_df=0.1)
    tfidf_mat = vec_tfidf.fit_transform(dataframe['solution_column'])
    feature_names = vec_tfidf.get_feature_names()
    dense = tfidf_mat.todense()
    denselist = dense.tolist()
    df_euclidean = pd.DataFrame(denselist, columns = feature_names)
    return tfidf_mat, df_euclidean