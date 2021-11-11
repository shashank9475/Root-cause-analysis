import numpy as np
import pandas as pd
import string
import re
import os
import enchant
import pathlib
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def read_csv(path):
    df = pd.read_excel(path, engine='openpyxl')
    df = df.replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'_x000D_\n',  ' ', regex=True)
    df = df.replace(r'\n', ' ', regex=True)  
    return df


def remove_NaN(df):
    cols = ["To be measured or simulated","Schematic modification","Layout modification","Modification of test program","Product specification modification","Measures against recurrence","Analog","Digital","Test","Layout","Software","Quality","Spec","Additional Information","Others"]
    df[cols] = df[cols].replace(np.nan, "-")
    df[cols] = df[cols].replace("n.a.", "-")
    df[cols] = df[cols].replace("n/a", "-")
    df[cols] = df[cols].replace("N.A.", "-")
    
    df['combined'] = df["Target version"] + " " + df["Circ status"] + " " + df["Block"] + " " + df['Type of failure'] + " " + df['Technical explanation']+ " " + df['Modification concept']+ " " + df['To be measured or simulated']+ " " + df['Schematic modification']+ " " + df['Layout modification']+ " " + df['Modification of test program']+ " " + df['Product specification modification']+ " " + df['Measures against recurrence'] + " " + df["Analog"] + " " + df["Digital"] + " " + df["Test"] + " " + df["Layout"] + " " + df["Software"] + " " + df["Quality"] + " " +df["Spec"] + " " +df["Additional Information"] + " " + df["Others"]
    return df


def detect_language(text):
    text = text.strip()
    
    #removing all the punctuations
    for punctuation in string.punctuation:
        text = text.replace(punctuation, " ")
    
    #removing all words containing digits
    text = ' '.join(s for s in text.split() if not any(c.isdigit() for c in s))
    text = re.sub(r'(?:^| )\w(?:$| )', ' ', text).strip()
    text = re.sub(r'(?:^| )\w(?:$| )', ' ', text).strip()
    
    dictionary_en = enchant.Dict("en_US")
    words = text.split(" ")
    en_count = 0.0
    for word in words:
        if word.strip():
            if dictionary_en.check(word.strip()):
                en_count += 1
    percent = en_count*100/len(words)
    #percent = "EN: " + str(perc) + "%" if len(words) != 0 else 0
    return percent

def document_print(text):
    print(text)
    print("\n----------------------------------------------\n")
    
    
def document_filter(df):
    df_filtered = df.loc[lambda x: x['Language'] >= 60.0]
    return df_filtered


def data_preprocess_lessRelevantColumns(text):
    if type(text)=='str':
        text = text
    else:
        text = str(text)

    text = text.strip()
    
    # removing texts which are inside braces.
    text = re.sub("[\(\[].*?[\)\]]", "", text)
    
    #remove punctuation except '-' , '.' , '>' , '_'
    punctuations = ['!','"','#','$','%','&','(',')','*','+',',','/',':',';','<','=','?','@','[',']','^','`','{','|','}','~',"'"]
    for punctuation in punctuations:
        text = text.replace(punctuation, " ")

    #remove stop words
    text = ' '.join([word for word in text.split() if word.lower() not in (ENGLISH_STOP_WORDS)])

    #lemmitize
    lem = WordNetLemmatizer()
    text = ' '.join([lem.lemmatize(word) for word in text.split()])
    #spell-check and correct
    text = ' '.join(i for i in text.split(' ') if len(i) > 2)
    
    return text


def data_preprocess(text):
    if type(text)=='str':
        text = text
    else:
        text = str(text)

    text = text.strip()
    
    # removing texts which are inside braces.
    text = re.sub("[\(\[].*?[\)\]]", "", text)
    
    #remove punctuation except '-','_'
    punctuations = ['!','"','#','$','%','&','(',')','*','+',',','.','/',':',';','<','=','>','?','@','[',']','^','`','{','|','}','~',"'"]
    for punctuation in punctuations:
        text = text.replace(punctuation, " ")

    #remove stop words
    custom_stopwords = ['just','try','neuer','einbau','inhalt','mus','keine','machen','el','besser','übers','hätten','da','werden','müssen','genauer','wird','soll','nicht','im','ok',  ]
    text = ' '.join([word for word in text.split() if word.lower() not in (ENGLISH_STOP_WORDS)])
    german_stop_words = stopwords.words('german')
    for i in custom_stopwords:
        german_stop_words.append(i)
    text = ' '.join([word for word in text.split() if word.lower() not in (german_stop_words)])
   
    #lemmitize
    lem = WordNetLemmatizer()
    text = ' '.join([lem.lemmatize(word) for word in text.split()])
    #spell-check and correct
    
    #removing continuous punctuations
    text = text.replace('-' * 2, '')
    
    #removing words containing less than 3 characters
    text = ' '.join(i for i in text.split(' ') if len(i) > 2)# or (re.search(pattern, i) and len(i) == 2))
    
    #removing all words containing digits
    text = ' '.join([i for i in re.sub(r'[.,!?]', '', text).split() if not re.search(r'\d', i)])

    text = text.strip()
    return text

def data_cleaninig(excel_folder, pickle_folder):
    try:
        list_of_files = list(pathlib.Path(excel_folder).glob('*.xlsx'))
        path = max(list_of_files, key=os.path.getctime)
        df = read_csv(path)
        df = remove_NaN(df)
        print(df["Index"])
        #for checking and filtering based on language percentage and then dropping the language column
        df["Language"] = df["combined"].apply(detect_language)
        df_filtered = document_filter(df)
        df_filtered.drop(df_filtered.columns[[23,24]], axis = 1, inplace = True)
        df_filtered.to_pickle(os.path.join(pickle_folder, 'dataset_without_cleaning.pkl'))
        
        #data preprocess and cleaning
        cleaning1_columns = ["Type of failure","Technical explanation","Modification concept","To be measured or simulated","Schematic modification","Layout modification","Modification of test program","Product specification modification","Measures against recurrence","Analog","Digital","Additional Information"]
        cleaning2_columns = ["Block","Test","Layout","Software","Quality","Spec"]
        
        for index in cleaning2_columns:
            df_filtered[index] = df_filtered[index].apply(data_preprocess_lessRelevantColumns)
        
        # data cleaning for relevant columns
        for i in cleaning1_columns:
            df_filtered[i] = df_filtered[i].apply(data_preprocess)
            
        df_filtered['solution_column'] = df_filtered['Modification concept']+ " " + df_filtered['To be measured or simulated'] + " " + df_filtered['Schematic modification'] + " " + df_filtered['Layout modification']+ " " + df_filtered['Modification of test program'] + " " + df_filtered['Product specification modification'] + " " + df_filtered['Measures against recurrence']
        df_filtered.to_pickle(os.path.join(pickle_folder, 'dataset.pkl'))
        return df_filtered
    
    except IndexError:
        raise IOError("No .xlsx files found in %r" % excel_folder)


def main():  
    excel_folder = "E:/Thesis/codes/final_codes/code_v7/UI/dataset/excel/"
    pickle_folder = "E:/Thesis/codes/final_codes/code_v7/UI/dataset/pickle/"
    df = data_cleaninig(excel_folder, pickle_folder)
    print(df.columns)

    
    
    
if __name__ == "__main__":
    main()