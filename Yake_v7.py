import yake
import re
import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag
import itertools
import spacy
nlp = spacy.load('en_core_web_sm')
from yake_extension_v7 import custom_main
#from RAKE_v5 import RAKE_extraction
#nltk.download('averaged_perceptron_tagger')


# Let's create a function to pull out nouns and adjectives from a string of text
def nouns_adj_verb(text):
    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
    #is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_adj_verb = [word for (word, pos) in pos_tag(tokenized) if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS' or pos == 'JJ' or pos == 'VBN' or pos == 'VBD' or pos == 'VBG')] 
    return ' '.join(nouns_adj_verb)

def nouns_adj(text):
    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
    #is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS' or pos == 'JJ')] 
    return ' '.join(nouns_adj)


def nouns_verb(text):
    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
    #is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_verb = [word for (word, pos) in pos_tag(tokenized) if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS' or pos == 'VBN' or pos == 'VBD' or pos == 'VBG')] 
    return ' '.join(nouns_verb)


def nouns(text):
    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
    #is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns = [word for (word, pos) in pos_tag(tokenized) if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')] 
    return ' '.join(nouns)

def pos_tag_check(text):
    tokens = nltk.word_tokenize(text)
    print("Parts of speech:\n ", nltk.pos_tag(tokens))

def get_keywords_yake(dataframe, li, score, flag, ngram, top):
    if flag is True:
        #print("true flag")
        df = dataframe.apply(nouns_adj)
    else:
        #print("false flag")
        df = dataframe.apply(nouns_adj_verb)
    
    #df1 = df.apply(pos_tag_check)
    
    for text in range(len(dataframe)):
        y = yake.KeywordExtractor(lan='en',          # language
                                 n = ngram,              # n-gram size
                                 dedupLim = 0.5,     # deduplicationthresold
                                 dedupFunc = 'seqm', #  deduplication algorithm
                                 windowsSize = 15,
                                 top = top,           # number of keys
                                 features=None)           
        
        keywords = y.extract_keywords(dataframe.iloc[text])
        for k in keywords:
            if k[1] <=score:
                li.append(k[0])
                #print(k[0])
        #return keywords


def remove_duplicate_words(list):
    li = []
    for text in list:
        li.append(re.sub(r'\b(\w+\s*)\1{1,}', '\\1', text)) # removing consecutive identical words
    return li

# for removing exactly similar phrases from a list
def remove_duplicate_phrase(list):
    res = []
    for i in list:
        if i not in res:
            res.append(i)
    return res

# for removing phrase which are some percent similar
def remove_repeated_phrases(list):
    lis = []
    for a, b in itertools.combinations(list, 2):
        words_a = a.split()
        words_b = b.split()
        percent = len(set(words_a) & set(words_b)) / float(len(set(words_a) | set(words_b))) * 100
        if percent >= 25:
            if len(a) < len(b):
                lis.append(a)
            else:
                lis.append(b)
     
    res = [i for i in list if i not in lis]
    return res
        
        


def replace(text, pattern):
    for (raw, rep) in pattern:
        regex = re.compile(raw)
        
        text = regex.sub(rep, text)
    return text


def Yake_main(df1):
    #df1 = pd.read_pickle('example_dataset.pkl')
    #stopword_dir =  "E:/Thesis/codes/final_codes/code_v4/keyword_extraction/stopwords.txt"
    #rel_columns = ["Type of failure","Technical explanation","Modification concept","To be measured or simulated","Schematic modification","Layout modification","Modification of test program","Product specification modification","Measures against recurrence"]
    rel_columns = ["Type of failure","Technical explanation","Modification concept","Schematic modification","Layout modification","To be measured or simulated","Modification of test program","Product specification modification","Additional Information"]
    #rel_columns = ["Type of failure", "Technical explanation", "Modification concept", "Schematic modification", "Layout modification", "Modification of test program", "Additional Information"]
    list = {}
    flag = False
    for column_name in rel_columns:
        list[column_name] = []
        df_for_singleText = df1[column_name]
        if (column_name in ["Type of failure"]):
            score1 = 300
            score2 = 1000
            #print("\nType of failure:")
            liste1 = custom_main(df_for_singleText, score1, score2)
            liste11 = remove_duplicate_phrase(liste1)
            value11 = []
            for i in liste11:
                pattern = [(r'Improvement ', '')]
                value = replace(i, pattern)
                value11.append(value)

            
        elif (column_name in ["Technical explanation"]):
            score1 = 300
            score2 = 1000
            #print("\nTechnical explanation:")
            #print("\n")
            liste3 = custom_main(df_for_singleText, score1, score2)
            liste33 = remove_duplicate_phrase(liste3)
            
        elif (column_name in ["Modification concept"]):
            score1 = 300
            score2 = 1000
            #print("\nModification concept:")
            liste2 = custom_main(df_for_singleText, score1, score2)
            liste22 = remove_duplicate_phrase(liste2)
            value22 = []
            for i in liste22:
                pattern = [(r'Improvement ', ''),(r'Reduce ', 'Increased '),(r'By this ', 'Due to this '),(r'modification ', 'error '),(r'more ', 'less '),(r'advantageous ', 'disadvantageous '),(r'Insert ', 'Missing '),(r'Set ', 'Need to set ')]
                value = replace(i, pattern)
                value22.append(value)
                
                
        elif (column_name in ["Schematic modification"]):
            flag = True
            ngram = 10
            score = 1#0.0001
            top = 3
            get_keywords_yake(df_for_singleText, list[column_name], score, flag, ngram, top)
        
        elif (column_name in ["Layout modification"]):
            flag = True
            ngram = 10
            score = 1#0.0001
            top = 3
            get_keywords_yake(df_for_singleText, list[column_name], score, flag, ngram, top)
            
        elif (column_name in ["Modification of test program"]):
            flag = True
            ngram = 10
            score = 1#0.0001
            top = 3
            get_keywords_yake(df_for_singleText, list[column_name], score, flag, ngram, top)
            
        elif (column_name in ["Additional Information"]):
            flag = True
            ngram = 10
            score = 1#0.0001
            top = 2
            get_keywords_yake(df_for_singleText, list[column_name], score, flag, ngram, top)
           
            
        elif (column_name in ["To be measured or simulated","Product specification modification"]):
            flag = True
            ngram = 10
            score = 1#0.0001
            top = 3
            get_keywords_yake(df_for_singleText, list[column_name], score, flag, ngram, top)

    
    final_text = []
    root_reason = []
    root_reason2 = []
    for keys,values in list.items():
        
        values = sorted(set(values), key=lambda x:values.index(x))   # remove duplicate phrases
        value = remove_duplicate_words(values)
        list_filtered = remove_repeated_phrases(value)
        
        if keys=="Modification concept":
            if value != []:
                value7 = []
                for i in list_filtered:
                    pattern = [(r'adjust ', 'wrong '),(r'update', 'not updated'),(r'add ','missing '),(r'adapt','Missfit'),(r'new','old'),(r' changed',' unchanged'),(r'fix', 'not be able to fix')]
                    value1 = replace(i, pattern)
                    value7.append(value1)
                #text = "\nFields which require modifications are: \n" + ' , '.join(value7)
                root_reason.append(' , '.join(value7))
        elif keys=="To be measured or simulated":
            if value != []:
                value6 = []
                for i in list_filtered:
                    pattern = [(r'update ', 'not updated'),(r'add ','missing '),(r'adapt ','Missfit '),(r'new ','old '),(r'changed ','unchanged '),(r'fix ','not be able to fix '),(r'check ','')]
                    value1 = replace(i, pattern)
                    value6.append(value1)
                text = "<br/>Fields that are required to be checked are: <br/>" + ' ,'.join(value6)
                final_text.append(text)
        elif keys=="Schematic modification":
            if value != []:
                value3 = []
                for i in list_filtered:
                    pattern2 = [(r'update', 'no updated'),(r'add ','missing '),(r'added ','missing '),(r'adapt','Missfit'),(r'new','old'),(r'changed','unchanged'),(r'fix', 'not be able to fix'),(r'replace', 'error in'),(r'modification',''),(r'modify','error in'),(r'complete','incomplete')]
                    value0 = replace(i, pattern2)
                    value3.append(value0)
                #text = "Schematic errors which can be a reason for failure are: \n" + ' ,'.join(value3)
                root_reason.append(' ,'.join(value3))
        elif keys=="Layout modification":
            if value != []:
                value5 = []
                for i in list_filtered:
                    pattern = [(r'update', 'not updated'),(r'add','missing '),(r'adapt','Missfit'),(r'new','old'),(r'changed','unchanged'),(r'fix', 'not be able to fix'),(r'replace','error in'),(r'modification','error')]
                    value1 = replace(i, pattern)
                    value5.append(value1)
                #text = "Layout errors which can be a reason for failure are: \n" + ' ,'.join(value5)
                root_reason.append(' ,'.join(value5))
        elif keys=="Modification of test program":
            if value != []:
                value4 = []
                for i in list_filtered:
                    pattern = [(r'update', 'not updated'),(r'add ','missing '),(r'adapt','Missfit'),(r'new','old'),(r'changed','unchanged'),(r'fix', 'not be able to fix'),(r'modification of ','')]
                    value1 = replace(i, pattern)
                    value4.append(value1)
                #text = "Failure may occur due to the errors in some of the test programs are: \n" + ', '.join(value4)
                root_reason.append(' ,'.join(value4))
        elif keys=="Additional Information":
            if value != []:
                value4 = []
                for i in list_filtered:
                    value4.append(i)
                #text = "Failure may occur due to the errors in some of the test programs are: \n" + ', '.join(value4)
                root_reason2.append(' ,'.join(value4))
        elif keys=="Product specification modification":
            if value != []:
                value2 = []
                for i in list_filtered:
                    pattern = [(r'update ', 'need to update '),(r'add ','need to add '),(r'adapt ','need to adapt '),(r'new ','need new '),(r'changed ','need to change '),(r'fix ', 'need to fix ')]
                    value1 = replace(i, pattern)
                    value2.append(value1)
                text = "Product specification modification that needs to be performed are: <br/>" + ' ,'.join(value2)
                final_text.append(text)


    tex = "Potential reasons for failure:<br/><br/>"
    #print("\nPotential reasons for failure:")
    count = 1
        
    for i in value11:
        tex = tex + str(count) + ". " + str(i) + "<br/>"
        #print(str(count) + ". " + str(i))
        count = count + 1
    for i in value22:
        tex = tex + str(count) + ". " + str(i) + "<br/>"
        #print(str(count) + ". " + str(i))
        count = count + 1
    root_reason1 = remove_duplicate_phrase(root_reason)
    for i in root_reason1:
        tex = tex + str(count) + ". Error in "+ str(i) + "<br/>"
        #print(str(count) + ". Error in "+ str(i))
        count = count + 1
    root_reason22 = remove_duplicate_phrase(root_reason2)
    for i in root_reason22:
        tex = tex + str(count) + ". Error due to "+ str(i) + "<br/>"
        #print(str(count) + ". Error due to "+ str(i))
        count = count + 1
  
    tex = tex +"<br/>"+ "<br/><br/>Detailed description:<br/>"
    #print("\n\nDetailed description:")
    if liste3 != []:
        tex = tex + "<br/>Technical explanations: <br/>"
        #print("\nTechnical explanation:")
        for i in liste33:
            tex = tex + i
            #print(i)
        #print("\n")
    final_text1 = remove_duplicate_phrase(final_text)
    tex = "<br/>" + tex + "<br/>".join(final_text1)
    #print('\n\n'.join(final_text1))
    return tex
    
if __name__ == "__main__":
    Yake_main()