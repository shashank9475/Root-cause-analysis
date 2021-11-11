import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from collections import defaultdict


def custom_main(contents, scre, scre2):
    lis = []
    liste = []
    #contents = df_withoutClean["Type of failure"]
    for row in contents:
        lis.append(row)
    #print(lis)
    for i in lis:
        sentences = nltk.sent_tokenize(i)
        #print(sentences)
    
        phrases = []
        
        for line in sentences:
            line = re.sub("[\(\[].*?[\)\]]", "", line)
            punctuations = ['!','"','#','$','%','&','(',')','*','+','/',':',';','<','=','?','@','[',']','^','`','{','|','}','~',"'"]
            for punctuation in punctuations:
                line = line.replace(punctuation, " ")
            
            words = nltk.word_tokenize(line)
            phrase = ''
            for word in words:
                if word not in ['.','?',';']:
                    phrase+=word + ' '
                else:
                    if phrase != '':
                        phrases.append(phrase.strip())
                        phrase = ''
        
        #print('******** Candidate Phrases **********************')
        #print(phrases)
    
        word_freq = defaultdict(int)
        word_degree = defaultdict(int)
        word_score = defaultdict(float)
        
        for phrase in phrases:
            words = phrase.split(' ')
            phrase_length = len(words)
            for word in words:
                word_freq[word]+=1
                word_degree[word]+=phrase_length
        
        
        for word,freq in word_freq.items():
            degree = word_degree[word]
            score = ( 1.0 * degree ) / (1.0 * freq )
            word_score[word] = score
        
        phrase_scores = defaultdict(float)
        
        
        for phrase in phrases:
            words = phrase.split(' ')
            score = 0.0
            for word in words:
                score+=word_score[word]
            phrase_scores[phrase] = score
        
        #print('\n\n******** Candidate Phrases scored ****************\n')
        
        for k in sorted(phrase_scores,key = phrase_scores.get,reverse=True):
            if phrase_scores[k] > scre and phrase_scores[k] < scre2:
                liste.append(k)
                #print('. '.join(k.split('\n')),phrase_scores[k])
                #print(k,phrase_scores[k])

    return liste