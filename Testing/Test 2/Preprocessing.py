import re
import numpy as np

tokens=[]
indices={}
filtered_corpus=[]
stopwords=[]
filtered_sentences=[]
def preprocess(text):
    all_sent=[]
    corpus=""
    for i in text:
        i=str(i).lower()
        i=i.replace("/"," ")
        i=i.replace("."," ")
        i=i.replace('"'," ")
        text = re.sub(r'[^a-zA-Z" "]', '', i)
        all_sent.append(text)

        corpus = corpus + " "+text #merging all words in the dataset

    corpus_words = corpus.split()
    token_freq={}
    for i in corpus_words:
        if i in token_freq:
            token_freq[i]+=1
        else:
            token_freq[i]=1

    # to find maximum frequency words for creating stopwords
    
    for i in token_freq:
        if token_freq[i]>300 or len(i)<3: #this numerical values aretrial and error method
            stopwords.append(i)

    np.save('stopwords',stopwords)
    #filtered sentences- removing stopwords from all sentence

    
    for i in all_sent:
        fil_text=[]
        
        text=i.split()
        
        for j in text:
            if not j in stopwords:
                fil_text.append(j)
        filtered_sentences.append(fil_text)

    #removing stopwords from the corpus

    
    for i in corpus_words:
    
        if not i in stopwords:
            filtered_corpus.append(i)

    #frequency of filtered corpus
    filtered_corpus_freq={}

    for i in filtered_corpus:
        if i in filtered_corpus_freq:
            filtered_corpus_freq[i]+=1
        else:
            filtered_corpus_freq[i]=1        

    #creating tokens and assigning indices

    
    for i in filtered_corpus_freq:
        tokens.append(i)


    for i,j in enumerate (tokens):
        indices[j]=i        
    np.save('indices.npy',indices)
    print("No of tokens :: ",len(tokens))
    return filtered_corpus,filtered_sentences,indices,tokens