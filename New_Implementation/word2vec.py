import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt

df = pd.read_csv('twitter_training.csv', names=['id','source','sentiment', 'text'])
df=df[['sentiment','text']]
possible_labels = df.sentiment.unique()
df = df[~df.sentiment.str.contains('Irrelevant')]

possible_labels = df.sentiment.unique()
no_labels = len(possible_labels)
one_hot=np.eye(no_labels)

label_dict = {'Positive':[1,0,0],'Negative':[0,1,0],'Neutral':[0,0,1]}

df['label'] = df['sentiment'].apply(lambda x: label_dict.get(x))
df=df.iloc[:,[1,0,2]]

#removing NAN values
df=df.dropna()

X= df['text']
y=df['label']

X_train, X_test, y_train, y_test = train_test_split(
  X,y , random_state=104,test_size=0.95, shuffle=True)

all_sent=[]
corpus=""
for i in X_train:
    i=str(i).lower()
    i=i.replace("/"," ")
    i=i.replace("."," ")
    text = re.sub(r'[^a-zA-Z" "]', '', i)
    all_sent.append(text)
    corpus = corpus + " "+text


corpus=corpus.lower()
corpus=corpus.strip("   ")

corpus_words = corpus.split()

#frequency of each words in corpus
token_freq={}
for i in corpus_words:
    if i in token_freq:
        token_freq[i]+=1
    else:
        token_freq[i]=1

stopwords=[]
for i in token_freq:
    if token_freq[i]>300 or len(i)<3: #this numerical values aretrial and error method
        stopwords.append(i)

#removing stopwords from the corpus
filtered_corpus=[]
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

tokens=[]
indices={} #tokens with index

for i in filtered_corpus_freq:
    tokens.append(i)


for i,j in enumerate (tokens):
    indices[i]=j

no_of_tokens=len(tokens)           

#one hot encoding
one_hot=np.eye(no_of_tokens)

#filtered tot sentence
 # remove stopwords from each text in the dataset
pure_totsent=[] #append all text in corpus after removing stopwords
m=0
for i in all_sent:
    pure_sent=[]
    m+=1
    words=i.split()
    
    
    for j in words:
        if j not in stopwords:
            pure_sent.append(j)
    
    
    pure_totsent.append(pure_sent)   


def softmax(a):
  a=a/np.linalg.norm(a,axis=0,keepdims=True)

  if len(a.shape)>1:
    max_matrix=np.max(a,axis=0) #compare all the rows for each column and takes the bigger value for each column make a single vector
    stable = a-max_matrix
    e=np.exp(stable)
    A= e/np.sum(e,axis=0)
  else : #If there is only one matrix
    max_matrix=np.max(a) #Take the max_value from the single vector
    stable = a-max_matrix
    e=np.exp(stable)
    A= e/np.sum(e)

  return A

np.random.seed(42)

lr=0.01
epoch_avgcost={} #for plotting the graph
total_cost=0
n_dim=1000
weight1=np.random.rand(no_of_tokens,n_dim)-.5
weight2=np.random.rand(n_dim,no_of_tokens)-.5

no_of_epochs=65

for epoch in range(no_of_epochs):
    avg_cost=[]
    for center_word in tokens:
        center_word_token_id = tokens.index(center_word)
        one_hot_center=one_hot[center_word_token_id ]
        centerword_embed=np.dot(one_hot_center,weight1)

        all_contexts = []
        for i in pure_totsent:
            if center_word in i:
                all_contexts.append(i)

        all_contexts_len=len(all_contexts)
        context_rnum = np.random.randint(all_contexts_len)
        context=all_contexts[context_rnum]

        center_wordid_in_context=context.index(center_word)


        #selecting the context words according to window size
        window = 3
        context_words=[]
        context_words=context[max(0,(center_wordid_in_context-window)):center_wordid_in_context]+context[center_wordid_in_context+1:center_wordid_in_context+window+1]


        word_check_centerword = np.dot(centerword_embed,weight2) #passing embedded center word throgh weight 2 to convert into original dimension)
        Soft_max=softmax(word_check_centerword)

        total_cost=[]
        cost=0
    #change made==================================
        for i in context_words:
            cost=(-np.log(Soft_max[tokens.index(i)]))
            total_cost.append(cost)
            if np.sum(total_cost)!=0:
                avg_cost.append(np.average(total_cost))

        


        #finding loss
        loss=Soft_max
        for i in context_words:
         loss[tokens.index(i)]-=1



        #finding gradient of loss and weight1 & subtracting from weight2
        dw2=np.dot(loss,weight1)


        weight2.T[center_word_token_id]=weight2.T[center_word_token_id]-dw2*lr

        #finding gradient of loss and weight1 & subtracting from weight2
        dw1=np.dot(loss,weight2.T)

        weight1[center_word_token_id]=weight1[center_word_token_id]-dw1*lr

  

    epoch_avgcost[epoch]=np.average(avg_cost)
    print(epoch,'----------',np.average(avg_cost))
        
    



  