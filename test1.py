import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
from Preprocessing import preprocess
from word2vec import wordvec
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from text_cleaning import clean

#reading CSV Dataset
df = pd.read_csv('test.csv', encoding='latin-1')

#Selecting the required feature
df=df[['sentiment','text']]

#Deleteing NAN values
df=df.dropna()

#assigning numerical values for the labels
label_dict = {'positive':0,'negative':1,'neutral':2}
df['label'] = df['sentiment'].apply(lambda x: label_dict.get(x))

#Reordering the features
df=df.iloc[:,[1,0,2]]

X= df['text']
y=df['label']

n_dim= 100 #no of dimension for word2vec
filtered_corpus,filtered_sentences,indices,tokens = preprocess(X)
word_embedding=wordvec(filtered_sentences,tokens,n_dim)

sentences_vector = [] #combined sentece vectors of entire corpus 
count=0

for sent in filtered_sentences:
    count+=1
    text_vector=[] #vector for each word in a sentence

    

    for token in sent:
        token_index=indices[token]
        text_vector.append(word_embedding[token_index])
    text_vector=np.array(text_vector) 

    
    text_vector=text_vector.mean(axis=0)

    #sometimes text will be null after removing stopwords so we have to make that vector Padded with 0
    if len(sent) == 0:
        text_vector=np.zeros(n_dim)    

    sentences_vector.append(text_vector)

sentences_vector=np.array(sentences_vector)
sentences_vector=torch.tensor(sentences_vector, dtype=torch.float32)
  

class analysis(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(n_dim, 550)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(550,250)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(250,120)
        self.act3 = nn.ReLU()
        self.hidden4 = nn.Linear(120,60)
        self.act4 = nn.ReLU()
        self.hidden5 = nn.Linear(60,8)
        self.act5 = nn.ReLU()
        self.output = nn.Linear(8, 3)
        self.act_output = nn.Softmax(dim=0)

    def forward(self,x):
        x = self.act1(self.linear(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act4(self.hidden4(x))
        x = self.act5(self.hidden5(x))
        x = self.act_output(self.output(x))
        return x    

model=analysis()

lr=0.0001
a=model(sentences_vector)
optimizer = optim.Adam(model.parameters(), lr)


X=sentences_vector
Y=torch.tensor(y)


acc=0
no_data=len(sentences_vector)
no_epoch =1
batch_size=80
epoch_avgloss={}
epoch_avgacc={}

for epoch in range(no_epoch):
    loss_sum=0
    no_batches=0
    avgg_acc=0
    for i in range(0,no_data,batch_size):
        x=sentences_vector[i:i+batch_size]
        y_pred=model(x)
        y=Y[i:i+batch_size]
        loss=F.cross_entropy(y_pred,y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum+=loss
        no_batches+=1

        #validation accuracy
        acc=0
        for j in range(len(y)):
            y_pred_ind=torch.argmax(y_pred[j])
            if y_pred_ind == y[j]:
                acc+=1
        avgg_acc+=acc/len(y)    
    avg_acc=avgg_acc/no_batches
    print(loss_sum/no_batches)  
    avg_loss =  loss_sum/no_batches
    epoch_avgloss[epoch]=avg_loss.detach().numpy()
    epoch_avgacc[epoch]=avg_acc
    print(epoch,'----------',"loss : ",avg_loss,"       acc : ",avg_acc)


torch.save(model.state_dict(), 'model_weights.pth')

#Graph Representation
import matplotlib.pyplot as plt
plt.plot(epoch_avgloss.keys(),epoch_avgloss.values(),label="learning rate = {}".format(lr))
plt.title("n-dim =100 , word2vec--lr=0.001, epoch =200,\n window =5, analysis -- lr=0.001, no_epoch=250, batch =80")
plt.xlabel("epochs")
plt.ylabel("cost")
plt.show()

plt.plot(epoch_avgacc.keys(),epoch_avgacc.values(),label="learning rate = {}".format(lr))
plt.title("n-dim =100 , word2vec--lr=0.001, epoch =200\n, window =5, analysis -- lr=0.001, no_epoch=250, batch =80")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.show()


#Testing phase

acc=0
for i in range(no_data):
    x=sentences_vector[i]
    y_pred=model(x)
    y_pred_ind=torch.argmax(y_pred)
    if y_pred_ind == Y[i]:
        acc+=1
avg_acc=acc/no_data
print("Testing accuracy :: ",avg_acc)


#Prediction Phase
# test_sent='hey peoples, dont you just hate being grounded haha, im just sat eating an apple and watching death note (some anime)'

# filtered_sent=clean(test_sent)

# text_vector=[] #vector for each word in a sentence

# for token in filtered_sent:
#     token_index=indices[token]
#     text_vector.append(word_embedding[token_index])
# text_vector=np.array(text_vector)
# text_vector=torch.tensor(text_vector.mean(axis=0),dtype=torch.float32)

# print("text sentence vector")
# print(text_vector.shape)
# pred_output=model(text_vector)

# print(pred_output)
# a=torch.argmax(pred_output)
# sentiment = [key for key, val in label_dict.items() if val == a]
# label_dict = {'positive':0,'negative':1,'neutral':2}

# print(sentiment)
# x=sentences_vector[]