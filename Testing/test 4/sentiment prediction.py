from text_cleaning import clean
import numpy as np
import torch
from test1 import analysis
from Preprocessing import indices

label_dict = {'positive':0,'negative':1,'neutral':2}

word_embedding=np.load('we1.npy', allow_pickle=True)

model = analysis()
model.load_state_dict(torch.load('model_weights.pth'))



test_sent=' soooooo wish i could, but im in school and myspace is completely blocked'

filtered_sent=clean(test_sent)

text_vector=[] #vector for each word in a sentence

for token in filtered_sent:
    token_index=indices[token]
    text_vector.append(word_embedding[token_index])
text_vector=np.array(text_vector)
text_vector=torch.tensor(text_vector.mean(axis=0),dtype=torch.float32)

print("text sentence vector")
print(text_vector.shape)
pred_output=model(text_vector)

print(pred_output)
a=torch.argmax(pred_output)
sentiment = [key for key, val in label_dict.items() if val == a]
label_dict = {'positive':0,'negative':1,'neutral':2}

print("========TEXT==========")
print(test_sent)
print("sentiment :: ",sentiment)
