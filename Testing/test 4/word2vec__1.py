import numpy as np
from numba import jit, cuda 

#@cuda.jit
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

def wordvec(sentences,tokens,n_dim):
    print("===============WORD2VEC INITIATED===============")
    #word embedding for tokens with one-hot
    no_of_tokens=len(tokens)
    one_hot=np.eye(no_of_tokens)

    np.random.seed(42)
    lr=0.001
    epoch_avgcost={} #for plotting the graph
    total_cost=0
    n_dim= n_dim
    weight1=np.random.rand(no_of_tokens,n_dim)-.5
    weight2=np.random.rand(n_dim,no_of_tokens)-.5

    no_of_epochs=1
    
    for epoch in range(no_of_epochs):
        avg_cost=[]
        for center_word in tokens:
            center_word_token_id = tokens.index(center_word)
            one_hot_center=one_hot[center_word_token_id ]
            centerword_embed=np.dot(one_hot_center,weight1)

            all_contexts = []
            for i in sentences:
                if center_word in i:
                    all_contexts.append(i)

            all_contexts_len=len(all_contexts)
            context_rnum = np.random.randint(all_contexts_len)
            context=all_contexts[context_rnum]

            center_wordid_in_context=context.index(center_word)

            #selecting the context words according to window size
            window = 5
            context_words=[]
            context_words=context[max(0,(center_wordid_in_context-window)):center_wordid_in_context]+context[center_wordid_in_context+1:center_wordid_in_context+window+1]


            word_check_centerword = np.dot(centerword_embed,weight2) #passing embedded center word throgh weight 2 to convert into original dimension)
            Soft_max=softmax(word_check_centerword)

            total_cost=[]
            cost=0
            
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
    print("================WORD2VEC MODEL PREPARED==============================")
    #saving weights    
    np.save('we1.npy',weight1)
    np.save('we2.npy',weight2)    
    return weight1    