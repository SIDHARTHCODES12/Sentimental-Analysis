import numpy as np
import torch

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