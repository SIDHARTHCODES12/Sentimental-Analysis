import numpy as np
import re


def clean(text):
    stopwords = np.load('stopwords.npy')
    filterd_text=[]
    text=text.lower()
    text=text.replace("/"," ")
    text=text.replace("."," ")
    text=text.replace('"'," ")
    text = re.sub(r'[^a-zA-Z" "]', '', text)
    tokens=text.split()
    for i in tokens:
        if not i in stopwords:
            filterd_text.append(i)

    return filterd_text
