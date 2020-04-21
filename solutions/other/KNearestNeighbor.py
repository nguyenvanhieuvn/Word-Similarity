import io
import numpy as np
import os
from scipy import spatial
from scipy import stats
from sklearn import neighbors

w2vfile = os.path.join(os.path.dirname(__file__), '../Word-Similarity/word2vec/W2V_150.txt')

def loadWordEmbbed(filename):

    dict = {}
    w2vData = open(w2vfile, 'r', encoding='utf-8')
    N = int(w2vData.readline())
    D = int(w2vData.readline())
    for i in w2vData:
        s = i.split()
        v = [float(val) for val in s[1:]]
        dict[s[0].strip()] = v
        # break
    w2vData.close()
    return dict

def calculate_sim(u1, u2):
    sim = (2 - spatial.distance.cosine(u1, u2)) / 2
    return 1 - sim

def nearest_neighbor_model(samples, k = 5):
    model = neighbors.NearestNeighbors(k, metric=calculate_sim)
    model.fit(samples)

    return model

dict = loadWordEmbbed(w2vfile)

data = np.array(list(dict.values()))
keys = np.array(list(dict.keys()))

model = nearest_neighbor_model(data, 5)

w = input('Enter your word: ')
if w.strip() in dict:
    k = int(input('Enter number of neighbors: '))
    w2v = dict[w.strip()]
    # print(w2v)
    neigh = model.kneighbors([w2v], k)
    print(neigh)
    print(keys[neigh[1]])
    # print(calculate_sim(data[0], data[0]))
else:
    print('Word %s is not existed in dictionary' %w)