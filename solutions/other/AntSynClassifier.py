import numpy as np
from scipy import spatial
from scipy import stats
from sklearn import linear_model
import os
from sklearn import metrics
from sklearn.preprocessing import normalize
import pandas as pd
from sklearn import neural_network

w2vfile = os.path.join(os.path.dirname(__file__), '../Word-Similarity/word2vec/W2V_150.txt')
train_ant_file = os.path.join(os.path.dirname(__file__), '../Word-Similarity/antonym-synonym set/Antonym_vietnamese.txt')
train_syn_file = os.path.join(os.path.dirname(__file__), '../Word-Similarity/antonym-synonym set/Synonym_vietnamese.txt')
test_noun_file = os.path.join(os.path.dirname(__file__), '../Word-Similarity/datasets/ViCon-400/400_noun_pairs.txt')
test_verb_file = os.path.join(os.path.dirname(__file__), '../Word-Similarity/datasets/ViCon-400/400_verb_pairs.txt')
test_adj_file = os.path.join(os.path.dirname(__file__), '../Word-Similarity/datasets/ViCon-400/600_adj_pairs.txt')

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

dict = loadWordEmbbed(w2vfile)
# print(dict)



def calculate_feature(u1, u2):

    u1 = np.array(list(u1))
    u2 = np.array(list(u2))
    # print(u1.shape)
    # u1 = normalize([u1])[0]
    # u2 = normalize([u2])[0]
    v = u1 + u2
    v = np.append(v, np.square(u1 + u2))
    v = np.append(v, u1 * u2)
    # v = np.append(v, np.square(u1 - u2))
    # v = np.append(u1, u2)
    sim = (2 - spatial.distance.cosine(u1, u2)) / 2
    v = np.append(v, sim)
    r = stats.pearsonr(u1, u2)
    # v = np.append(v, r)

    # v = normalize([v])
    return v


def load_data(dict, filename, type='training', label='SYN'):
    samples = []
    labels = []
    data = open(filename, 'r', encoding='utf-8')

    if type == 'test':
        data.readline()

    for i in data:
        s = i.split()
        if ((type == 'training' and len(s) > 1) or (len(s) > 2)):
            w1 = s[0].strip()
            w2 = s[1].strip()
            if (w1 in dict and w2 in dict):
                v = calculate_feature(dict[w1], dict[w2])
                samples.append(v)
                # print(v.shape)
                if type == 'training':
                    labels.append(label)
                else:
                    gt = s[2].strip()
                    labels.append(gt)
    data.close()
    samples = normalize(samples)
    samples = list(samples)

    return samples,labels


X1, y1 = load_data(dict, train_ant_file, label='ANT')
X2, y2 = load_data(dict, train_syn_file)
# print(np.array(X1).shape)
X = list(X1) + list(X2)
y = y1 + y2

XTest, yTest = load_data(dict, test_noun_file, type='test')

N = len(X)
N_train = int(2 * N / 3)

# shuffle data
arr = np.arange(N)
np.random.shuffle(arr)
# print(arr)
X = np.array(X)[arr.astype(int)]
y = np.array(y)[arr.astype(int)]

# df = pd.DataFrame(X)
# print(df.describe())

model = linear_model.LogisticRegression(random_state=0, max_iter=1000)
# model.fit(X, y)


mlp_model = neural_network.MLPClassifier((80, ), random_state=0, max_iter=1000, alpha=0.0001)
mlp_model.fit(X[:N_train], y[:N_train])

print('Validation: ')
y_pred = mlp_model.predict(X[N_train : N])
# print(y_pred)
# print(y[N_train : N])
print(metrics.precision_recall_fscore_support(y[N_train : N], y_pred, average='micro'))

y_pred = mlp_model.predict(XTest)
# print(y_pred[:10])
# print(yTest[:10])
print('Test with nours:')
print(metrics.precision_recall_fscore_support(yTest, y_pred))

XTest, yTest = load_data(dict, test_verb_file, type='test')
y_pred = mlp_model.predict(XTest)
print('Test with verbs:')
print(metrics.precision_recall_fscore_support(yTest, y_pred))

XTest, yTest = load_data(dict, test_adj_file, type='test')
y_pred = mlp_model.predict(XTest)
print('Test with adjectives:')
print(metrics.precision_recall_fscore_support(yTest, y_pred))

# Output:
#
# Validation on unseen 1/3 data :
# (0.9104197901049476, 0.9104197901049476, 0.9104197901049476, None)
# Test with nours:
# (array([0.98639456, 0.84916201]), array([0.84302326, 0.98701299]), array([0.90909091, 0.91291291]), array([172, 154], dtype=int64))
# Test with verbs:
# (array([0.99375   , 0.85393258]), array([0.85945946, 0.99346405]), array([0.92173913, 0.918429  ]), array([185, 153], dtype=int64))
# Test with adjectives:
# (array([0.98181818, 0.94871795]), array([0.95744681, 0.97797357]), array([0.96947935, 0.96312364]), array([282, 227], dtype=int64))
