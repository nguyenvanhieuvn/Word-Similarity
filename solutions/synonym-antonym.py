import os

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support as score

try:
    from solutions import cosin_similarity
except:
    import cosin_similarity

w2v_path = os.path.join(os.path.dirname(__file__), '../word2vec/W2V_150.txt')
word2vec = cosin_similarity.Word2Vec(w2v_path)


def to_feature(word1, word2):
    # Thay đổi featured cho mô hình ở đây
    # featured đang dùng gồm (w2v của 2 từ, độ đo spearmanr & pearsonr của hai từ)
    x = []
    x.extend(word2vec.w2v[word1])
    x.extend(word2vec.w2v[word2])
    x.extend(word2vec.spearmanr(word1, word2))
    x.extend(word2vec.pearsonr(word1, word2))
    return x


def load_train_data(ant_path, syn_path):
    X = []
    Y = []
    for line in open(ant_path).read().splitlines():
        line = line.strip()
        if line and len(line.split()) == 2:
            word1, word2 = line.split()
            if word1 in word2vec.w2v and word2 in word2vec.w2v:
                X.append(to_feature(word1, word2))
                Y.append('ANT')
    for line in open(syn_path).read().splitlines():
        line = line.strip()
        if line and len(line.split()) == 2:
            word1, word2 = line.split()
            if word1 in word2vec.w2v and word2 in word2vec.w2v:
                X.append(to_feature(word1, word2))
                Y.append('SYN')
    return X, Y


def load_test_data(test_path):
    X = []
    Y = []
    for line in open(test_path).read().splitlines():
        line = line.strip()
        if line and len(line.split()) == 3:
            word1, word2, label = line.split()
            if word1 in word2vec.w2v and word2 in word2vec.w2v:
                X.append(to_feature(word1, word2))
                Y.append(label)
    return X, Y


ant_file = os.path.join(os.path.dirname(__file__), '../antonym-synonym set/Antonym_vietnamese.txt')
syn_file = os.path.join(os.path.dirname(__file__), '../antonym-synonym set/Synonym_vietnamese.txt')
test_noun_path = os.path.join(os.path.dirname(__file__), '../datasets/ViCon-400/400_noun_pairs.txt')
test_verb_path = os.path.join(os.path.dirname(__file__), '../datasets/ViCon-400/400_verb_pairs.txt')
test_adj_path = os.path.join(os.path.dirname(__file__), '../datasets/ViCon-400/600_adj_pairs.txt')

X_train, Y_train = load_train_data(ant_file, syn_file)
le = preprocessing.LabelEncoder()
le.fit(Y_train)
Y_train = le.transform(Y_train)
print(len(X_train), len(Y_train))
clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, Y_train)

print('Test on noun:')
X_test, Y_test = load_test_data(test_noun_path)
Y_test = le.transform(Y_test)
Y_pred = clf.predict(X_test)

precision, recall, fscore, support = score(Y_test, Y_pred)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

print('Test on adj:')
X_test, Y_test = load_test_data(test_adj_path)
Y_test = le.transform(Y_test)
Y_pred = clf.predict(X_test)

precision, recall, fscore, support = score(Y_test, Y_pred)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

print('Test on verb:')
X_test, Y_test = load_test_data(test_verb_path)
Y_test = le.transform(Y_test)
Y_pred = clf.predict(X_test)

precision, recall, fscore, support = score(Y_test, Y_pred)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


# Kết quả
"""
Test on noun:
precision: [0.98876404 0.64556962]
recall: [0.51162791 0.99350649]
fscore: [0.6743295 0.7826087]
support: [172 154]
Test on adj:
precision: [0.86055777 0.74418605]
recall: [0.76595745 0.84581498]
fscore: [0.81050657 0.79175258]
support: [282 227]
Test on verb:
precision: [0.95876289 0.61825726]
recall: [0.5027027  0.97385621]
fscore: [0.65957447 0.75634518]
support: [185 153]
"""