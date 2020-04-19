import os
import sys

try:
    from solutions import cosin_similarity
except:
    import cosin_similarity

if __name__ == '__main__':
    w2v_path = os.path.join(os.path.dirname(__file__), '../word2vec/W2V_150.txt')
    word2vec = cosin_similarity.Word2Vec(w2v_path)
    try:
        word = sys.argv[1]
        k = int(sys.argv[2])
    except:
        word = 'nóng'
        k = 5
    distance = {}
    for key in word2vec.w2v:
        if key == word:
            continue
        distance[key] = word2vec.cosin(word, key)
    print('Top {} best similar word of `{}`:'.format(k, word))
    for key in sorted(distance, key=distance.get, reverse=True)[:k]:
        print('{}\t{}'.format(key, distance.get(key)))
