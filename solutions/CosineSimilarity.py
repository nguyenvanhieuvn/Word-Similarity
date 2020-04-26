import io
import numpy as np
import os
from scipy import spatial
from scipy import stats
from sklearn import neighbors
import pandas as pd


w2vfile = os.path.join(os.path.dirname(__file__), '../Word-Similarity/word2vec/W2V_150.txt')
vsim_400_file = os.path.join(os.path.dirname(__file__), '../Word-Similarity/datasets/ViSim-400/Visim-400.txt')
vsim_400_file_out = os.path.join(os.path.dirname(__file__), '../Word-Similarity/datasets/ViSim-400/Visim-400_1.txt.csv')

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

def test_similarity(dict):
    sample_sim = []
    vsim_output = []
    rs = []
    dataset = open(vsim_400_file, 'r', encoding='utf-8')
    dataset.readline()
    for i in dataset:
        s = i.split()

        if (s[0].strip() in dict and s[1].strip() in dict):
            u1 = dict[s[0].strip()] # word 1
            u2 = dict[s[1].strip()] # word 2
            sim = (2 - spatial.distance.cosine(u1, u2))/2
            rs.append(sim)
            sim1 = (float(s[4]) / 10)
            sample_sim.append(sim1)
            pearson = stats.pearsonr(u1, u2)
            spearmanr = stats.spearmanr(u1, u2)
            vsim_output.append((s[0], s[1], s[2], s[3], s[4], sim1, sim, s[5]))

        # break

    # print(sample_sim)
    df = pd.DataFrame(columns=['Word1', 'Word2', 'Pos', 'Sim1 [0,6]', 'Sim2 [0,10]', 'Sim3 [0, 1]', 'Sim Computed [0, 1]', 'STD'],
                      data=vsim_output)
    # print(df.sample())
    df.to_csv(vsim_400_file_out, encoding='utf-8', header=True)
    print(" Pearson correlation coefficient: ", stats.pearsonr(rs,sample_sim))
    print(" Spearman's rank correlation coefficient: ", stats.spearmanr(rs,sample_sim))

    # print(rs)

dict = loadWordEmbbed(w2vfile)

test_similarity(dict)

print('Output is written to ' + vsim_400_file_out)

# Pearson correlation coefficient:  (0.4468197550880767, 2.7581737626379143e-18)
# Spearman's rank correlation coefficient:  SpearmanrResult(correlation=0.4077568887734169, pvalue=3.26456245952008e-15)
# Output is written to %sD:/Projects/PycharmProjects/NLP_WordSimilarity/venv\../Word-Similarity/datasets/ViSim-400/Visim-400_1.txt.csv

