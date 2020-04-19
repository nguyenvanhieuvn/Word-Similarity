import os
import re

import pandas as pd
from scipy import spatial
from scipy import stats


def load_w2v(path):
    w2v, w2v_size, w2v_dim = {}, 0, 0
    f = open(path, encoding='utf8')
    for idx, line in enumerate(f):
        line = line.strip()
        if idx == 0:
            w2v_size = int(line)
        elif idx == 1:
            w2v_dim = int(line)
        else:
            parts = re.split('\\s+', line)
            key = parts[0]
            value = [float(x) for x in parts[1:]]
            w2v[key] = value
    print('Load w2v done!')
    return w2v, w2v_size, w2v_dim


class Word2Vec:
    def __init__(self, path):
        try:
            self.w2v, self.w2v_size, self.w2v_dim = load_w2v(path)
        except:
            print('Can not load w2v model!')
            exit(0)

    def cosin(self, word1, word2):
        if word1 in self.w2v and word2 in self.w2v:
            v1 = self.w2v.get(word1)
            v2 = self.w2v.get(word2)
            # return spatial.distance.cosine(v1, v2)
            # TODO: Why?
            return (2 - spatial.distance.cosine(v1, v2)) / 2
        raise KeyError('{} or {} not found in dictionary!'.format(word1, word2))

    def pearsonr(self, word1, word2):
        if word1 in self.w2v and word2 in self.w2v:
            v1 = self.w2v.get(word1)
            v2 = self.w2v.get(word2)
            return stats.pearsonr(v1, v2)
        raise KeyError('{} or {} not found in dictionary!'.format(word1, word2))

    def spearmanr(self, word1, word2):
        if word1 in self.w2v and word2 in self.w2v:
            v1 = self.w2v.get(word1)
            v2 = self.w2v.get(word2)
            return stats.spearmanr(v1, v2)
        raise KeyError('{} or {} not found in dictionary!'.format(word1, word2))

    def evaluate(self, input_path, output_path):
        with open(output_path, 'w', encoding='utf8') as fp:
            for idx, line in enumerate(open(input_path, encoding='utf8')):
                line = line.strip()
                if idx == 0:
                    # Column name row
                    fp.write('Word1\tWord2\tPOS\tSim\tSim1\tSim2\tCosin\tPearsonr\tSpearmanr\tSTD\n')
                elif line:
                    _word1, _word2, _pos, _sim1, _sim2, _std = line.split('\t')
                    _sim1 = float(_sim1)
                    _sim = round(_sim1 / 6, 2)
                    try:
                        _cosin = round(self.cosin(_word1, _word2), 2)
                        _pearsonr = [round(x, 2) for x in self.pearsonr(_word1, _word2)]
                        _spearmanr = [round(x, 2) for x in self.spearmanr(_word1, _word2)]
                    except:
                        _cosin = 'NaN'
                        _pearsonr = 'NaN'
                        _spearmanr = 'NaN'

                    fp.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(_word1, _word2, _pos, _sim, _sim1, _sim2,
                                                                               _cosin, _pearsonr, _spearmanr, _std))

    def evaluate_csv(self, input_path, output_path):
        word1 = []
        word2 = []
        pos = []
        sim = []
        sim1 = []
        sim2 = []
        cosin = []
        pearsonr = []
        spearmanr = []
        std = []
        for idx, line in enumerate(open(input_path, encoding='utf8')):
            line = line.strip()
            if idx == 0:
                # Column name row
                continue
                # fp.write('Word1\tWord2\tPOS\tSim\tSim1\tSim2\tCosin\tPearsonr\tSpearmanr\tSTD\n')
            elif line:
                _word1, _word2, _pos, _sim1, _sim2, _std = line.split('\t')
                _sim1 = float(_sim1)
                _sim = round(_sim1 / 6, 2)
                try:
                    _cosin = round(self.cosin(_word1, _word2), 2)
                    _pearsonr = [round(x, 2) for x in self.pearsonr(_word1, _word2)]
                    _spearmanr = [round(x, 2) for x in self.spearmanr(_word1, _word2)]
                except:
                    _cosin = 'NaN'
                    _pearsonr = 'NaN'
                    _spearmanr = 'NaN'

            word1.append(_word1)
            word2.append(_word2)
            pos.append(_pos)
            sim.append(_sim)
            sim1.append(_sim1)
            sim2.append(_sim2)
            cosin.append(_cosin)
            pearsonr.append(_pearsonr)
            spearmanr.append(_spearmanr)
            std.append(_std)
        df = pd.DataFrame()
        df['word1'] = word1
        df['word2'] = word2
        df['pos'] = pos
        df['sim'] = sim
        df['sim1'] = sim1
        df['sim2'] = sim2
        df['cosin'] = cosin
        df['pearsonr'] = pearsonr
        df['spearmanr'] = spearmanr
        df['std'] = std
        df.to_csv(output_file, encoding='utf8')


if __name__ == '__main__':
    w2v_path = os.path.join(os.path.dirname(__file__), '../word2vec/W2V_150.txt')
    input_file = os.path.join(os.path.dirname(__file__), '../datasets/ViSim-400/Visim-400.txt')
    output_file = os.path.join(os.path.dirname(__file__), '../datasets/ViSim-400/Visim-400.txt.csv')
    word2vec = Word2Vec(w2v_path)
    word2vec.evaluate_csv(input_file, output_file)
