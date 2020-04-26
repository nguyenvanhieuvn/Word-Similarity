## Tổng quan
- Task 1: Tại sao lại cần (2 - cosin_distance) / 2 ?
- Task 3: Một số mẫu dữ liệu bị thiếu từ đi cặp cùng

1. Cosine Similarity

- Pearson correlation coefficient:  (0.4468197550880767, 2.7581737626379143e-18)
- Spearman's rank correlation coefficient:  SpearmanrResult(correlation=0.4077568887734169, pvalue=3.26456245952008e-15)
- Output is written to ../Word-Similarity/datasets/ViSim-400/Visim-400_1.txt.csv

2. K-Nearest neighbor

- Enter your word: không_biết
- Enter number of neighbors: 5
 (array([[0.        , 0.14774525, 0.17461857, 0.1928641 , 0.19320361]]), array([[ 2498,  5814, 23615, 12074, 44306]], dtype=int64))
 [['không_biết' 'không_hiểu' 'không_biết_được' 'không_rõ' 'không_hay_biết']]

3. Ant - Syn Classifier

- Training on 2/3 data...
- Validation on unseen 1/3 data :
- (0.914167916041979, 0.914167916041979, 0.914167916041979, None)

-                  ANT Precision  SYN Precision  ANT Recall  SYN Recall
- Noun Pairs            0.987013       0.883721    0.883721    0.987013
- Verb Pairs            0.982249       0.887574    0.897297    0.980392
- Adjective Pairs       0.975524       0.986547    0.989362    0.969163

-                    ANT F1    SYN F1
- Noun Pairs       0.932515  0.932515
- Verb Pairs       0.937853  0.931677
- Adjective Pairs  0.982394  0.977778