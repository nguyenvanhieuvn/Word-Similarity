# task of measure similarity between two words

BÀI TẬP ĐO LƯỜNG ĐỘ TƯƠNG TỰ CỦA TỰ CỦA TỪ (WORD SIMILARITY) DỰA TRÊN WORD EMBEDDINGS

Sinh viên đăng ký project theo một trong ba mức sau:

STANDARD
- Viết chương trình đo Word Similarity sử dụng pre-trained word embeddings (thư mục word2vec) và bộ dữ liệu VSim400 (thư mục Datasets/ViSim-400).

MEDIUM
- Tìm hiểu Gensim để training Word2Vec, sử dụng word embeddings đã học được để đo Word Similarity bộ dữ liệu VSim400 (thư mục Datasets/ViSim-400).

HARD
1. Thực nghiệm và so sánh kết quả giữa hướng tiếp cận dựa trên co-occurrence counts vectors với word embeddings vectors
 + Trích xuất ma trận đồng xuất hiện word-context matrix (Extracting co-occurrence counts)
 + Lượng giá trọng số (Pointwise mutual information- PMI)
 + Dimensionality reduction (single value decomposition)
 + Áp dụng một số độ đo khảng cách giữa 2 vector (cosine,...) để lượng giá độ tương tự
 + So sánh kết quả thực nghiệm giữa kỹ thuật dựa trên co-occurrence counts vectors với word embeddings vectors.
 
 2. Sử dụng WordNet làm tăng tính phân tách quan hệ Synonym, Antonym cho không gian Word Embeddings (tham khảo: Semantic Specialisation of Distributional Word Vector Spaces using
Monolingual and Cross-Lingual Constraints, Nikola Mrkšic´1;2, Ivan Vulic´1, Diarmuid Ó Séaghdha2, Ira Leviant3, Roi Reichart3, Milica Gašic´1, Anna Korhonen1, Steve Young1;2
1, 2017)


