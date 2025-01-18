import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np


file_path = '/Users/chenjianheng/Desktop/DM HW 2/Reviews.csv'
df = pd.read_csv(file_path, usecols=['Text', 'Score'], nrows=10000)

# 將 'Score' 轉換為二元分類
df['Score'] = df['Score'].apply(lambda x: 1 if x >= 4 else 0)

# 將 'Text' 分詞（基於標點符號進行分割）
df['Text'] = df['Text'].apply(lambda x: re.findall(r'\b\w+\b', x.lower()))

# 移除停用詞並進行向量化
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Text'].apply(lambda x: ' '.join(x)))

# 使用 TF-IDF 向量化
tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(df['Text'].apply(lambda x: ' '.join(x)))

# 訓練 Word2Vec 模型
sentences = df['Text'].apply(lambda x: x)
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)

# 將文字轉換為向量並做平均
def vectorize_text(text, model):
    vectors = [model.wv[word] for word in text if word in model.wv]
    if vectors:
        return sum(vectors) / len(vectors)
    else:
        return [0] * model.vector_size

X_word2vec = df['Text'].apply(lambda x: vectorize_text(x, word2vec_model))

# 確保 Word2Vec 的結果為數字資料，並轉換為適合模型使用的矩陣形式
X_word2vec = np.array(X_word2vec.tolist())

# 分類
clf = RandomForestClassifier()

# 使用 TF-IDF 進行 k=4 的交叉驗證
cv_scores_tfidf = cross_val_score(clf, X_tfidf, df['Score'], cv=4, scoring='accuracy')
print("TF-IDF 4-Fold 交叉驗證準確度: {:.4f}".format(cv_scores_tfidf.mean()))

# 使用 Word2Vec 進行 k=4 的交叉驗證
cv_scores_word2vec = cross_val_score(clf, X_word2vec, df['Score'], cv=4, scoring='accuracy')
print("Word2Vec 4-Fold 交叉驗證準確度: {:.4f}".format(cv_scores_word2vec.mean()))