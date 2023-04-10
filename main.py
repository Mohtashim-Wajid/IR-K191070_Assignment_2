
import nltk 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import glob
import os
import natsort
import pandas as pd




# -------------------- corpus Loading -------------------------

corpus_path = './Dataset/*.txt'  # change to your directory and file extension
doc_paths = natsort.natsorted(glob.glob(corpus_path))  # sort the file paths using natsorted

corpus_list = []  # initialize an empty list to store the document texts

# iterate over the document paths and read their contents into corpus_list as strings
for path in doc_paths:
    with open(path, 'r') as f:
        text = f.read()  # read the file contents as a string
        corpus_list.append(text)  # add the text to the corpus_list


# vectorizer = TfidfVectorizer()
# x = vectorizer.fit_transform(corpus_list)
# print(vectorizer.get_feature_names_out())
# print(x.shape)


# nltk.download('punkt')
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))



# ----------------- PREPROCESSING -----------------

# --------------------- tokenization ------------------- 
def get_tokenized_list(doc_text):
    tokens = nltk.word_tokenize(doc_text)
    return tokens


def word_stemmmer(token_list):
    ps = nltk.stem.PorterStemmer()
    stemmed = []
    for words in token_list:
        stemmed.append(ps.stem(words))
    return stemmed

def remove_stopwords(doc_text):
    cleaned_text = []
    for words in doc_text:
        if words not in stop_words:
            cleaned_text.append(words)
    return cleaned_text


cleaned_corpus = []
for doc in corpus_list:
    tokens = get_tokenized_list(doc)
    doc_text = remove_stopwords(tokens)
    doc_text = word_stemmmer(doc_text)
    doc_text = ' '.join(doc_text)
    cleaned_corpus.append(doc_text)
# print(cleaned_corpus)



vectorizerX = TfidfVectorizer()
vectorizerX.fit(cleaned_corpus)
doc_vector = vectorizerX.transform(cleaned_corpus)
print(vectorizerX.get_feature_names_out())
print(doc_vector.shape)

dataframe = pd.DataFrame(doc_vector.toarray(), columns= vectorizerX.get_feature_names_out())
# print(dataframe)


# ------------------------------------ Query Processing -------------------------



#                            modify the query here to search ....
query = 'andrew pakistan'



query = get_tokenized_list(query)
query = remove_stopwords(query)
q = []
for w in word_stemmmer(query):
    q.append(w)
q = ' '.join(q)

query_vector = vectorizerX.transform([q]) 
cosineSimilarity = cosine_similarity(doc_vector,query_vector).flatten()
related_docs_indices = cosineSimilarity.argsort()[:-10:-1]
print(related_docs_indices)

