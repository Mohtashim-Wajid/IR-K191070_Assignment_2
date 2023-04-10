
import nltk 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

import glob
import os
import natsort




# -------------------- corpus Loading -------------------------

corpus_path = './Dataset/*.txt'  # change to your directory and file extension
doc_paths = natsort.natsorted(glob.glob(corpus_path))  # sort the file paths using natsorted

corpus_list = []  # initialize an empty list to store the document texts

# iterate over the document paths and read their contents into corpus_list as strings
for path in doc_paths:
    with open(path, 'r') as f:
        text = f.read()  # read the file contents as a string
        corpus_list.append(text)  # add the text to the corpus_list


vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(corpus_list)
print(vectorizer.get_feature_names_out())
print(x.shape)


nltk.download('punkit')
nltk.download('stopwords')

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
