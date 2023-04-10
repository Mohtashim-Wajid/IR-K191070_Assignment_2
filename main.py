from sklearn.feature_extraction.text import TfidfVectorizer

import glob
import os
import natsort

# corpus_path = './Dataset/'  # change to your directory and file extension
# doc_files = natsorted(os.listdir(corpus_path))  # get a sorted list of file names in the directory

# doc_list = []  # initialize an empty list to store the document texts

# # iterate over the document files and read their contents into doc_list as nested lists
# for file in doc_files:
#     with open(os.path.join(corpus_path, file), 'r') as f:
#         text = f.read()  # read the file contents as a string
#         doc_list.append([text])  # add the text to a new sublist in the doc_list

# # iterate over the doc_list and do something with each sublist


# # for sublist in doc_list:
# #     print(sublist)  # print the current sublist




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