import os
from nltk.tokenize import word_tokenize
from nltk.tokenize.casual import TweetTokenizer, casual_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
from natsort import natsorted
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict

# Set up NLTK
import nltk
nltk.download('punkt')

# Function to read a file, tokenize its content, and apply stemming
def process_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        tokens = word_tokenize(content)

        # Apply stemming using Porter Stemmer
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in tokens]

        return stemmed_tokens

# Directory containing the text files
directory = "D:\docs"

# List to store tokenized content of each file
tokenized_documents = []


# Process each file in the directory
for filename in natsorted(os.listdir(directory)):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        tokens = process_file(file_path)
        tokenized_documents.append(tokens)


positional_index = {}
doc_num = 1

for doc in tokenized_documents:  # loop throught terms
    for index, term in enumerate(doc):
        if term in positional_index:
            positional_index[term][0] += 1  #increment frequency by 1
            if doc_num in positional_index[term][1]:
                positional_index[term][1][doc_num].append(index + 1)
            else:
                positional_index[term][1][doc_num] = [index + 1]
        else:
            positional_index[term] = [1, {doc_num: [index + 1]}]     # creation of positional index
    doc_num += 1

print(" \n Positional Index:\n")
print(positional_index)

###########################################################################

#calculating tf

all_words = []
for doc in tokenized_documents:
    for word in doc:
        all_words.append(word)


def Getfreq(doc):
    words = dict.fromkeys(all_words, 0)
    for word in doc:
        words[word] += 1
    return words

termFrequency = pd.DataFrame(Getfreq(tokenized_documents[0]).values(),index=Getfreq(tokenized_documents[0]).keys())

for i in range (1, len(tokenized_documents)):
    termFrequency[i]=Getfreq(tokenized_documents[i]).values()

termFrequency.columns=['DOC'+str(i) for i in range(1,11)]
print("\nTerm Frequency\n")
print(termFrequency)

########################################


# calculating weightedTF


def getWeightedTF(TF):
    if TF > 0:
        return math.log(TF) + 1
    return 0

for i in range(1, len(tokenized_documents) ):
    termFrequency['DOC'+str(i)] = termFrequency['DOC'+str(i)].apply(getWeightedTF)

print("\nWeighted Term Frequency\n")
print(termFrequency)


####################################################3
# idf and tf-idf
TF_IDF = pd.DataFrame(columns=['TF','IDF'])
for i in range(len(termFrequency)):
    Freq = int(termFrequency.iloc[i].values.sum())
    TF_IDF.loc[i,'TF'] = Freq
    TF_IDF.loc[i, 'IDF'] = math.log10(10 /(float(Freq)))

TF_IDF.index = termFrequency.index
print("\n TF and IDF \n")
print(TF_IDF) 

print("\n TF.IDF \n")
TfByIdf = termFrequency.multiply(TF_IDF['IDF'], axis=0)
print(TfByIdf)


#####################################
# doc len and norm

print("\n 	document length	\n")
document_length = pd.DataFrame()
def get_docs_length (col):
    return np.sqrt(TfByIdf[col].apply(lambda x: x**2).sum())
for column in TfByIdf.columns:
    document_length.loc[0,column+'_len'] = get_docs_length(column)
print(document_length)

# Normalization
print("\n 	Normalized tf.idf	\n")
normalized_term_freq_idf = pd.DataFrame()
def get_normalized(column):
    try:
        return TfByIdf[column] / document_length[column+'_len'].values[0]
    except:
        return 0
for column in TfByIdf.columns:
    normalized_term_freq_idf[column] = get_normalized(column)
print(normalized_term_freq_idf)
#######################
#phrase query

def query_stemming(phrase_query):
    # Tokenization
    Qtokens = word_tokenize(phrase_query)
    # Stemming using Porter Stemmer
    stemmer = PorterStemmer()
    stemmed_query = [stemmer.stem(token) for token in Qtokens]
    return stemmed_query


def phrase_query(query, stemmed_query, positional_index):
    if not set(stemmed_query).issubset(positional_index.keys()):
        return []

    matched_documents = OrderedDict()

    for i, word in enumerate(stemmed_query):
        for document_id, positions in positional_index[word][1].items():
            if document_id not in matched_documents:
                matched_documents[document_id] = []

            if i > 0 and matched_documents[document_id] and matched_documents[document_id][-1] != positions[0] - 1:
                continue

            matched_documents[document_id].extend(positions)

    matched_documents = {key: value for key, value in matched_documents.items() if len(value) == len(stemmed_query)}
    matched_documents = OrderedDict(sorted(matched_documents.items(), key=lambda item: item[1][0]))

    return matched_documents.keys()


query = input("\nPlease enter your query:\n")
stemmed_query = query_stemming(query)

#Execute phrase query and get matched documents
matched_documents = phrase_query(query, stemmed_query, positional_index)

def matchDoc(matched_documents):
    # Continue with the rest of the code only if there are matched documents
    if matched_documents:
        print("\nOutput of phrase query:\n")
        for document_id in matched_documents:
            print(f"doc {document_id}")
    else:
        print("No relevant documents found")
        return

#matchDoc(matched_documents)

print('######################################################')
print(' QUERY:  ', query)
count = len(stemmed_query)

def QueryTable(stemmed_query):
    queryy = pd.DataFrame(index=stemmed_query)  
    
    queryy['tf'] = [1 if x in stemmed_query else 0 for x in queryy.index]
    queryy['wtf'] = queryy['tf'].apply(lambda x: getWeightedTF(x))

    for word in stemmed_query:
        if word in normalized_term_freq_idf.index:
            # Calculate 'idf', 'tf_idf', and 'norm' for the current word
            queryy.loc[word, 'idf'] = TF_IDF.loc[word, 'IDF'] * queryy.loc[word, 'wtf']
            queryy.loc[word, 'tf_idf'] = queryy.loc[word, 'wtf'] * queryy.loc[word, 'idf']
            queryy['norm'] = 0
        else: 
            # Set 'idf', 'tf_idf', and 'norm' to zero for words not in normliazed_tf_idf.index
            queryy.loc[word, 'idf'] = 0
            queryy.loc[word, 'tf_idf'] = 0
            queryy.loc[word, 'norm'] = 0
    
    
    
    for i in range(len(queryy)):
        try:
            queryy.loc[:, 'norm'] = queryy['tf_idf'] / math.sqrt(sum(queryy['idf'].values**2))
        except:ZeroDivisionError

    # Print or return the resulting DataFrame
    print(queryy)
    return queryy 

queryy = QueryTable(stemmed_query)

def matchDoc2(matched_documents):
    # Continue with the rest of the code only if there are matched documents
    if matched_documents:

        product = normalized_term_freq_idf.multiply(queryy['wtf'], axis=0)
        product2 = product.multiply(queryy['norm'], axis=0)

        # calculate similarity score
        score = {}
        for col in product2.columns:
            if set(stemmed_query).issubset(normalized_term_freq_idf.index):  # Check if all query terms are in the index
                if 0 in product2[col].loc[stemmed_query].values:
                    pass
                else:
                    score[col] = product2[col].sum()
        else:
            # If all query terms are found in the index, proceed to calculate similarity scores
            keys = score.keys()

            print(product2[keys].loc[stemmed_query])
            product_results = product2[keys].loc[stemmed_query]
            print('sum : ')
            # print(product_results)
            print(product_results.sum())

            # Qlen:
            print('\n Q length:')
            print(math.sqrt(sum([x**2 for x in queryy['tf_idf'].loc[stemmed_query]])))

            # print similarity score
            for col, value in score.items():
                print(f"\n similarity(query, {col}) is {value} ")

            # sort documents in descending order
            final_score = sorted(score.items(), key=lambda x: x[1], reverse=True)
            for doc in final_score:
                print(doc[0], end=' ')
    else:
        print("No relevant documents found")
        return

matchDoc2(matched_documents)




