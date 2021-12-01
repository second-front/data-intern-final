#!/usr/bin/env python
# coding: utf-8

# ## preprocess.ipynb
# ### post-webscrape, pre-model-training

# Before running running this notebook, database 'data' should contain the following collections from the webscraping process:  
# ```
# > use data  
# switched to db data  
# > show collections  
# companies  
# documents  
# failures  
# sublinks  
# ```
# If old collections exist:
# ```  
# > db.[OLD_COLLECTION].drop()  
# true
# ```  
# 

# #### Docker setup

# ```
# docker pull mongo  
# docker run -d -v [PWD]:/home \ 
# -p 27017-27019:27017-27019 \
# --name mongodb mongo:latest \ 
# docker exec -it mongodb bash
# ``` 

# #### Restore Webscrape MongoDB Dump

# ```
# cd [DUMP_DIR]  
# mongorestore -d data ./data
# ```

# #### Create Indexes to Improve Query Performance

# ```
# db.companies.createIndex({"id":1})
# db.documents.createIndex({"id":1})
# db.sublinks.createIndex({"id":1}) 
# ```

# ```json
# > db.companies.createIndex({"id":1})  
# {  
#         "numIndexesBefore" : 1,  
#         "numIndexesAfter" : 2,  
#         "createdCollectionAutomatically" : false,  
#         "ok" : 1  
# }  
# > db.documents.createIndex({"id":1})  
# {  
#         "numIndexesBefore" : 1,  
#         "numIndexesAfter" : 2,  
#         "createdCollectionAutomatically" : false,  
#         "ok" : 1  
# }  
# > db.sublinks.createIndex({"id":1})  
# {  
#         "numIndexesBefore" : 1,  
#         "numIndexesAfter" : 2,  
#         "createdCollectionAutomatically" : false,  
#         "ok" : 1  
# }
# ```

# In[1]:


import csv
#import nltk
import numpy as np
import regex as re
import concurrent.futures
import multiprocessing as mp
#import gensim.downloader as api
from nltk.corpus import wordnet
from pymongo import MongoClient
#from nltk.corpus import stopwords
from collections import defaultdict
#from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, _preprocess


# In[ ]:


DB_CONNECTION = 'mongodb://localhost:27017'
from headers import HEADERS, TARGET_URLS, DIRECT_SUB_DOMAINS, ENTRYPOINT_POSITIVES, CUSTOM_STOPWORDS
from headers import ID, NAME, DOMAIN, YEAR_FOUNDED, INDUSTRY, SIZE_RANGE, LOCALITY, COUNTRY, LINKEDIN_URL, EMPLOYEE_ESTIMATE, INDEX


# In[ ]:


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
model = api.load('glove-wiki-gigaword-50')
negatives = list(stopwords.words())
negatives.extend(CUSTOM_STOPWORDS)


# In[ ]:


# PREPROCESSING PARAMS
END_TOKEN = '>'
START_TOKEN = '<'
MIN_DOC_FREQ = 150 # minimum of documents that an extracted feature needs to appear in
MIN_FEATURES = 33 # minimum number of non-stop word features in order to qualify as training example
N_GRAM_LOWER_LIM = 1 # lower bound for count vectorizer's n-gram range
N_GRAM_UPPER_LIM = 2 # upper ..... ... ..... ............ ...... .....


# In[ ]:


def create_training_example(tokens, metadata):
    try:
        link_vectorizer = CountVectorizer(ngram_range=(N_GRAM_LOWER_LIM,N_GRAM_UPPER_LIM))
        X1 = link_vectorizer.fit_transform([tokens[0]])
        link_grams = link_vectorizer.get_feature_names()
        doc_vectorizer = CountVectorizer(ngram_range=(N_GRAM_LOWER_LIM,N_GRAM_UPPER_LIM))
        X2 = doc_vectorizer.fit_transform([tokens[1]])
        doc_grams = doc_vectorizer.get_feature_names()
        training_example = {
            'id': metadata[1],
            'label': metadata[0],
            'relevant': metadata[2],
            'link_array': X1.toarray().tolist(),
            'link_grams': link_grams,
            'doc_array': X2.toarray().tolist(),
            'doc_grams': doc_grams
        }
        client = MongoClient(DB_CONNECTION)
        if len(doc_grams) + len(link_grams) > MIN_FEATURES:
            client.data.train.insert_one(training_example)
        else:
            print('not enough tags, moving on...')
            client.data.companies.delete_one({'id': metadata[1]})
            client.data.sublinks.delete_many({'id': metadata[1]})
            client.data.documents.delete_many({'id': metadata[1]})
        return
    except ValueError:
        print('stop words only, moving on...')
    client = MongoClient(DB_CONNECTION)
    client.data.companies.delete_one({'id':metadata[1]})

    return


# Refactored cell below for gensim 4.0 update:\
# i.e. 
# ```python
# x in model.vw.vocab.keys() # new
# ```
# ```python
# x in embeddings.key_to_index() # old
# ```

# In[ ]:


def tokenize(raw_links):
    links = str(raw_links).strip('][')
    while '\'' in links:
        links = links.replace('\'', START_TOKEN + " ", 1)
        links = links.replace('\'', " " + END_TOKEN, 1)
    text = " ".join(links.split(', '))
    pattern = re.compile('[^a-zA-z\s<>]+', re.UNICODE)
    text = re.sub(pattern, ' ', text)
    text = re.sub('\s+', ' ', text)
    text = ' '.join([x for x in nltk.word_tokenize(text) if (x not in negatives) and (x in model.wv.vocab.keys()) and (len(x) > 3 or x == '<' or x == '>')])
    text = text.replace(START_TOKEN, '[S]')
    text = text.replace(END_TOKEN, '[E]')
    text = text.replace('[S]' + ' ' + '[E]', '')
    text = re.sub('\s+', ' ', text)
    return text


# In[ ]:


def preprocess(links_list, documents):
    hp_tokens = ''
    for links in links_list:
        hp_tokens = tokenize(links)
    lp_tokens = tokenize(documents)
    return (hp_tokens, lp_tokens)


# In[ ]:


def preprocessing_unit(links, documents, metadata):
    tokens = preprocess(links, documents)
    create_training_example(tokens, metadata)


# In[ ]:


def load(entry, order_id, max_id, client):
    #print('loading...', entry['id'])
    if order_id % 100 == 0: print(str(int(order_id/max_id*100))+'%')
    links = []
    documents = []
    #client = MongoClient(DB_CONNECTION)
    for link_list in client.data.sublinks.find({'id':entry['id']}):
        links.append(link_list['links'])
    for doc in client.data.documents.find({'id':entry['id']}):
        documents.append(doc['text'])
    #client.close()

    return (links, documents, entry)


# In[ ]:


def load_from_db():
    entry_list = []
    client = MongoClient(DB_CONNECTION)
    for entry in client.data.companies.find():
        entry_list.append(entry)
    #client.close()
    metadata = []
    links_list = []
    documents_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        future_to_load = {
            executor.submit(load, obj[1], obj[0], len(entry_list), client): \
                obj for obj in enumerate(entry_list)
        }
        for future in concurrent.futures.as_completed(future_to_load):
            res = future.result()
            links_list.append(res[0])
            documents_list.append(str(res[1]))
            metadata.append(tuple((res[2]['industry'], res[2]['id'], res[2]['relevant'])))
    client.close()
    return (links_list, documents_list, metadata)


# Function submit_for_processing() creates the db.train collection.

# In[ ]:


def submit_for_processing():
    links_list, documents_list, metadata = load_from_db()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_preprocess = {
            executor.submit(preprocessing_unit, obj[0], obj[1], obj[2]): \
                obj for obj in zip(links_list, documents_list, metadata)
        }


# In[ ]:


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'): return wordnet.ADJ
    elif nltk_tag.startswith('V'): return wordnet.VERB
    elif nltk_tag.startswith('N'): return wordnet.NOUN
    elif nltk_tag.startswith('R'): return wordnet.ADV
    else: return None


# In[ ]:


def stem(lemma,w,p):
    if p is not None: return lemma.lemmatize(w,p)
    else: return lemma.lemmatize(w)


# In[ ]:


"""def vocab_work_unit(entry_id, order_id, max_id, lemma, client):
    if order_id % 100 == 0: print(str(int(order_id/max_id*100))+'%')
    entry = client.data.train.find_one({'id':entry_id})
    #lemma = WordNetLemmatizer()
    doc_res = [n_gram.strip().split() for n_gram in entry['doc_grams']]
    link_res = [n_gram.strip().split() for n_gram in entry['link_grams']]
    # LEMMATIZE ACCORDING TO PART-OF-SPEECH ESTIMATE
    stemmed_d = [" ".join([stem(lemma,w,pos_tagger(p)) for w,p in elem]) for elem in list(map(nltk.pos_tag, doc_res))]
    stemmed_l = [" ".join([stem(lemma,w,pos_tagger(p)) for w,p in elem]) for elem in list(map(nltk.pos_tag, link_res))]
    combined_grams = stemmed_d + stemmed_l
    combined_freqs = []
    combined_freqs.extend(entry['doc_array'][0])
    combined_freqs.extend(entry['link_array'][0])
    document = {
        'id': entry['id'],
        'label': entry['relevant'],
        'combined_freqs': combined_freqs,
        'combined_grams': combined_grams
    }
    #client = MongoClient(DB_CONNECTION)
    client.data.norm.stemmed.insert_one(document)
    #client.close()
    return (combined_grams,entry['id'])"""


# In[ ]:


"""def create_vocabulary(out_file):
    corpus = set()
    lemma = WordNetLemmatizer()
    document_freq = defaultdict(int)
    client = MongoClient(DB_CONNECTION)
    entries = [entry['id'] for entry in client.data.train.find()]
    train_set = set(np.random.choice(entries, size=int(len(entries)*0.9), replace=False))
    print(len(train_set))
    print(len(entries))
    #entry_list = []
    # CREATE LEMMATIZED TRAINING EXAMPLE & INSERT INTO DB.NORM.STEMMED COLLECTION
    #for entry in client.data.train.find(): entry_list.append(entry)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_lemma = {
            executor.submit(vocab_work_unit, obj[1], obj[0], len(entries), lemma, client): \
                obj for obj in enumerate(entries)
        }
        for future in concurrent.futures.as_completed(future_to_lemma):
            res = future.result()
            # DO NOT INCLUDE FEATURES EXTRACTED FROM VALIDATION SET EXAMPLES INTO VOCABULARY
            # VALIDATION SET NEEDS TO BE 'UNSEEN'
            if res[1] in train_set:
                corpus.update(res[0])
            for elem in res[0]:
                document_freq[elem] += 1
    # CREATE ENUMERATED VOCABULARY
    corpus_subset = {val for val in corpus if document_freq[val] > MIN_DOC_FREQ}
    vocabulary = {item:val for val,item in enumerate(corpus_subset)}
    # EXPORT VOCABULARY
    with open(out_file, 'a') as f:
        for key,val in vocabulary.items():
            f.write(str(val)+','+str(key)+'\n')
        f.close()
    # CREATE DENSE TRAINING EXAMPLE
    for entry in client.data.norm.stemmed.find():
        features = [(vocabulary[a],b) for a,b in zip(entry['combined_grams'], entry['combined_freqs']) if a in vocabulary.keys()]
        dense_example = {
            'id': entry['id'],
            'label': entry['label'],
            'features': features
        }
        if entry['id'] in train_set:
            client.data.norm.train.vectorized.insert_one(dense_example)
        else:
            client.data.norm.test.vectorized.insert_one(dense_example)
    client.close()
    print(dense_example)"""


# In[ ]:


def create_vocabulary(out_file):
    corpus = set()
    lemma = WordNetLemmatizer()
    document_freq = defaultdict(int)
    client = MongoClient(DB_CONNECTION)
    entries = [entry['id'] for entry in client.data.train.find()]
    train_set = set(np.random.choice(entries, size=int(len(entries)*0.9), replace=False))
    print(len(train_set))
    max_id = len(entries)
    print(max_id)
    # CREATE LEMMATIZED TRAINING EXAMPLES & INSERT INTO DB.NORM.STEMMED COLLECTION
    for order_id, entry in enumerate(client.data.train.find()):
        if order_id % 100 == 0: print(str(int(order_id/max_id*100))+'%')
        entry = client.data.train.find_one({'id':entry['id']})
        doc_res = [n_gram.strip().split() for n_gram in entry['doc_grams']]
        link_res = [n_gram.strip().split() for n_gram in entry['link_grams']]
        # LEMMATIZE ACCORDING TO PART-OF-SPEECH ESTIMATE
        stemmed_d = [" ".join([stem(lemma,w,pos_tagger(p)) for w,p in elem]) for elem in list(map(nltk.pos_tag, doc_res))]
        stemmed_l = [" ".join([stem(lemma,w,pos_tagger(p)) for w,p in elem]) for elem in list(map(nltk.pos_tag, link_res))]
        combined_grams = stemmed_d + stemmed_l
        combined_freqs = []
        combined_freqs.extend(entry['doc_array'][0])
        combined_freqs.extend(entry['link_array'][0])
        document = {
            'id': entry['id'],
            'label': entry['relevant'],
            'combined_freqs': combined_freqs,
            'combined_grams': combined_grams
        }
        client.data.norm.stemmed2.insert_one(document)
        # DO NOT INCLUDE FEATURES EXTRACTED FROM VALIDATION SET EXAMPLES INTO VOCABULARY
        # VALIDATION SET NEEDS TO BE 'UNSEEN'
        if entry['id'] in train_set:
            corpus.update(combined_grams)
        for elem in combined_grams:
            document_freq[elem] += 1
    # CREATE ENUMERATED VOCABULARY
    corpus_subset = {val for val in corpus if document_freq[val] > MIN_DOC_FREQ}
    vocabulary = {item:val for val,item in enumerate(corpus_subset)}
    # EXPORT VOCABULARY
    with open(out_file, 'a') as f:
        for key,val in vocabulary.items():
            f.write(str(val)+','+str(key)+'\n')
        f.close()
    # CREATE DENSE TRAINING EXAMPLE
    for entry in client.data.norm.stemmed2.find():
        features = [(vocabulary[a],b) for a,b in zip(entry['combined_grams'], entry['combined_freqs']) if a in vocabulary.keys()]
        dense_example = {
            'id': entry['id'],
            'label': entry['label'],
            'features': features
        }
        if entry['id'] in train_set:
            client.data.norm.train2.vectorized.insert_one(dense_example)
        else:
            client.data.norm.test2.vectorized.insert_one(dense_example)
    client.close()
    print(dense_example)


# In[ ]:


def vectorize_unseen_data(vocab_src):
    vocabulary = {}
    lemma = WordNetLemmatizer()
    client = MongoClient(DB_CONNECTION)
    #entries = [entry['id'] for entry in client.data.train.find()]
    #max_id = len(entries)
    """for order_id, entry in enumerate(client.data.train.find()):
        if order_id % 100 == 0: print(str(int(order_id/max_id*100))+'%')
        entry = client.data.train.find_one({'id':entry['id']})
        doc_res = [n_gram.strip().split() for n_gram in entry['doc_grams']]
        link_res = [n_gram.strip().split() for n_gram in entry['link_grams']]
        # LEMMATIZE ACCORDING TO PART-OF-SPEECH ESTIMATE
        stemmed_d = [" ".join([stem(lemma,w,pos_tagger(p)) for w,p in elem]) for elem in list(map(nltk.pos_tag, doc_res))]
        stemmed_l = [" ".join([stem(lemma,w,pos_tagger(p)) for w,p in elem]) for elem in list(map(nltk.pos_tag, link_res))]
        combined_grams = stemmed_d + stemmed_l
        combined_freqs = []
        combined_freqs.extend(entry['doc_array'][0])
        combined_freqs.extend(entry['link_array'][0])
        document = {
            'id': entry['id'],
            'label': entry['relevant'],
            'combined_freqs': combined_freqs,
            'combined_grams': combined_grams
        }
        client.data.norm.stemmed.insert_one(document)"""
    with open(vocab_src) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for r in reader:
            vocabulary[r[1]] = int(r[0])
    #print(vocabulary)
    # CREATE DENSE TRAINING EXAMPLE
    for entry in client.data.norm.stemmed.find():
        print(entry)
        features = [(vocabulary[a],b) for a,b in zip(entry['combined_grams'], entry['combined_freqs']) if a in vocabulary.keys()]
        dense_example = {
            'id': entry['id'],
            'label': entry['label'],
            'features': features
        }
        print(dense_example)
        client.data.norm.unseen.vectorized.insert_one(dense_example)
    #print(dense_example)
    client.close()


# After running below cell, database 'data' should contain the following collections:
# ```
# > show collections
# companies
# documents
# failures
# norm.stemmed # new
# norm.test.vectorized # new
# norm.train.vectorized # new
# sublinks
# train # new
# ```

# ### Execution Cells

# In[ ]:


#submit_for_processing()


# #### Create db.train index

# ```json
# > db.train.createIndex({"id":1})  
# {  
#         "numIndexesBefore" : 1,  
#         "numIndexesAfter" : 2,  
#         "createdCollectionAutomatically" : false,  
#         "ok" : 1  
# }  
# ```

# In[ ]:



#create_vocabulary(out_file='vocabulary_100k_docfreq_100_hq.csv')
vectorize_unseen_data(vocab_src='./vocabularies/vocabulary_100k_docfreq_100_hq.csv')


# ### Exporting MongoDB database
# #### in container: 
# ```bash
# mongodump --db data -o ./[DB_DUMP_DIR_NAME] # database
# mongodump --db data --collection [COLLECTION] -out [COLL_DUMP_DIR_NAME] # collection
# ```
# #### in host: 
# ```
# docker cp [CONTAINER_ID]:[DUMP_DIR_NAME] .  
# ```

# 
