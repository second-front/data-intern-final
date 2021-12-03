import numpy as np
from scipy import sparse
from pymongo import MongoClient
from scipy.sparse import csr_matrix

DB_CONNECTION = 'mongodb://localhost:27017'

def predict(model):
    client = MongoClient(DB_CONNECTION)
    validation_examples = []
    validation_labels = []
    with open("./vocabularies/vocabulary_100k_docfreq_100_hq.csv") as f:
        for line in f: pass
        last_line = line
        last_id = last_line.split(',')[0]
    max_id = int(last_id)
    counter = 1
    for example in client.kyle.norm.test.vectorized.find():
        v = np.zeros((max_id+1,), dtype=float)
        for id,val in example['features']: v[id] = val # X_j^(i)
        v = np.divide(v,np.max(v))
        validation_examples.append(csr_matrix(v))
        validation_labels.append(int(example['label']))
        counter += 1
        print(counter)
    X_test = sparse.vstack(validation_examples)
    Y_test = np.asarray(validation_labels)
    Y_pred = model.predict(X_test)
    print(Y_pred)
    print(Y_test)
    accuracy = sum([1/len(Y_pred) * (round(i) == j) for i,j in zip(Y_pred,Y_test)])
    print(accuracy)

def predict_unseen(model):
    client = MongoClient(DB_CONNECTION)
    validation_examples = []
    ids = []
    with open("./vocabularies/vocabulary_100k_docfreq_100_hq.csv") as f:
        for line in f: pass
        last_line = line
        last_id = last_line.split(',')[0]
    max_id = int(last_id)
    counter = 1
    for example in client.kyle.norm.unseen.vectorized.find():
        v = np.zeros((max_id+1,), dtype=float)
        for id,val in example['features']: v[id] = val # X_j^(i)
        v = np.divide(v,np.max(v))
        cmpny = client.kyle.companies.find_one({'id':example['id']})
        if cmpny:
            validation_examples.append(csr_matrix(v))
            ids.append(tuple((example['id'],cmpny['name'])))
            counter += 1
            print(counter)
        if counter > 1000: break
    X_test = sparse.vstack(validation_examples)
    Y_pred = model.predict(X_test)
    positives = [tuple((a,b)) for a,b in zip(Y_pred,ids) if a == 1]
    print(positives)
    print(len(positives)/len(ids))