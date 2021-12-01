import sys
import time
import pickle
import numpy as np
import xgboost as xgb
import matplotlib.pyplot  as plt
from pymongo import MongoClient
from scipy.sparse import csr_matrix
from scipy import sparse

DB_CONNECTION = 'mongodb://localhost:27017'

def train_xgb_model(vocabulary_src):
    with open(vocabulary_src) as f:
        for line in f: pass
        last_line = line
        last_id = last_line.split(',')[0]
    max_id = int(last_id)
    client = MongoClient(DB_CONNECTION)
    training_examples = []
    training_labels = []
    validation_examples = []
    validation_labels = []
    counter = 0
    for example in client.data.norm.train.vectorized.find():
        v = np.zeros((max_id+1,), dtype=float)
        for id,val in example['features']: v[id] = val # X_j^(i)
        v = np.divide(v,np.max(v))
        training_examples.append(csr_matrix(v))
        training_labels.append(int(example['label']))
        counter += 1
        print(counter)
    X_train = sparse.vstack(training_examples)
    Y_train = np.asarray(training_labels)
    dtrain = xgb.DMatrix(X_train,label=Y_train)
    print('done querying train...')
    counter = 0
    for example in client.data.norm.test.vectorized.find():
        v = np.zeros((max_id+1,), dtype=float)
        for id,val in example['features']: v[id] = val # X_j^(i)
        v = np.divide(v,np.max(v))
        validation_examples.append(csr_matrix(v))
        validation_labels.append(int(example['label']))
        counter += 1
        print(counter)
    X_test = sparse.vstack(validation_examples)
    Y_test = np.asarray(validation_labels)
    dtest = xgb.DMatrix(X_test,label=Y_test)
    client.close()
    eval_metrics = ['error', 'logloss']
    eval_set = [(X_train, Y_train),(X_test, Y_test)]
    model = xgb.XGBClassifier(objective='binary:logistic',n_estimators=1500,max_depth=6,learning_rate=0.1)
    model.fit(X_train,Y_train,eval_set=eval_set,eval_metric=eval_metrics,verbose=True)
    pickle.dump(model, open("pima.pickle.dat", "wb"))
    res = model.evals_result()
    """
    print('done querying test...')
    params = {
        'objective': 'binary:logistic', # Specify binary classification
        'tree_method': 'gpu_hist', # Use GPU accelerated algorithm
        'eval_metric': ['error', 'logloss'],
        'num_boost_round': 100,
        'early_stopping_rounds': 10
    }
    gpu_res = {}
    tmp = time.time()
    model = xgb.train(params,dtrain,evals=[(dtrain,'train'),(dtest,'test')],evals_result=gpu_res,verbose_eval=True)
    pickle.dump(model, open("pima.pickle.dat", "wb"))
    print("GPU Training Time: %s seconds" % (str(time.time() - tmp)))
    print(gpu_res)
    return gpu_res
    """
    return res

def plot_loss(results_dict, output_path, show_plot):    
    epochs = len(results_dict['validation_0']['error'])
    x_axis = range(0, epochs)
    # plot log loss
    fig = plt.figure(figsize=(7,5))
    plt.plot(x_axis, results_dict['validation_0']['logloss'], label='Train')
    plt.plot(x_axis, results_dict['validation_1']['logloss'], label='Test')
    plt.legend()
    plt.ylabel('Log Loss')
    plt.xlabel('Epochs')
    plt.title('Log-Loss')
    plt.savefig(output_path)
    if show_plot==True:
        plt.show()

def plot_error(results_dict, output_path, show_plot):        
    # plot classification error
    epochs = len(results_dict['validation_0']['error'])
    x_axis = range(0, epochs)
    # plot error
    fig = plt.figure(figsize=(7,5))
    plt.plot(x_axis, results_dict['validation_0']['error'], label='Train')
    plt.plot(x_axis, results_dict['validation_1']['error'], label='Test')
    plt.legend()
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    plt.title('Classification Error')
    plt.savefig(output_path)
    if show_plot==True:
        plt.show()

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
    for example in client.data.norm.test.vectorized.find():
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
    with open("./vocabularies/vocabulary_100k_docfreq_100_hq.csv") as f:
        for line in f: pass
        last_line = line
        last_id = last_line.split(',')[0]
    max_id = int(last_id)
    counter = 1
    for example in client.data.norm.test.vectorized.find():
        v = np.zeros((max_id+1,), dtype=float)
        for id,val in example['features']: v[id] = val # X_j^(i)
        v = np.divide(v,np.max(v))
        validation_examples.append(csr_matrix(v))
        counter += 1
        print(counter)
    X_test = sparse.vstack(validation_examples)
    Y_pred = model.predict(X_test)
    print(Y_pred)

if __name__ == '__main__':
    #res = train_xgb_model(vocabulary_src='./vocabularies/vocabulary_100k_docfreq_100_hq.csv')
    loaded_model = pickle.load(open('pima.pickle.dat', "rb"))
    print(loaded_model)
    res = loaded_model.evals_result()
    print(res)
    plot_loss(res, output_path="log-loss-init.png", show_plot=True)
    plot_error(res, output_path="error-init.png", show_plot=True)
    predict(loaded_model)
