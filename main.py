import sys
import time
import pickle
import numpy as np
import xgboost as xgb
import matplotlib.pyplot  as plt
from pymongo import MongoClient

DB_CONNECTION = 'mongodb://localhost:27017'

def train_xgb_model(vocabulary_src):
    with open(vocabulary_src) as f:
        for line in f: pass
        last_line = line
        last_id = last_line.split(',')[0]
    max_id = int(last_id)
    client = MongoClient(DB_CONNECTION)
    training_examples = []
    validation_examples = []
    for example in client.data.norm.train.vectorized.find():
        v = np.zeros((max_id+2,), dtype=int)
        for id,val in example['features']: v[id] = val # X_j^(i)
        v[max_id+1] = int(example['label']) # Y^(i)
        training_examples.append(v)
        print(sys.getsizeof(training_examples))
    D = np.asarray(training_examples,dtype=int)
    X_train = D[:,0:max_id+1]
    Y_train = D[:,max_id+1]
    print('done querying train...')
    for example in client.data.norm.test.vectorized.find():
        v = np.zeros((max_id+2,), dtype=int)
        for id,val in example['features']: v[id] = val # X_j^(i)
        v[max_id+1] = int(example['label']) # Y^(i)
        validation_examples.append(v)
    D = np.asarray(validation_examples,dtype=int)
    X_test = D[:,0:max_id+1]
    Y_test = D[:,max_id+1]
    client.close()
    print('done querying test...')
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)
    param = {
        'objective': 'binary:logistic', # Specify binary classification
        'tree_method': 'gpu_hist', # Use GPU accelerated algorithm
        'eval_metric': ['error', 'logloss']
    }
    gpu_res = {}
    tmp = time.time()
    model = xgb.train(param,dtrain,evals=[(dtrain,'train'),(dtest,'test')],evals_result=gpu_res,verbose_eval=True)
    pickle.dump(model, open("pima.pickle.dat", "wb"))
    print("GPU Training Time: %s seconds" % (str(time.time() - tmp)))
    print(gpu_res)
    return gpu_res

def plot_loss(results_dict, output_path, show_plot):    
    epochs = len(results_dict['train']['error'])
    x_axis = range(0, epochs)
    # plot log loss
    fig = plt.figure(figsize=(7,5))
    plt.plot(x_axis, results_dict['train']['logloss'], label='Train')
    plt.plot(x_axis, results_dict['test']['logloss'], label='Test')
    plt.legend()
    plt.ylabel('Log Loss')
    plt.xlabel('Epochs')
    plt.title('Log-Loss')
    plt.savefig(output_path)
    if show_plot==True:
        plt.show()

def plot_error(results_dict, output_path, show_plot):        
    # plot classification error
    epochs = len(results_dict['train']['error'])
    x_axis = range(0, epochs)
    # plot error
    fig = plt.figure(figsize=(7,5))
    plt.plot(x_axis, results_dict['train']['error'], label='Train')
    plt.plot(x_axis, results_dict['test']['error'], label='Test')
    plt.legend()
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    plt.title('Classification Error')
    plt.savefig(output_path)
    if show_plot==True:
        plt.show()

def extract_split(example_list, max_id):
    data = []
    for example in example_list[:int(len(example_list))]:
        v = np.zeros((max_id+2,), dtype=np.int64)
        for id,val in example['features']: v[id] = val # X_j^(i)
        v[max_id+1] = int(example['label']) # Y^(i)
        data.append(v)
    D = np.asarray(data,dtype=np.int64)
    X = D[:,0:max_id+1]
    Y = D[:,max_id+1]
    return (X,Y)

def predict(model_src):
    client = MongoClient(DB_CONNECTION)
    validation_examples = []
    for example in client.data.norm.test.vectorized.find():
        validation_examples.append(example)
    with open("./vocabularies/vocabulary_100k_docfreq_100_hq.csv") as f:
        for line in f: pass
        last_line = line
        id = last_line.split(',')[0]
    loaded_model = pickle.load(open(model_src, "rb"))
    X_test, Y_test = extract_split(example_list=validation_examples,max_id=int(id))
    dtest = xgb.DMatrix(X_test, label=Y_test)
    Y_pred = loaded_model.predict(dtest)
    print(Y_pred)
    print(Y_test)
    accuracy = sum([1/len(Y_pred) * (round(i) == j) for i,j in zip(Y_pred,Y_test)])
    print(accuracy)

if __name__ == '__main__':
    res = train_xgb_model(vocabulary_src='./vocabularies/vocabulary_100k_docfreq_100_hq.csv')
    plot_loss(res, output_path="log-loss-init.jpg", show_plot=False)
    plot_error(res, output_path="error-init.jpg", show_plot=False)
    predict('pima.pickle.dat')

