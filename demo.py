import pickle
from scrape import webscrape
from preprocess import submit_for_processing, vectorize_unseen_data
from predict import predict_unseen

if __name__ == '__main__':
    webscrape(num_enqueues=0,src_file='./sam_entities.json')
    submit_for_processing()
    #vectorize_unseen_data(vocab_src="./vocabularies/vocabulary_100k_docfreq_100_hq.csv")
    #loaded_model = pickle.load(open('pima.pickle.dat', 'rb'))
    #predict_unseen(loaded_model)
