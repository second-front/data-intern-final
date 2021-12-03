import pickle

if __name__ == '__main__':
    #from scrape import webscrape
    #webscrape(num_enqueues=5,src_file='./sam_entities.json')

    from preprocess import submit_for_processing, vectorize_unseen_data
    submit_for_processing()
    #vectorize_unseen_data(vocab_src="./vocabularies/vocabulary_100k_docfreq_100_hq.csv")

    #from predict import predict_unseen
    #loaded_model = pickle.load(open('pima.pickle.dat', 'rb'))
    #predict_unseen(loaded_model)
