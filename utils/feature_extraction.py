from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.models.doc2vec import LabeledSentence
import gensim
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

#count vectorizer
def get_count_vector(x_train, x_test, ngram_range=None, min_df=0.0002, remove_stopwords=False):
    cv = CountVectorizer(min_df=min_df)

    if ngram_range != None:
        cv.ngram_range = ngram_range

    if remove_stopwords:
        cv.stop_words = 'english'

    cv.fit(x_train)
    x_train_vector = cv.transform(x_train)
    x_test_vector = cv.transform(x_test)

    return cv.vocabulary_, x_train_vector, x_test_vector

def get_tfidf_vector(x_train, x_test, remove_stopwords=False, ngram_range = None):
    
    tfidf = TfidfVectorizer(min_df=0.0002)
    
    if remove_stopwords:
        tfidf.stop_words = 'english'
        
    if ngram_range != None:
        tfidf.ngram_range = ngram_range
        
    tfidf.fit(x_train)
    x_train_vector = tfidf.transform(x_train)
    x_test_vector = tfidf.transform(x_test)
    
    return tfidf.vocabulary_, x_train_vector, x_test_vector

#helper function for word2vec
def word_vector(model_w2v, tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model_w2v.wv[word].reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary
            continue
    if count != 0:
        vec /= count
    return vec

def create_word2vec_model(x_train, x_test):
    x_train_tokenized = x_train.apply(lambda x: x.split()) 
    x_test_tokenized = x_test.apply(lambda x: x.split())
    
    x_train_len = len(x_train)
    x_test_len = len(x_test)
    
    model_w2v = gensim.models.Word2Vec(
            x_train_tokenized,
            vector_size=200, # desired no. of features/independent variables
            window=5, # context window size
            min_count=10, # Ignores all words with total frequency lower than 10.                                  
            sg = 1, # 1 for skip-gram model
            hs = 0,
            negative = 10, # for negative sampling
            workers= 32, # no.of cores
            seed = 34
    )
    
    model_w2v.train(x_train_tokenized, total_examples= x_train_len, epochs=20)
    
    return model_w2v
 
def get_word2vec_embedding(model, x_train, x_test):
    x_train_tokenized = x_train.apply(lambda x: x.split()) 
    x_test_tokenized = x_test.apply(lambda x: x.split())
    
    x_train_len = len(x_train)
    x_test_len = len(x_test)
    
    x_train_vector = np.zeros((x_train_len, 200))
    for i in range(x_train_len):
        x_train_vector[i,:] = word_vector(model, x_train_tokenized[i], 200)
    x_train_vector = pd.DataFrame(x_train_vector)
    
    x_test_vector = np.zeros((x_test_len, 200))
    for i in range(x_test_len):
        x_test_vector[i,:] = word_vector(model, x_test_tokenized[i], 200)
    x_test_vector = pd.DataFrame(x_test_vector)
    
    return x_train_vector, x_test_vector

#helper functions for doc2vec
def add_label(twt):
    twt = twt.apply(lambda x: x.split())

    output = []
    for i, s in zip(twt.index, twt):
        output.append(LabeledSentence(s, ["tweet_" + str(i)]))
    return output

def create_doc2vec_model(x_train):
    labeled_x_train = add_label(x_train)

    model_d2v = gensim.models.Doc2Vec(dm=1,  # dm = 1 for ‘distributed memory’ model
                                      dm_mean=1,  # dm_mean = 1 for using mean of the context word vectors
                                      vector_size=200,  # no. of desired features
                                      window=5,  # width of the context window
                                      negative=7,  # if > 0 then negative sampling will be used
                                      min_count=5,
                                      # Ignores all words with total frequency lower than 5.
                                      workers=32,  # no. of cores
                                      alpha=0.1,  # learning rate
                                      seed=23,  # for reproducibility
                                      )

    model_d2v.build_vocab([i for i in tqdm(labeled_x_train)])

    model_d2v.train(labeled_x_train, total_examples=len(x_train), epochs=15)

    x_train_tokenized = x_train.apply(lambda x: x.split())
    docvec_arrays = np.zeros((len(x_train_tokenized), 200))
    for i in range(len(x_train)): #combi
        docvec_arrays[i, :] = model_d2v.docvecs[i].reshape((1, 200))

    docvec_df = pd.DataFrame(docvec_arrays)

#helper function for glove
def glove_word_vector(model, tokens, size):
    
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model.get_vector(word)
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary
            continue
    if count != 0:
        vec /= count
    return vec

def load_glove_model(filepath, outputfile):
    glove2word2vec(filepath, outputfile)
    
    model = KeyedVectors.load_word2vec_format(outputfile, binary=False)
    
    return model

def get_glove_embedding(model, x_train, x_test):
    x_train_tokenized = x_train.apply(lambda x: x.split()) 
    x_test_tokenized = x_test.apply(lambda x: x.split())
    
    x_train_len = len(x_train)
    x_test_len = len(x_test)
    
    x_train_vector = np.zeros((x_train_len, 200))
    for i in range(x_train_len):
        x_train_vector[i,:] = glove_word_vector(model, x_train_tokenized[i], 200)
    x_train_vector = pd.DataFrame(x_train_vector)
    
    x_test_vector = np.zeros((x_test_len, 200))
    for i in range(x_test_len):
        x_test_vector[i,:] = glove_word_vector(model, x_test_tokenized[i], 200)
    x_test_vector = pd.DataFrame(x_test_vector)
    
    return x_train_vector, x_test_vector

#BILSTM

def fit_transform_word(model, data):
    # determine the dimensionality of vectors
    dim = model.get_vector('king').shape[0]

    # the final vector
    X = np.zeros((len(data), dim))
    emptycount = 0
    
    i = 0
    for word in data.items(): 
        
        embedding_vector = None
        
        try:
            embedding_vector = model.get_vector(word[0])
        except KeyError:
            pass   
        
        if embedding_vector is not None:
            X[i] = embedding_vector
        else  :
            emptycount += 1
            
        i += 1
        
        
    print("Numer of samples with no words found: %s / %s" % (emptycount, len(data)))
    
    return X
    

def get_dictionary(df, num_words=10000, column_name='review'):
    
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(df[column_name])
    words_to_index = tokenizer.word_index
    
    return tokenizer, words_to_index

def get_glove_embedding_BiLSTM(model, dictionary):
    
    embedding = fit_transform_word(model, dictionary)
    
    return embedding
    
    
