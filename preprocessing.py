#%%
import pandas as pd
from utils import preprocessing
from utils import feature_extraction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os


#%%
#BASIC PREPROCESSING
df = pd.read_csv('IMDB Dataset.csv')
#df = preprocessing.remove_na(df)
#df = preprocessing.preprocess_reviews(df)
#df = preprocessing.remove_punctuation(df)
#df.to_csv('IMDB dataset-preprocessed.csv', index = False, encoding='utf-8')

Encoder = LabelEncoder()
df['sentiment'] = Encoder.fit_transform(df['sentiment'])


#%%
#TRAIN TEST SPLIT
#1 - positive, 0 - negative
x_train, x_test, y_train, y_test = train_test_split(df['review'],df['sentiment'],test_size=0.15, shuffle=True)

x_train.to_csv('x_train_without_preprocessing.csv', index = False, encoding='utf-8')
x_test.to_csv('x_test_without_preprocessing.csv', index = False, encoding='utf-8')
y_train.to_csv('y_train_without_preprocessing.csv', index = False, encoding='utf-8')
y_test.to_csv('y_test_without_preprocessing.csv', index = False, encoding='utf-8')
'''
# %%
#COUNT VECTORIZER
x_train = pd.read_csv('x_train.csv', converters = {'review': str})
x_test = pd.read_csv('x_test.csv', converters = {'review': str})
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')
#posto je ovo ucitano ko df pri pozivanju modela ce se morat slat y_train['sentiment'], y_test['sentiment'] vjv

dictionary, x_train_vector, x_test_vector = feature_extraction.get_count_vector(x_train['review'], x_test['review'], remove_stopwords=False)
#len(dictionary) - 26599
#x_train_vector.shape - (42500, 26599)
#x_test_vector.shape - (7500, 26599)

# %%
#TF-IDF
dictionary, x_train_vector, x_test_vector = feature_extraction.get_tfidf_vector(x_train['review'], x_test['review'], remove_stopwords=False, ngram_range=None)
#len(dictionary) - 26599
#x_train_vector.shape - (42500, 26599)
#x_test_vector.shape - (7500, 26599)

# %%
#WORD2VEC EMBEDDING - ovo se izvrsava oko 10 minuta
model = feature_extraction.create_word2vec_model(x_train['review'], x_test['review']) 
x_train_vector, x_test_vector = feature_extraction.get_word2vec_embedding(model, x_train['review'], x_test['review']) 
#x_train_vector.shape - (42500, 200)
#x_test_vector.shape - (7500, 200)

# %%
#GLOVE EMBEDDING
dirname = os.path.dirname(__file__)
filepath = os.path.join(dirname, 'glove.6B.200d.txt')
    
word2vec_output_file = 'glove.6B.200d' +'.word2vec'

model = feature_extraction.load_glove_model(filepath, word2vec_output_file)
x_train_vector, x_test_vector = feature_extraction.get_glove_embedding(model, x_train['review'], x_test['review'])
#x_train_vector.shape - (42500, 200)
#x_test_vector.shape - (7500, 200)


# %%
'''