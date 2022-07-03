#%%
import pandas as pd
from utils import preprocessing


#%%
#BASIC PREPROCESSING
df = pd.read_csv('IMDB Dataset.csv')
df = preprocessing.remove_na(df)
df = preprocessing.preprocess_reviews(df)
df = preprocessing.remove_punctuation(df)
df.to_csv('IMDB dataset-preprocessed.csv',index = False, encoding='utf-8')


# %%
