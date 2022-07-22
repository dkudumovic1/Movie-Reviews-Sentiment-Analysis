#%%
import pandas as pd
from torch import seed
from utils import preprocessing
from utils import feature_extraction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#%%
df = pd.read_csv('IMDB Dataset.csv')
df = preprocessing.remove_na(df, 'review')
df = preprocessing.preprocess_reviews(df)
df = preprocessing.remove_punctuation(df)
df = preprocessing.remove_overlap_words(df)

#%%
Encoder = LabelEncoder()
df['sentiment'] = Encoder.fit_transform(df['sentiment'])

#%%
#TRAIN TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(df['review'],df['sentiment'],test_size=0.15, shuffle=True, random_state=27)

x_train.to_csv('data/x_train_without_overlap.csv', index = False, encoding='utf-8')
x_test.to_csv('data/x_test_without_overlap.csv', index = False, encoding='utf-8')
y_train.to_csv('data/y_train_without_overlap.csv', index = False, encoding='utf-8')
y_test.to_csv('data/y_test_without_overlap.csv', index = False, encoding='utf-8')
# %%

# %%

#Most common unigrams after removing overlap - positive reviews
#{'great': 473.19739800207947, 'love': 349.26061625946124, 'best': 326.48075205258414, 'life': 292.60838699425307, 'characters': 290.0421471348098, 'ltle': 257.33805177266464, 'character': 248.25455054248337, 'know': 245.60544318216245, 'ing': 238.98534891754406, 'acting': 235.43173240163264, 'funny': 229.1129523497811, 'years': 221.85822709345527, 'man': 221.2033826548083, 'end': 219.73006236993166, 'plot': 215.7508315432961, 'the': 214.77194237042687, 'better': 214.21679636098406, 'real': 213.98595731499432, 'scenes': 207.165394208361, 'scene': 203.74420679729818}

#Most common unigrams after removing overlap - negative reviews
#{'bad': 535.8987493536234, 'acting': 346.32572999819934, 'plot': 335.6624719611686, 'ing': 317.01871010707913, 'characters': 302.65839176253616, 'better': 290.77523845738796, 'worst': 268.60019542028834, 'know': 267.7329036491419, 'character': 259.0705221350275, 'im': 255.99354520545998, 'didnt': 251.97517966319955, 'thing': 251.83203490729275, 'actors': 245.34719524646877, 'ltle': 243.8502303711047, 'funny': 243.60825518053096, 'scenes': 243.2494217411898, 'end': 240.37291250348616, 'great': 234.82370696739147, 'scene': 229.43411711701435, 'actually': 226.41544650568082}