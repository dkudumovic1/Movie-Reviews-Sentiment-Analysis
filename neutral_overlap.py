#%%
import pandas as pd
import numpy as np
from utils import preprocessing

#%%

df = pd.read_csv('data/neutral_dataset_preprocess.csv', converters = {'Phrase': str})
# %%
df = preprocessing.remove_overlap_words(df, 'Phrase')
# %%
df.to_csv('data/neutral_without_overlap.csv', index = False, encoding='utf-8')

# %%
