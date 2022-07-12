#%%
import pandas as pd
from utils import preprocessing


#CLEANING DATASET
train = pd.read_csv('train.tsv', sep='\t')
test = pd.read_csv('test.tsv', sep='\t')

data = pd.concat([train, test], ignore_index=True)
data = data.drop('PhraseId', axis=1)

df = pd.DataFrame(columns=['SentenceId', 'Phrase', 'Sentiment'])
df = df.append(data.loc[0], ignore_index=True)

#delete rows with phrases
old_sentenceId = data.loc[0, 'SentenceId']
for i in range(len(data['SentenceId'])):
    new_sentenceId = data.loc[i, 'SentenceId']
    if old_sentenceId != new_sentenceId:
        df = df.append(data.loc[i], ignore_index=True)
        old_sentenceId = new_sentenceId

df.to_csv('neutral_dataset.csv', index = False, encoding='utf-8')


#BASIC PREPROCESSING
df = pd.read_csv('neutral_dataset.csv')
df.columns = ['id', 'SentenceId', 'Phrase', 'Sentiment']
df = df.drop('id', axis=1)
df = preprocessing.remove_na(df, 'Sentiment')
df['Sentiment'] = df['Sentiment'].astype('int')
df = preprocessing.preprocess_reviews(df, 'Phrase')
df = preprocessing.remove_punctuation(df, 'Phrase')
df.to_csv('neutral-dataset-preprocessed.csv', index = False, encoding='utf-8')