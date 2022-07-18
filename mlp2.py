#%%
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn import svm
from sklearn.linear_model import SGDClassifier
import utils
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import spacy
from utils import feature_extraction

#%%
neutral = pd.read_csv('data/neutral_dataset_without_preprocessing.csv', converters = {'Phrase': str})
y_neutral = neutral['Sentiment']

model = pickle. load(open('MLP.sav', 'rb'))

#%%
#converting the data to a vector representation

nlp_lg = spacy.load('en_core_web_lg')

def convert_data(corpus):
    new_corpus = []
    for document in corpus:
        doc = nlp_lg(document)
        new_corpus.append(doc.vector)
    return(new_corpus)

x_neutral = convert_data(neutral['Phrase'])

#%%
predicted_probability = model.predict_proba(x_neutral)
predicted = model.predict(x_neutral)

ids_neutral = neutral.index

list_of_all_neutral = list(zip(ids_neutral, y_neutral, predicted, predicted_probability))

# Converting lists of tuples into pandas Dataframe.
df_prob_neutral = pd.DataFrame(list_of_all_neutral, columns=['Id', 'Label', 'Prediction', 'Probability'])

# %%
df_prob_neutral.to_csv('predictions/mlp_neutral_without_preprocessing.csv', index=False)


# %%
