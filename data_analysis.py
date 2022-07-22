#%%
import pandas as pd
import numpy as np
from utils import preprocessing

#%%
df = pd.read_csv('IMDB Dataset.csv')

df[1:10]

df.info()

#Num of missing values
print(df.isnull().sum())

#Num of positive/negative reviews
positive = len(df[df.sentiment=='positive'])
negative = len(df[df.sentiment=='negative'])
print("Positive count: " + str(positive))
print("Negative count: " + str(negative))

#%%
#average length of review

import matplotlib.pyplot as plt

def length_graph(df):
    df_positive = df[df.sentiment=='positive']
    df_negative = df[df.sentiment=='negative']

    fig, axes = plt.subplots(1, 2)
    fig.suptitle("Average review length", fontsize=14)
    axes[0].set_title('Positive reviews')
    axes[0].set_xlabel('Length')
    axes[0].set_ylabel('Number of reviews')
    axes[1].set_title('Negative reviews')
    axes[1].set_xlabel('Length')
    axes[1].set_ylabel('Number of reviews')
    
    df_positive['review'].str.len().hist(ax=axes[0], bins=10,range=(0,6000), color='pink')
    df_negative['review'].str.len().hist(bins=10, ax=axes[1], range=(0,6000))
    plt.show()

length_graph(df)

#%%
#Wordcloud

from wordcloud import WordCloud

def wordcloud(df):
    #uporredit rezultat s dzenetinim

    text = " ".join(sentiment for sentiment in df.review)

    wordcloud = WordCloud(
            background_color='black',
            max_words=100,
            max_font_size=30,
            scale=3,
            random_state=1)

    wordcloud=wordcloud.generate(text)

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')

    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.show()


#%%
#Preprocessing

df['review'] = df['review'].transform(preprocessing.preprocess_review)
df['review']

wordcloud(df[df.sentiment=='positive'])
wordcloud(df[df.sentiment=='negative'])

#%%
#N grams

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import plotly.express as px

def get_top_text_ngrams(corpus, n, g, mode=1):
    if mode == 2:
        vec = TfidfVectorizer(ngram_range=(g, g)).fit(corpus)
    else:
        vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

#mode 2 - TF-ID
def draw_plot_for_common_ngrams(text, n=1, number_of_common=20, title = "Common N-grams in Text", mode=1):
    most_common = get_top_text_ngrams(text, number_of_common, n, 2)
    most_common = dict(most_common)
    print(most_common)
    temp = pd.DataFrame(columns=["Common_words", "Count"])
    temp["Common_words"] = list(most_common.keys())
    temp["Count"] = list(most_common.values())
    fig = px.bar(temp, x="Count", y="Common_words", title= title, orientation='h', width = 1000,
                color='Common_words', color_discrete_sequence=px.colors.qualitative.Plotly)
    
    fig.layout.showlegend = False
    fig.show()

draw_plot_for_common_ngrams(df[df.sentiment=='positive']['review'], 3, 20, "Common Trigrams in Text (Positive reviews)", 2)
draw_plot_for_common_ngrams(df[df.sentiment=='negative']['review'], 3, 20, "Common Trigrams in Text (Negative reviews)", 1)

draw_plot_for_common_ngrams(df[df.sentiment=='positive']['review'], 2, 20, "Common Bigrams in Text (Positive reviews)", 2)
draw_plot_for_common_ngrams(df[df.sentiment=='negative']['review'], 2, 20, "Common Bigrams in Text (Negative reviews)", 2)
# %%
draw_plot_for_common_ngrams(df[df.sentiment=='positive']['review'], 1, 20, "Common Unigrams in Text (Positive reviews)", 2)
draw_plot_for_common_ngrams(df[df.sentiment=='negative']['review'], 1, 20, "Common Unigrams in Text (Negative reviews)", 2)
# %%


#Common words - uverlap:
# movie, film, it, like, good, story, time, movies, people, watch, seen, way, think, don't (don')