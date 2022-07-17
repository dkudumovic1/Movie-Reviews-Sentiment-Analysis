#%%
import pandas as pd
import numpy as np
from utils import preprocessing

#%%

df = pd.read_csv('data/neutral_dataset_preprocess.csv', converters = {'Phrase': str})
# %%
#Num of missing values
print(df.isnull().sum())
# %%
#Num of positive/negative reviews
num0 = len(df[df.Sentiment==0])
num1 = len(df[df.Sentiment==1])
num2 = len(df[df.Sentiment==2])
num3 = len(df[df.Sentiment==3])
num4 = len(df[df.Sentiment==4])
print("Number of instances with label 0: " + str(num0))
print("Number of instances with label 1: " + str(num1))
print("Number of instances with label 2: " + str(num2))
print("Number of instances with label 3: " + str(num3))
print("Number of instances with label 4: " + str(num4))

#%%

import matplotlib.pyplot as plt

def length_graph(df, label1, label2):
    print(label1)
    df_positive = df[df.Sentiment==label1]
    df_negative = df[df.Sentiment==label2]

    fig, axes = plt.subplots(1, 2)
    fig.suptitle("Average review length", fontsize=14)
    axes[0].set_title('Reviews with label' + str(label1))
    axes[0].set_xlabel('Length')
    axes[0].set_ylabel('Number of reviews')
    axes[1].set_title('Reviews with label' + str(label2))
    axes[1].set_xlabel('Length')
    axes[1].set_ylabel('Number of reviews')
    
    df_positive['Phrase'].str.len().hist(ax=axes[0], bins=10,range=(0,6000), color='pink')
    df_negative['Phrase'].str.len().hist(bins=10, ax=axes[1], range=(0,6000))
    plt.show()

length_graph(df, 0, 1)
length_graph(df, 2, 2)
length_graph(df, 3, 4)
# %%
#Wordcloud

from wordcloud import WordCloud

def wordcloud(df):
    #uporredit rezultat s dzenetinim

    text = " ".join(sentiment for sentiment in df.Phrase)

    wordcloud = WordCloud(
            background_color='white',
            max_words=100,
            max_font_size=30,
            scale=3,
            random_state=1)

    wordcloud=wordcloud.generate(text)

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')

    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.show()
# %%
wordcloud(df[df.Sentiment==0])
wordcloud(df[df.Sentiment==1])
wordcloud(df[df.Sentiment==2])
wordcloud(df[df.Sentiment==3])
wordcloud(df[df.Sentiment==4])
# %%
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
    fig = px.bar(temp, x="Count", y="Common_words", title= title, orientation='h', width = 1000, height=700,
                color='Common_words', color_discrete_sequence=px.colors.qualitative.Plotly)
    
    fig.layout.showlegend = False
    fig.show()
    
#%%
draw_plot_for_common_ngrams(df[df.Sentiment==0]['Phrase'], 1, 20, "Common Unigrams in Text (Label 0)", 2)
draw_plot_for_common_ngrams(df[df.Sentiment==1]['Phrase'], 1, 20, "Common Unigrams in Text (Label 1)", 2)
draw_plot_for_common_ngrams(df[df.Sentiment==2]['Phrase'], 1, 20, "Common Unigrams in Text (Label 2)", 2)
draw_plot_for_common_ngrams(df[df.Sentiment==3]['Phrase'], 1, 20, "Common Unigrams in Text (Label 3)", 2)
draw_plot_for_common_ngrams(df[df.Sentiment==4]['Phrase'], 1, 20, "Common Unigrams in Text (Label 4)", 2)
# %%

draw_plot_for_common_ngrams(df[df.Sentiment==0]['Phrase'], 3, 20, "Common Trigrams in Text (Label 0)", 2)
draw_plot_for_common_ngrams(df[df.Sentiment==1]['Phrase'], 3, 20, "Common Trigrams in Text (Label 1)", 2)
draw_plot_for_common_ngrams(df[df.Sentiment==2]['Phrase'], 3, 20, "Common Trigrams in Text (Label 2)", 2)
draw_plot_for_common_ngrams(df[df.Sentiment==3]['Phrase'], 3, 20, "Common Trigrams in Text (Label 3)", 2)
draw_plot_for_common_ngrams(df[df.Sentiment==4]['Phrase'], 3, 20, "Common Trigrams in Text (Label 4)", 2)


# %%
