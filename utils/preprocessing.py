#%%
import numpy as np
import re
import emoji
from gensim.parsing.preprocessing import remove_stopwords
import string
string.punctuation

def remove_na(df, column):
    df = df.dropna(subset = [column])
    df = df.reset_index(drop = True)

    return df

def fill_na(df, column):
    df[column] = df[column].fillna('')
    
    return df

def preprocess_reviews(df, column='review'):
    df[column] = df[column].transform(func = preprocess_review)

    return df
    
def preprocess_review(s):
    
    s = s.lower()
    
    #removing urls, htmls tags, etc
    s = re.sub(r'https\S+', r'', str(s))
    s = re.sub(r'\\n', ' ', s)
    s = re.sub(r'\s', ' ', s)
    s = re.sub(r'<br>', ' ', s)
    s = re.sub(r'<br />', ' ', s)
    s = re.sub(r'&amp;', '&', s)
    s = re.sub(r'&#039;', "'", s)
    s = re.sub(r'&gt;', '>', s)
    s = re.sub(r'&lt;', '<', s)
    s = re.sub(r'\'', "'", s)
    
    #removing stopwords
    s = remove_stopwords(s)
    
    s = emoji.demojize(s)
    
    #removing hashtags
    HASHTAG_BEFORE = re.compile(r'#(\S+)')
    s = re.sub(HASHTAG_BEFORE, r'\1!!', s)
    
    # Just in case -- remove any non-ASCII and unprintable characters, apart from whitespace  
    s = "".join(x for x in s if (x.isspace() or (31 < ord(x) < 127)))
    s = ' '.join(s.split())
    
    return s

def remove_punctuation(df, column='review'):
    df[column] = df[column].transform(remove_punct)

    return df

def remove_punct(text):
    if(type(text)==float):
        return text
    
    ans=""  
    for i in text:     
        if i not in string.punctuation:
            ans+=i    
            
    return ans


def remove_overlap_words(df, column='review'):
    df[column] = df[column].transform(remove_overlap)

    return df

def remove_overlap(s):
    s = s.lower()
    
    s = s.replace("movie", "")
    s = s.replace("film", "")
    s = s.replace("it", "")
    s = s.replace("like", "")
    s = s.replace("good", "")
    s = s.replace("story", "")
    s = s.replace("time", "")
    s = s.replace("movies", "")
    s = s.replace("people", "")
    s = s.replace("watch", "")
    s = s.replace("seen", "")
    s = s.replace("way", "")
    s = s.replace("think", "")
    s = s.replace("don", "")
    
    return s
    