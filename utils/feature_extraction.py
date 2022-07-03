from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#count vectorizer
def get_count_vector(x_train, x_test, remove_stopwords=False):

    cv = CountVectorizer(min_df=0.0002)
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

    