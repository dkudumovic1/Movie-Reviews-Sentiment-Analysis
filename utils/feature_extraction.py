from sklearn.feature_extraction.text import CountVectorizer

#count vectorizer
def get_count_vector(df, x_train, x_test, remove_stopwords=False):

    cv = CountVectorizer(min_df=0.0002)
    if remove_stopwords:
        cv.stop_words = 'english'
        
    cv.fit(df['review'])
    x_train_vector = cv.transform(x_train)
    x_test_vector = cv.transform(x_test)
    
    return cv.vocabulary_, x_train_vector, x_test_vector
    