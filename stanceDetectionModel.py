# -*- coding: utf-8 -*-
"""
Created on Sun May 26 19:45:05 2019

@author: user
"""

import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix


class StanceDetectionModel():
    

    def load_dataset(data):
        dataset = data[pd.notnull(data['claims.claimSentiment'])]
        train =[]
        test = []
        for index, row in dataset.iterrows():
            if row['split'] == "train":
                train.append((row[2:]))
            elif row['split'] == "test":
                test.append((row[2:]))
    
        train = pd.DataFrame(train)
        test = pd.DataFrame(test)   
        return train,test

    def _tfidf_features(evidence):
        avg_tfidf_feature = 0
        max_tfidf_feature = 0
        evidence_words = nltk.word_tokenize(evidence)
        tfidf = TfidfVectorizer()
        # Whole dataset has to  be given for tfidf model
        tfidf_model = tfidf.fit(claim_corrected_data)
        tfidf_vector = tfidf_model.transform([evidence])
        avg_tfidf_feature = np.sum(tfidf_vector.toarray())/len(evidence_words)
        max_tfidf_feature = np.max(tfidf_vector.toarray())
        return avg_tfidf_feature, max_tfidf_feature
    # sentiment analyzer scores 
    def sentiment_analyzer_scores(evidence):
        analyser = SentimentIntensityAnalyzer()
        score = analyser.polarity_scores(evidence)
        return  score['neg'],score['neu'],score['pos'],score['compound']

    # sentiment analyzer for bag of words of negative/ positive/ nuetral
    def _sentiment_analyzer_noOfWords(evidence):
        analyser = SentimentIntensityAnalyzer()
        premise_words = nltk.word_tokenize(evidence)
        num_of_positive_words = 0
        num_of_negative_words = 0
        num_of_neutral_words  = 0
        for word in premise_words:
            if analyser.polarity_scores(word)['neg'] > 0: 
                num_of_negative_words +=1
            if analyser.polarity_scores(word)['pos'] > 0: 
                num_of_positive_words +=1
            if analyser.polarity_scores(word)['neu'] > 0:
                num_of_neutral_words+=1
        return num_of_positive_words, num_of_negative_words , num_of_neutral_words


    # Number of words
    def _num_of_words_feature(evidence):
        premise_words = nltk.word_tokenize(evidence)
        return len(premise_words)


    def feature_representation(evidence):
            # Number of words
        num_of_words_feature = _num_of_words_feature(evidence)
            
            # Avg. Max. tfidf
        avg_tfidf_feature, max_tfidf_feature = _tfidf_features(evidence)
        
            # positive score, nuetral score,  negative score, compound score
        negative_score,neutral_score,positive_score,compound_score = sentiment_analyzer_scores(evidence)
        
        # Number of postive/negative/neutral words
        num_of_positive_words, num_of_negative_words , num_of_neutral_words  = _sentiment_analyzer_noOfWords(evidence)

        return [num_of_words_feature,
                negative_score,
                neutral_score,
                positive_score,
                compound_score,
                num_of_positive_words, 
                num_of_negative_words, 
                num_of_neutral_words,
                avg_tfidf_feature, 
                max_tfidf_feature]
        

    def _instance_features(evidences):
        
        evidence_features = pd.DataFrame([feature_representation(evidence) for evidence in evidences])
        return evidence_features
    # Import dataset
    import pandas as pd
    data = pd.read_csv('Dataset/claim_stance_dataset_v1.csv') 
    train,test = load_dataset(data)
    claim_corrected_data_train = train.iloc[:,5]
    claim_sentiment_data_train = train.iloc[:,17]
    X_train_data =_instance_features(claim_corrected_data_train)
