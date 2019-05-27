# -*- coding: utf-8 -*-
"""
Created on Sun May 26 19:45:05 2019

@author: Harsh Shah
"""

import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix

from nltk.corpus import stopwords

data = pd.read_csv('Dataset\claim_stance_dataset_v1.csv')
english_stop_words = stopwords.words('english')
claim_corrected_data = data.iloc[:,7]





class StanceDetectionModel(object):
   
    def load_dataset(self,data):
        # remove the missing values
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
    
    def remove_stop_words(self,corpus):
        removed_stop_words = []
        for evidence in corpus:
            removed_stop_words.append(
                ' '.join([word for word in evidence.split() 
                          if word not in english_stop_words])
            )
        return removed_stop_words
   
    def get_stemmed_text(self,corpus):
        from nltk.stem.porter import PorterStemmer
        stemmer = PorterStemmer()
        return [' '.join([stemmer.stem(word) for word in evidence.split()]) for evidence in corpus]
    
   
    def get_lemmatized_text(self,corpus):
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        return [' '.join([lemmatizer.lemmatize(word) for word in evidence.split()]) for evidence in corpus]  

    def _tfidf_features(self,evidence):
        
        
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
    def _sentiment_analyzer_scores(self,evidence):
        analyser = SentimentIntensityAnalyzer()
        score = analyser.polarity_scores(evidence)
        return  score['neg'],score['neu'],score['pos'],score['compound']

    # sentiment analyzer for bag of words of negative/ positive/ nuetral
    def _sentiment_analyzer_noOfWords(self,evidence):
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


    def feature_representation(self,evidence):
            
            
        # Avg. Max. tfidf
        avg_tfidf_feature, max_tfidf_feature =self._tfidf_features(evidence)
        
            # positive score, nuetral score,  negative score, compound score
        negative_score,neutral_score,positive_score,compound_score = self._sentiment_analyzer_scores(evidence)
        
        # Number of postive/negative/neutral words
        num_of_positive_words, num_of_negative_words , num_of_neutral_words  = self._sentiment_analyzer_noOfWords(evidence)

        return  [negative_score,
                neutral_score,
                positive_score,
                compound_score,
                num_of_positive_words, 
                num_of_negative_words, 
                num_of_neutral_words,
                avg_tfidf_feature, 
                max_tfidf_feature]
        

    def _instance_features(self,evidences):
        evidence_features = pd.DataFrame([self.feature_representation(evidence) for evidence in evidences])
        return evidence_features
       

