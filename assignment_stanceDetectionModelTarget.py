# -*- coding: utf-8 -*-
"""
Created on Tue May 28 01:05:11 2019

@author: user
"""

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from   nltk.corpus import stopwords
from   matplotlib import pyplot as plt
from sklearn import svm
import textdistance
data = pd.read_csv('Dataset\claim_stance_dataset_v1.csv')
claim_corrected_data = data.iloc[:,7]
claim_target_data =data.iloc[:,16]
class StanceDetectionModelTarget(object):

    #tf_idf feature
    def _tfidf_features(self,evidence):
            avg_tfidf_feature = 0
            max_tfidf_feature = 0
            premise_words = nltk.word_tokenize(evidence)
            tfidf = TfidfVectorizer()
            # Whole dataset has to  be given for tfidf model
            tfidf_model = tfidf.fit(claim_corrected_data)
            tfidf_vector = tfidf_model.transform([evidence])
            avg_tfidf_feature = np.sum(tfidf_vector.toarray())/len(premise_words)
            max_tfidf_feature = np.max(tfidf_vector.toarray())
            return avg_tfidf_feature, max_tfidf_feature
        
    # sentiment analyzer scores 
    def sentiment_analyzer_scores(self,evidence):
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
                #print("ne",word)
                num_of_negative_words +=1
            if analyser.polarity_scores(word)['pos'] > 0: 
                #print("po",word)
                num_of_positive_words +=1
            if analyser.polarity_scores(word)['neu'] > 0:
                #print(word)
                num_of_neutral_words+=1
        return num_of_positive_words, num_of_negative_words , num_of_neutral_words
    
    # Word Similarity 
    def _similarity_feature (self,claim,evidence):
        return textdistance.levenshtein.normalized_similarity(claim, evidence)

    
    def feature_representation(self,evidence,target):
            # Avg. Max. tfidf
            avg_tfidf_feature, max_tfidf_feature =self._tfidf_features(evidence)
            
            # positive score, nuetral score,  negative score, compound score
            negative_score,neutral_score,positive_score,compound_score = self.sentiment_analyzer_scores(evidence)
            
            # Number of postive/negative/neutral words
            num_of_positive_words, num_of_negative_words , num_of_neutral_words  = self. _sentiment_analyzer_noOfWords(evidence)
            
            # Word Similarity feature
            similarity_score = self._similarity_feature(evidence,target)
            
            return [negative_score,
                    neutral_score,
                    positive_score,
                    compound_score,
                    num_of_positive_words, 
                    num_of_negative_words, 
                    num_of_neutral_words,
                    avg_tfidf_feature,
                    similarity_score,
                    max_tfidf_feature]
           
            
    def _instance_features(self,evidences,targets):
        evidence_features = pd.DataFrame([self.feature_representation(evidence,target) for evidence,target in zip(evidences, targets)])
        return evidence_features