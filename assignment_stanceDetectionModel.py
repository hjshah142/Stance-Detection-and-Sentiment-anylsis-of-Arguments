# -*- coding: utf-8 -*-
"""
Created on Tue May 28 06:11:18 2019

@author: user
"""

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
print(data)
english_stop_words = stopwords.words('english')
claim_corrected_data = data.iloc[:,7]


class StanceDetectionModel(object):
   
    def load_dataset(self,data):
        # remove the missing values
        dataset = data[pd.notnull(data['claims.claimSentiment'])]
        train =[]
        test = []
        # TO_Do split  dataset to tain and test( Based on column2) 
    
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
    
    def _vadersentiment_analysis(self,evidence):
		# baseline_approach vadersentiment
        analyser = SentimentIntensityAnalyzer()
        score = analyser.polarity_scores(evidence)
        if score['compound'] >= 0 :  
            stance = +1
        elif score['compound'] < 0 :  
            stance = -1 
        return  stance
    
    # sentiment analyzer scores 
    def _sentiment_analyzer_scores(self,evidence):
        # To_Do 
		# Extract Negative,neutral score  and positive score 
		# using polarity_scores function and return is as features 
        return 0

    # sentiment analyzer for bag of words of negative/ positive/ nuetral
    def _sentiment_analyzer_noOfWords(self,evidence):
        analyser = SentimentIntensityAnalyzer()
        premise_words = nltk.word_tokenize(evidence)
        num_of_positive_words = 0
        num_of_negative_words = 0
        num_of_neutral_words  = 0
    		# To_Do
		# Find number of posive words negative words, neutral words  
		# Usng polarity_scores function of vader

        return num_of_positive_words, num_of_negative_words , num_of_neutral_words


    def feature_representation(self,evidence):
        # Avg. Max. tfidf
        avg_tfidf_feature, max_tfidf_feature =self._tfidf_features(evidence)
        

		# To_Do (This method calls every features and return as array of feaures)
		# call _sentiment_analyzer_noOfWords and _sentiment_analyzer_scores methods here
  
		# positive score, nuetral score,  negative score,
        # Number of postive/negative/neutral words


        return  [negative_score,
                neutral_score,
                positive_score,
                num_of_positive_words, 
                num_of_negative_words, 
                num_of_neutral_words,
                avg_tfidf_feature, 
                max_tfidf_feature]
        

    def _instance_features(self,evidences):
        evidence_features = pd.DataFrame([self.feature_representation(evidence) for evidence in evidences])
        return evidence_features
       

