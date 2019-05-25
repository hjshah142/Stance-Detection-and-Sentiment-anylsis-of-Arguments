# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:49:56 2019

@author: Harsh Shah

"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix
import nltk
from   nltk.corpus import sentiwordnet as swn
from   nltk.corpus import stopwords
from   matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
#from gensim.models import tfidfModel
from sklearn.feature_extraction.text import TfidfVectorizer

dataset = pd.read_csv('claim_stance_dataset_v1.csv')
# Handle missing data
dataset = dataset[pd.notnull(dataset['claims.claimSentiment'])]
#X_train, X_test, y_train, y_test = train_test_split(claim_data ,claim_data_class , test_size=0.33, random_state=42)
#split traning and test data
train =[]
test = []
for index, row in dataset.iterrows():
    if row['split'] == "train":
         train.append((row[2:]))
    elif row['split'] == "test":
         test.append((row[2:]))

train = pd.DataFrame(train)
test = pd.DataFrame(test)

#Importing dataset
claim_corrected_data = dataset.iloc[:,7]
claim_target_data = dataset.iloc[:,16]
claim_sentiment_data = dataset.iloc[:,19]


#relation_claim_topic = dataset.iloc[:,20] 

#Sentiment_feature
def _build_tfidf_model(texts):
     tfidf = TfidfVectorizer()
     tfidf_model = tfidf.fit(texts)
     return tfidf_model
 
def _tfidf_features(premise):
        avg_tfidf_feature = 0
        max_tfidf_feature = 0
        
        premise_words = nltk.word_tokenize(premise)
        tfidf = TfidfVectorizer()
# Whole dataset has to  be given for tfidf model
        tfidf_model = tfidf.fit(claim_corrected_data)
        tfidf_vector = tfidf_model.transform([premise])
        avg_tfidf_feature = np.sum(tfidf_vector.toarray())/len(premise_words)
        max_tfidf_feature = np.max(tfidf_vector.toarray())
        return avg_tfidf_feature, max_tfidf_feature

def sentiment_analyzer_scores(sentence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)
    #print("{:-<40} {}".format(sentence, str(score)))
    return  score['neg'],score['neu'],score['pos'],score['compound']
def _sentiment_features(premise):
    analyser = SentimentIntensityAnalyzer()
    premise_words = nltk.word_tokenize(premise)
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
def _num_of_words_feature(premise):
    premise_words = nltk.word_tokenize(premise)
    return len(premise_words)


def feature_representation(premise):
        # Number of words
        num_of_words_feature = _num_of_words_feature(premise)
        
        # Avg. Max. tfidf
        #avg_tfidf_feature, max_tfidf_feature = self._tfidf_features(premise)
        
        # positive score, nuetral score,  negative score, compound score
        negative_score,neutral_score,positive_score,compound_score = sentiment_analyzer_scores(premise)
        # Number of postive/negative/neutral words
        num_of_positive_words, num_of_negative_words , num_of_neutral_words  = _sentiment_features(premise)
             #   arguments = [' '.join(argument[0]) for argument in data]

#        tfidf_model = _build_tfidf_model(claim_target_data[0])

        avg_tfidf_feature, max_tfidf_feature = _tfidf_features(premise)
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
        

def _instance_features(premises):
    #premises_text = ' '.join(premises)
    premises_features = pd.DataFrame([feature_representation(premise) for premise in premises])
    return premises_features

X_train_class_data =  claim_sentiment_data[0:2000]
# Feautere calculations for the train data 
X_train_data =_instance_features(claim_corrected_data[0:2000])

#print(X_train_data)

from sklearn import svm
clf=svm.SVC(gamma='auto')
clf.fit(X_train_data,X_train_class_data)
Y_test_data = claim_corrected_data[2001:2260]
Y_test_data_transfom =_instance_features(claim_corrected_data[2001:2260])
actual =claim_sentiment_data[2001:2260]
Y_test_class_data = clf.predict(Y_test_data_transfom)


matrix = confusion_matrix(actual, Y_test_class_data)
print(matrix)

accuray = (matrix[0][0]+matrix[1][1]) / 259
precision = matrix[1][1] / (matrix[0][1]+matrix[1][1])
recall    = matrix[0][0] / (matrix[0][0]+matrix[1][0])

f_measure = (2 * precision * recall ) / (precision + recall)



topic_sentment_test =  dataset.iloc[2001:2260,4]
relation =  dataset.iloc[2001:2260,-1]

predicted_stance =  topic_sentment_test * relation * Y_test_class_data

stance =  dataset.iloc[2001:2260,6]
stance_filtered = stance.replace('PRO', 1)
stance_filtered = stance_filtered.replace('CON', -1)
# pro/ con to 1 / -1 
matrix = confusion_matrix(stance_filtered, predicted_stance)
print(matrix)

accuray2 = (matrix[0][0]+matrix[1][1]) / 259
precision2 = matrix[1][1] / (matrix[0][1]+matrix[1][1])
recall2    = matrix[0][0] / (matrix[0][0]+matrix[1][0])

f_measure2 = (2 * precision * recall ) / (precision + recall)

