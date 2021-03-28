import pandas as pd
import numpy as np

df=pd.read_csv("balanced_reviews.csv")

df['overall'].value_counts()

df.isnull().any(axis=0)

#Drop records having missing text

df.dropna(inplace=True)
#make changes in original dataframe

#Since it is binary classifier we need to drop rating 3

df=df[df['overall'] !=3]

df['Positivity']=np.where(df['overall']>3,1,0)
#condition,if true output, if false output
"""
#data cleaning
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]
for i in range(0,df.shape[0]):
    review=re.sub('[^a-zA-Z]',' ',df.iloc[i,1])
    
    review=review.lower()
    
    review=review.split()
    ps=PorterStemmer()
    
    review=[word for word in review if not word in stopwords.words('english')]
    
    review=[ps.stem(word) for word in review]
    
    review=' '.join(review)
    
    corpus.append(review)
    
"""

#nlp coming up
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test=train_test_split(df['reviewText'],df['Positivity'],random_state=42)
#Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer

vect=TfidfVectorizer(min_df=5).fit(features_train)

features_train_vectorized=vect.transform(features_train)

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(features_train_vectorized,labels_train)

predictions=model.predict(vect.transform(features_test))

from sklearn.metrics import confusion_matrix
confusion_matrix(labels_test,predictions)

from sklearn.metrics import roc_auc_score
roc_auc_score(labels_test,predictions)

#storing model in pickle file
import pickle

pkl_filename = "pickle_model.pkl"


file = open(pkl_filename, 'wb')

pickle.dump(model, file)

file.close()


