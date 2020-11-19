# importing the necessary libraries
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm # just for fun in the for loops😀😀

# lets us pull the data in

data = pd.read_csv("data/spam",sep="\t",names=['Target','Content'])

from collections import Counter
def is_balanced(arr,threshold):
    """Return True or False if the data is balanced or not **For now only two unique labels are allowed**

    Args:
        arr (List): This the input arr which contains the target labels of the Dataframe
        threshold (int): The threshold value to test for the is_balanced
    """
    val_count = Counter(arr).values()
    for val in val_count:
        val = val/sum(val_count)
    if (min(val_count)<=threshold):
        return False
    else:
        return True

if is_balanced(data['Target'],0.4):
    print("The given data is balanced")
else:
    print("The given data is Imbalanced")

# Since this is somekind of spam/ham message let us stem the words using porter stemmer and remove the stopwords
ps = PorterStemmer()
corpus = []
for i in tqdm(range(0,len(data)),ascii ="123456789$",desc="Processing the Stemmer"):
    review = re.sub('[^a-zA-Z]',' ',data['Content'][i])
    # lets us convert the words to lower case
    review = review.lower()
    # split the data into lists "a b c"=> ["a","b","c"]
    review = review.split()
    # list compression by removing the stopwords and stem the 
    review = [ps.stem(words) for words in review if words not in stopwords.words("english") ]
    # convert those list into the strings
    review = ''.join(review)
    # append those review to corpus
    corpus.append(review)

# let us create a Bag of Words with the help of Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(data['Target'])
y = y.iloc[:,1].values

print(y.shape)
print(X.shape)

# let us now split the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 0)

# We can use multinomial Naive Bayes for this problem
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train,y_train)

# checking the model on the X_test data
y_pred = mnb.predict(X_test)

# let us check the confusion matrix of the y_pred and actual y_test
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test,y_pred)
print("The confusion matrix is \n",mat)

# let us check the accuracy score of the model
from sklearn.metrics import accuracy_score
print("The Accuracy of the Multinomial NB is \n",accuracy_score(y_test,y_pred))


'''
*** Need to tune the hyperparameters ***
'''

