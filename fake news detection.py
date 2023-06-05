#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


train=pd.read_csv('input/fake-news/train.csv')
train


# In[3]:


test=pd.read_csv('input/fake-news/test.csv')
test


# In[4]:


dataset=pd.read_csv('/kaggle/input/fake-news/submit.csv')
dataset


# In[5]:


train.isnull().sum()


# In[6]:


train.shape


# In[7]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer


# In[8]:


train=train.dropna()


# In[9]:


train.shape


# In[10]:


messages=train.copy()
messages


# In[11]:


messages.reset_index(inplace=True)
messages


# In[12]:


messages['text'][6]


# In[13]:


import nltk
from nltk.corpus import stopwords

import re

def preprocess_corpus(messages):
    corpus = []
    for i in range(0, len(messages)):
        review = re.sub('[^a-zA-Z]', ' ', messages['text'][i])
        review = review.lower()
        review = review.split()

        review = [word for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
        
    return corpus   


# In[14]:


preprocessed_corpus = preprocess_corpus(messages)


# In[15]:


## TFidf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
X=tfidf_v.fit_transform(preprocessed_corpus).toarray()


# In[16]:


y=messages['label']


# In[17]:


## Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[18]:


tfidf_v.get_feature_names_out()[:20]


# In[19]:


tfidf_v.get_params()


# In[20]:


count_df = pd.DataFrame(X_train, columns=tfidf_v.get_feature_names_out())


# In[21]:


count_df


# In[22]:


import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# **MultinomialNB Algorithm**

# In[23]:


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import itertools

classifier=MultinomialNB()


# In[24]:


classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


# **Logistic Regression**

# In[25]:


from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(X_train,y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = metrics.accuracy_score(X_train_prediction, y_train)
print('Accuracy score of the training data:',training_data_accuracy)


# In[26]:


# accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = metrics.accuracy_score(X_test_prediction, y_test)
print('Accuracy score of the test data : ', test_data_accuracy)


# **For test data**

# In[27]:


test=pd.read_csv('/kaggle/input/fake-news/test.csv')
test


# In[28]:


test = test.fillna('')


# In[29]:


test.isnull().sum()


# In[30]:


test.shape


# In[31]:


tfidf_df_test = pd.DataFrame(tfidf_v.transform(test['text']).todense())
tfidf_df_test.columns = sorted(tfidf_v.get_feature_names_out())
y_pred_test = model.predict(tfidf_df_test)
y_pred_test


# In[32]:


idx=list(test['id'])
submission=pd.DataFrame({'id':idx,'label':list(y_pred_test)})


# In[33]:


submission.to_csv('Submission.csv', index=False)


# In[34]:


out=pd.read_csv('/kaggle/working/Submission.csv')
out


# In[ ]:




