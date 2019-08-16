#!/usr/bin/env python
# coding: utf-8

# In[305]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
train = pd.read_csv('ramen-ratings.csv')


# In[306]:


train.head()


# In[307]:


train.drop('TopTen', axis=1, inplace=True)


# In[308]:


train.head()


# In[309]:


def fillStyle(style):
    style = style[0]
    if style=='Cup':
        return 0
    elif style=='Bar':
        return 1
    elif style=='Bowl':
        return 2
    elif style=='Box':
        return 3
    elif style=='Can':
        return 4
    elif style=='Pack':
        return 5
    else:
        return 6
train['Style']=train[['Style']].apply(fillStyle,axis=1)


# In[310]:


def fillCountry(country):
    country = country[0]
    if country=='Japan':
        return 0
    elif country=='USA':
        return 1
    elif country=='South Korea':
        return 2
    elif country=='Taiwan':
        return 3
    elif country=='Thailand':
        return 4
    elif country=='China':
        return 5
    else:
        return 6
train['Country']=train[['Country']].apply(fillCountry,axis=1)


# In[311]:


def fillStars(stars):
    stars = stars[0]
    if '3' in stars:
        return 1
    elif '2' in stars:
        return 0
    elif '1' in stars:
        return 0
    else:
        return 1
train['Stars']=train[['Stars']].apply(fillStars,axis=1)


# In[312]:


train.drop(['Brand','Variety'],axis=1, inplace=True)


# In[313]:


from sklearn.model_selection import train_test_split


# In[ ]:


dummieStars=pd.get_dummies(train['Stars'], drop_first=True)


# In[314]:


X_train,X_test,y_train,y_test = train_test_split(train.drop('Stars',axis = 1),
                                                 train['Stars'], test_size=0.99,
                                                 random_state=101)


# In[315]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions=logmodel.predict(X_test)


# In[316]:


train.head()


# In[317]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[318]:


def score():
    dummieStars=stars[0]
    stars=stars[1]
    if dummieStars==stars:
        print('True')
    else:
        print('False')
score()


# In[ ]:




