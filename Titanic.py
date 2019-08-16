#!/usr/bin/env python
# coding: utf-8

# In[286]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[287]:


train = pd.read_csv('train.csv')


# In[288]:


train.head()


# In[289]:


train.isnull()


# In[290]:


sns.heatmap(train.isnull())


# In[291]:


train.groupby('Pclass')['Age'].mean()


# In[292]:


sns.boxplot(x='Pclass',y='Age',data=train)


# In[293]:


def fillAge(passenger):
    age = passenger[0]
    pclass = passenger[1]
    if pd.isnull(age):
        if pclass == 1:
            return 38
        elif pclass == 2:
            return 29
        else:
            return 25
    else:
        return age


# In[294]:


train['Age']=train[['Age', 'Pclass']].apply(fillAge, axis=1)


# In[295]:


train.drop('Cabin', axis=1, inplace=True)


# In[296]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[297]:


gender=pd.get_dummies(train['Sex'], drop_first=True)


# In[298]:


train = pd.concat([train,gender],axis=1)


# In[299]:


embark=pd.get_dummies(['Embarked'],drop_first=True)


# In[300]:


train=pd.concat([train, embark], axis=1)


# In[301]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1, inplace=True)


# In[302]:


train.head()


# In[303]:


sns.countplot(x='Survived', hue='male',data=train)


# In[304]:


sns.distplot(train['Age'].dropna(), bins=30, kde=False)


# In[305]:


sns.countplot(x='SibSp',data=train)


# In[306]:


sns.distplot(train['Fare'],bins=20,kde=False)


# In[307]:


from sklearn.model_selection import train_test_split


# In[308]:


X_train,X_test,y_train,y_test = train_test_split(train.drop('Survived',axis = 1),
                                                 train['Survived'], test_size=0.30,
                                                 random_state=101)


# In[309]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions=logmodel.predict(X_test)


# In[310]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[ ]:




