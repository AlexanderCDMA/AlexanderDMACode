#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


dataset = pd.read_csv("./housing_dataset.csv")


# In[45]:


dataset.head()


# In[19]:


sns.pairplot(dataset)


# In[21]:


sns.distplot(dataset['Price'])


# In[22]:


sns.heatmap(dataset.corr())


# In[25]:


X = dataset[['Avg. Area Income','Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]


# In[26]:


y = dataset[['Price']]


# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)


# In[28]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)


# In[32]:


print(lm.intercept_)


# In[35]:


lm.coef_


# In[36]:


predictions = lm.predict(X_test)


# In[38]:


plt.scatter(y_test,predictions)


# In[43]:


from sklearn import metrics


# In[44]:


metrics.mean_absolute_error(y_test,predictions)

