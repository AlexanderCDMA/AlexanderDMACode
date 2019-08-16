#!/usr/bin/env python
# coding: utf-8

# In[216]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random as r


# In[217]:


fashion_train_df=pd.read_csv('fashion-mnist_train.csv')
fashion_test_df=pd.read_csv('fashion-mnist_test.csv')


# In[218]:


fashion_train_df.head


# In[219]:


fashion_train_df.shape


# In[220]:


training=np.array(fashion_train_df,dtype='float32')
testing=np.array(fashion_test_df,dtype='float32')


# In[337]:


labels=['T-shirt/Top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
i=r.randint(1,60000)
plt.imshow(training[i,1:].reshape((28,28)))
print(labels[int(training[i,0])])


# In[328]:


X_train=training[:,1:]/255
y_train=training[:,0:]
X_test=testing[:,1:]/255
y_test=testing[:,0]


# In[340]:


from sklearn.model_selection import train_test_split
X_train, X_validate,y_train,y_validate=train_test_split(X_train,y_train, test_size = 0.3,random_state=101)


# In[342]:


X_train=X_train.reshape((X_train.shape[0],28,28,1))
X_test=X_test.reshape((X_test.shape[0],28,28,1))
X_validate=X_validate.reshape(X_validate.shape[0],*(28,28,1))


# In[ ]:




