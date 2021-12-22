#!/usr/bin/env python
# coding: utf-8

# In[71]:


import pandas as pd
from pandas import set_option
import numpy as py
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras import models
from keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


# In[72]:


training = pd.read_csv(r"C:\Users\Andrew\Downloads\training_set.csv", sep = r'\s*,\s*')
test = pd.read_csv(r"C:\Users\Andrew\Downloads\test_set.csv", sep = r'\s*,\s*')
url = 'https://raw.githubusercontent.com/werowe/logisticRegressionBestModel/master/KidCreative.csv'
url_2  = "https://github.com/adriangb/scikeras"


# In[73]:


X_train = training.drop(['class','ra','dec'],axis = 1)
Y_train = training['class']
X_test = test.drop(['class','ra','dec'],axis = 1)
Y_test = test['class']
encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_Y = encoder.transform(Y_train)
dummy_y = np_utils.to_categorical(encoded_Y)

encoder.fit(Y_test)
encoded_Y_test = encoder.transform(Y_test)
dummy_Y_test = np_utils.to_categorical(encoded_Y_test)

def baseline_model():
    model = models.Sequential()
    model.add(Dense(6, activation='relu', input_dim = 4))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model
model = baseline_model()


# In[74]:


model.fit(X_train,dummy_y,epochs=8,batch_size=1)


# In[75]:


score = model.evaluate(X_test,dummy_Y_test)


# In[90]:


from sklearn.metrics import classification_report, confusion_matrix
y_pred = py.argmax(model.predict(X_test),axis=1)+1
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))


# In[87]:





# In[44]:





# In[ ]:





# In[69]:





# In[23]:





# In[24]:





# In[28]:





# In[29]:





# In[30]:





# In[ ]:




