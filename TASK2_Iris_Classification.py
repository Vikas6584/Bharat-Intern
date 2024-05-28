#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')


# In[30]:


get_ipython().system('pip install numpy==1.26.4')


# In[18]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[19]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)


# In[20]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[21]:


le = LabelEncoder()
y = le.fit_transform(y)


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[24]:


mlp_clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
mlp_clf.fit(X_train, y_train)


# In[25]:


print("MLP Classifier (scikit-learn):")
print(classification_report(y_test, mlp_clf.predict(X_test)))


# In[26]:


model = Sequential([
    Dense(10, activation='relu', input_shape=(4,)),
    Dense(3, activation='softmax')
])


# In[27]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[28]:


history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)


# In[29]:


print("\nTensorFlow Model:")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", test_accuracy)


# In[ ]:




