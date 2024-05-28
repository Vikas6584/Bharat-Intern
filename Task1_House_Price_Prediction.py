#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


data = pd.read_csv(r"kc_house_data.csv")


# In[12]:


data.head()


# In[13]:


data.describe()


# In[14]:


data.info()


# In[15]:


data.isnull().sum()


# In[16]:


data = data.drop(['id','date'],axis=1)


# In[17]:


data.head()


# In[19]:


data['bedrooms'].value_counts().plot(kind='bar')
plt.title('Number of Bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine()


# In[20]:


plt.figure(figsize=(10,10))
sns.jointplot(x=data.lat.values,y =data.long.values, height=10)
plt.ylabel('Longitude',fontsize=12)
plt.xlabel('Latitude',fontsize=12)
sns.despine()
plt.show()


# In[21]:


plt.scatter(data.price,data.sqft_living)
plt.title("Price of square feet")


# In[22]:


plt.scatter(data.price,data.long)
plt.title("Price vs location of the area")


# In[23]:


plt.scatter(data.price,data.lat)
plt.xlabel("Price")
plt.ylabel("Latitude")
plt.title("Latitude vs Price")


# In[24]:


plt.scatter(data.bedrooms,data.price)
plt.title("Bedroom and Price")
plt.xlabel("Bedrooms")

plt.ylabel("Price")
sns.despine()
plt.show()


# In[25]:


plt.scatter((data['sqft_living']+data['sqft_basement']),data['price'])


# In[26]:


plt.scatter(data.waterfront,data.price)
plt.title("Waterfront vs Price (0= no waterfront)")


# In[27]:


y=data['price']
X=data.drop(['price'],axis=1)


# In[29]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.10,random_state=42)


# In[30]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
reg.score(x_test,y_test)


# In[ ]:




