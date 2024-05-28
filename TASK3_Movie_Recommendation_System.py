#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd


# In[17]:


data = {
    'user_id': [1, 1, 1, 2, 2],
    'item_id': ['A', 'B', 'D', 'B', 'C'],
    'rating': [5, 4, 1, 3, 2]
}


# In[18]:


df = pd.DataFrame(data)


# In[19]:


user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)


# In[20]:


item_similarity = user_item_matrix.corr(method='pearson')


# In[21]:


def get_recommendations(user_id, top_n=5):
    user_ratings = user_item_matrix.loc[user_id]
    similar_items = pd.Series()
    
    for item_id, rating in user_ratings.iteritems():
        similar_items = similar_items.append(item_similarity[item_id].drop(item_id).mul(rating))
    
    similar_items = similar_items.groupby(similar_items.index).sum()
    similar_items = similar_items.sort_values(ascending=False)
    
    unrated_items = similar_items.drop(user_ratings.index)
    recommendations = unrated_items.head(top_n)
    
    return recommendations


# In[28]:


user_id = 2
recommendations = get_recommendations(user_id)
print(f"Top 5 recommendations for user {user_id}: {recommendations}")


# In[ ]:




