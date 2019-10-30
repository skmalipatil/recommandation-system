#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pa
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


credit_score = pa.read_csv(r'C:\Users\samalipa\Desktop\project\moviedataset\tmdb_5000_credits.csv\tmdb_5000_credits.csv')


# In[3]:


movies_list = pa.read_csv(r'C:\Users\samalipa\Desktop\project\moviedataset\tmdb_5000_movies.csv\tmdb_5000_movies.csv')


# In[4]:


credit_score.head()


# In[5]:


movies_list.head()


# In[6]:


print('credit score shape :', credit_score.shape)
print('movie list shape :', movies_list.shape)


# #### merging of the two tables on movie id

# In[7]:


credit_column_renamed = credit_score.rename(index= str, columns={"movie_id" : "id"})
movies_list_merge = movies_list.merge(credit_column_renamed, on = 'id')
movies_list_merge.head(1)


# In[8]:


movies_list_merge.shape


# In[9]:


movies_list_merge = movies_list_merge.drop(columns = ['homepage', 'title_x', 'title_y', 'status', 'production_countries'])
movies_list_merge.head()


# In[10]:


movies_list_merge = movies_list_merge.drop('tagline', axis=1)


# In[11]:


movies_list_merge.shape


# In[12]:


movies_list_merge.info()


# In[13]:


movies_list_merge.isnull().sum()


# In[14]:


movies_list_merge.head(2)


# ### USING WEIGHTED AVERAGE FOR EACH MOVIE'S AVERAGE RATING

# In[15]:


v = movies_list_merge['vote_count']
r = movies_list_merge['vote_average']
c = movies_list_merge['vote_average'].mean()
m = movies_list_merge['vote_count'].quantile(0.7)


# In[16]:


movies_list_merge['weighted_average'] = ((r*v)+(c*m))/(v+m)
movies_list_merge.head()


# In[17]:


movies_sorted_ranking = movies_list_merge.sort_values('weighted_average', ascending = False)
movies_sorted_ranking[['original_title', 'vote_count', 'vote_average', 'weighted_average', 'popularity']].head(20)


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sb


# #### as we can see even popularity will also decide movies people watch we can give consideration to popularity

# ### scaling down the weighted average and popularity 

# In[19]:


movies_clean_df = movies_list_merge


# In[20]:


from sklearn.preprocessing import MinMaxScaler

scaling = MinMaxScaler()
movie_scaled = scaling.fit_transform(movies_clean_df[['weighted_average', 'popularity']])
movie_normalize = pa.DataFrame(movie_scaled, columns=['weighted_average', 'popularity'])
movie_normalize.head()


# In[21]:


movies_clean_df[['normalized weighted average', 'normalized popularity']] = movie_normalize


# In[22]:


movies_clean_df.head(2)


# In[23]:


movies_clean_df['score'] = movies_clean_df['normalized weighted average'] * 0.5 + movies_clean_df['normalized popularity']*0.5
movies_normalized_score = movies_clean_df.sort_values(['score'], ascending=False)
movies_normalized_score[['original_title', 'normalized weighted average', 'normalized popularity', 'score']].head()


# In[25]:


scored = movies_normalized_score


plt.figure(figsize = (16, 9))

ax = sb.barplot(x = scored['score'].head(10), y = scored['original_title'].head(10), data = scored)

plt.title= ("best rated & popular movies")
plt.xlabel('score', weight= 'bold')
plt.ylabel('original_title', weight= 'bold')


# In[ ]:





# In[ ]:




