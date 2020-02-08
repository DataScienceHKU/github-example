
# coding: utf-8

# In[2]:


import numpy as np # library for vectorized computation
import pandas as pd # library to process data as dataframes


# In[47]:


# extract data from downloaded postal information
df = pd.read_csv(r'C:\Users\daryle\Desktop\Canada.csv', header = None)

# assign column names
df.columns = ['Postcode', 'Borough', 'Neighbourhood']

# Ignore cells with a borough that is Not assigned
df = df[~df['Borough'].isin(['Not assigned'])]

# Rename neighborhood by borough if Not assigned neighborhood
for i in range(df.shape[0]):
    if (df.iloc[i,2])=='Not assigned':
        df.iloc[i,2]=df.iloc[i,1]

# Groupby "Postcode" into one row, and join the Neighbourhood cell togehter with a comma
df=df.groupby('Postcode')['Neighbourhood'].apply(', '.join).reset_index()

# Return shape
df.shape

