# In[36]:

import pandas as pd
import numpy as np

# In[37]:

max_size = 101
max_sigma = 3
min_mean, max_mean = 0, 1
Group1 = np.array([[[0.0]*max_size]*(max_sigma)]*5)
Group2 = np.array([[0.0]*max_size]*(max_sigma))

# In[38]:

# create data
for i in range(5):
    for j in range(1,max_sigma+1):
        Group1[i,j-1,:] = np.random.normal(loc=i/4, scale=j, size=max_size)
for j in range(1,max_sigma+1):
    Group2[j-1,:] = np.random.normal(loc=0, scale=j, size=max_size)

# In[39]:

# reorganize data into vector format
data1 = [Group1[i,j-1,k] 
         for i in range(5) 
         for j in range(1,max_sigma+1)
         for k in range(max_size)]
data2 = [Group2[j-1,k] 
         for j in range(1,max_sigma+1) 
         for k in range(max_size)]

# In[40]:

# create multi-level index for dataframe
range_ = np.linspace(0,1,5) 
tuples1 = tuple([location,scale,size] 
               for location in range_ 
               for scale in range(1,max_sigma+1) 
               for size in range(max_size))
tuples2 = tuple([scale, size]
               for scale in range(1,max_sigma+1)
               for size in range(max_size))
index1 = pd.MultiIndex.from_tuples(tuples1, names=['location', 'scale', 'size'])
index2 = pd.MultiIndex.from_tuples(tuples2, names=['scale','size'])
# build dataframes 
df1 = pd.DataFrame(data1,index=index1)
df2 = pd.DataFrame(data2, index=index2)

# In[41]:

df1.to_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\df1.csv')
df2.to_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\df2.csv')
