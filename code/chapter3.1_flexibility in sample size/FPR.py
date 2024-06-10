# In[ ]:

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# In[ ]:

nsim = 1000
size = 100

# minsize = 10
# minsize = 20
minsize = 30

alpha = 0.01
# alpha = 0.05
# alpha = 0.1

# In[ ]:

# deleting outliers 
count = []
for n in range(nsim):
    df1 = pd.DataFrame([np.random.normal(0,1,size)]).transpose()
    df2 = pd.DataFrame([np.random.normal(0,1,size)]).transpose()
    data1, data2 = df1.values.tolist(), df2.values.tolist()
    hit = 0
    for i in range(minsize,size+1):
        data1.remove(np.min(data1))
        data2.remove(np.max(data2))
        p = stats.ttest_rel(a=data1, b=data2)[1]
        if p <= alpha:
            hit += 1
    count.append(hit)

# In[ ]:

# intermediate testing 
inter_count = []
for n in range(nsim):
    df1 = pd.DataFrame([np.random.normal(0,1,size)]).transpose()
    df2 = pd.DataFrame([np.random.normal(0,1,size)]).transpose()
    data1, data2 = df1.values.tolist(), df2.values.tolist()
    hit = 0
    for i in range(minsize,size):
        del (data1[len(data1)-1])
        del (data2[len(data2)-1])
        p = stats.ttest_rel(a=data1, b=data2)[1]
        if p <= alpha:
            hit += 1
    inter_count.append(hit)

# In[ ]:

df = pd.DataFrame(data = [inter_count, count],index = ['intermediate','outlier']).transpose()
FPR_outlier = len(df.query('outlier > 0'))/nsim
FPR_intermediate = len(df.query('intermediate > 0.05'))/nsim

[FPR_intermediate, FPR_outlier]

# In[ ]:

# arbitrary sample size 
arb_result = np.array([[0.0]*(size-minsize)]*nsim)
for n in range(nsim):
    arb = []
    for i in range(minsize,size):
        data1 = np.random.normal(loc=0, scale=1, size=i)
        data2 = np.random.normal(loc=0, scale=1, size=i)
        p = stats.ttest_rel(data1,data2)[1] 
        arb.append(p)
    arb_result[n,:] = arb
arb_count = []
for i in range(size-minsize): 
    hit = 0
    for n in range(nsim):
        if arb_result[n,i] <= alpha: 
            hit += 1
    arb_count.append(hit)
np.mean([arb_count[i]/nsim for i in range(size-minsize)])
