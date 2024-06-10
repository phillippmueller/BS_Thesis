# In[ ]:

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from itertools import combinations 

# In[ ]:

nobs = 30
nvar = 20
nsim = 1000

# In[ ]:

denom_res = np.array([[0.0]*nvar]*nsim)
numer_res = np.array([[0.0]*nvar]*nsim)
prob_res = np.array([[0.0]*nvar]*nsim)
for a in range(nsim):
    df,columns = pd.DataFrame(np.random.normal(loc=0,scale=2,size=nobs)),[0]
    denom, numer = [],[] 
    for i in range(nvar):
        columns.append(i+1)
        df.insert(loc=i+1,column=i+1,value=np.random.normal(loc=0,scale=2,size=nobs))
        df.columns=columns
        tuples = list(combinations(columns,2))
        denom.append(len(tuples))
        count = []
        for j in range(len(tuples)):
            sample = tuples[j]
            result = (stats.pearsonr(x=df[sample[0]],y=df[sample[1]]))[1] # correlation test
#             result = stats.ttest_rel(a=df[sample[0]],b=df[sample[1]])[1] # mean difference test
            if result <= 0.05:
                count.append(1)
            else:
                count.append(0)
        numer.append(np.sum(count))
    denom_res[a,:], numer_res[a,:], prob_res[a,:] = denom, numer, [numer[z]/denom[z] for z in range(len(numer))]

# In[ ]:

result_df = pd.DataFrame([
    [np.mean(prob_res[:,i]) for i in range(nvar)],
    [np.mean(numer_res[:,i]) for i in range(nvar)],
    [np.mean(denom_res[:,i]) for i in range(nvar)]],
    index=['probability','hit_count','trial_count']).transpose()

# In[ ]:

nobs = 200
nvar = 20
nsim = 1000

# In[ ]:

denom_res = np.array([[0.0]*nvar]*nsim)
numer_res = np.array([[0.0]*nvar]*nsim)
prob_res = np.array([[0.0]*nvar]*nsim)
for a in range(nsim):
    df,columns = pd.DataFrame(np.random.normal(loc=0,scale=2,size=nobs)),[0]
    denom, numer = [],[] 
    for i in range(nvar):
        columns.append(i+1)
        df.insert(loc=i+1,column=i+1,value=np.random.normal(loc=0,scale=2,size=nobs))
        df.columns=columns
        tuples = list(combinations(columns,2))
        denom.append(len(tuples))
        count = []
        for j in range(len(tuples)):
            sample = tuples[j]
            result = (stats.pearsonr(x=df[sample[0]],y=df[sample[1]]))[1] # correlation test
#             result = stats.ttest_rel(a=df[sample[0]],b=df[sample[1]])[1] # mean difference test
            if result <= 0.05:
                count.append(1)
            else:
                count.append(0)
        numer.append(np.sum(count))
    denom_res[a,:], numer_res[a,:], prob_res[a,:] = denom, numer, [numer[z]/denom[z] for z in range(len(numer))]

# In[ ]:

second_result_df = pd.DataFrame([
    [np.mean(prob_res[:,i]) for i in range(nvar)],
    [np.mean(numer_res[:,i]) for i in range(nvar)],
    [np.mean(denom_res[:,i]) for i in range(nvar)]],
    index=['probability','hit_count','trial_count']).transpose()

# In[ ]:

# mean values:

# saving ttest results as csv
# result_df.to_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\added_nobs30_ttest.csv')
# second_result_df.to_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\added_nobs200_ttest.csv')

# saving correlation test results as csv
result_df.to_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\added_nobs30_rtest.csv')
second_result_df.to_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\added_nobs200_rtest.csv')

# In[ ]:

df_30r = pd.read_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\added_nobs30_rtest.csv')
df_200r = pd.read_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\added_nobs200_rtest.csv')
df_30t = pd.read_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\added_nobs30_ttest.csv')
df_200t = pd.read_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\added_nobs200_ttest.csv')

# In[ ]:

x = np.arange(2,22)
with sns.axes_style('darkgrid'):
    ax = sns.lineplot(x=x, y=df_30r.hit_count, color='darkcyan', label='correlation test, nobs=30')
    sns.lineplot(x=x, y=df_200r.hit_count, color='darkturquoise', label='correlation test, nobs=200')
    sns.lineplot(x=x, y=df_30t.hit_count, color='firebrick', label='t-test nobs=30')
    sns.lineplot(x=x, y=df_200t.hit_count, color='orangered', label='t-test nobs=200')
    ax.set(xlim=(2,10), ylim=(0,3), xlabel='number of variables', ylabel='number of significant findings')
plt.savefig(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\plots\chapter4\plot2.png')

# In[ ]:

with sns.axes_style('darkgrid'):
    ax = sns.lineplot(x=x, y=df_30r.probability, color='darkcyan', label='correlation test, nobs=30')
    sns.lineplot(x=x, y=df_200r.probability, color='darkturquoise', label='correlation test, nobs=200')
    sns.lineplot(x=x, y=df_30t.probability, color='firebrick', label='t-test nobs=30')
    sns.lineplot(x=x, y=df_200t.probability, color='orangered', label='t-test nobs=200')
    ax.set(xlim=(2,21),ylim=(0,0.1),xlabel='number of variables', ylabel='percentage of significant findings')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
plt.savefig(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\plots\chapter4\plot1.png')
