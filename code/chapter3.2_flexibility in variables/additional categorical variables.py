# In[ ]:

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# In[ ]:

nobs = 120 # number of observations
nsim = 100 # number of simulations 
max_man = 20 # maximum number of additionally measured manipulated variables (categorical)

# In[ ]:

# ANOVA: all conditions differ at once
denom_res = np.array([[0.0]*max_man]*nsim)
numer_res = np.array([[0.0]*max_man]*nsim)
prob_res = np.array([[0.0]*max_man]*nsim)
numer = []
for a in range(nsim):
    data = np.random.normal(loc=0, scale=1, size=nobs)
    hit, trial = 0, 0
    numer, denom = [],[]
    for i in range(max_man):
        man_var = np.round(np.random.randint(low=0,high=4,size=nobs),1) # manipulated variable
        df = pd.DataFrame([data,man_var], index=['data', 'conditions']).transpose()
        result = stats.f_oneway(df.query('conditions==0').data,df.query('conditions==1').data,
                               df.query('conditions==2').data, df.query('conditions==3').data,)
        if result[1] <= 0.05:
            hit += 1
        trial += 1
        numer.append(hit)
        denom.append(trial)
    denom_res[a,:], numer_res[a,:], prob_res[a,:] = denom, numer, [numer[z]/denom[z] for z in range(len(numer))]

# In[ ]:

ANOVA_df = pd.DataFrame([
    [np.mean(prob_res[:,i]) for i in range(max_man)],
    [np.mean(numer_res[:,i]) for i in range(max_man)],
    [np.mean(denom_res[:,i]) for i in range(max_man)]],
    index=['probability','hit_count','trial_count']).transpose()

# In[ ]:

# t-test: detect each single mean difference 
denom_res = np.array([[0.0]*max_man]*nsim)
numer_res = np.array([[0.0]*max_man]*nsim)
prob_res = np.array([[0.0]*max_man]*nsim)

for a in range(nsim):
    hit,trial = 0,0
    numer, denom = [],[]
    data = np.random.normal(loc=0, scale=1, size=nobs)
    for i in range(max_man):
        man_var = np.round(np.random.randint(low=0,high=4,size=nobs),1) # manipulated variable
        df = pd.DataFrame([data,man_var], index=['data', 'condition']).transpose()
        result1 = stats.ttest_ind(a=df.query('condition==0').data, b=df.query('condition==1').data)[1]
        if result1 <= 0.05:
            hit += 1
        result2 = stats.ttest_ind(a=df.query('condition==0').data, b=df.query('condition==2').data)[1]
        if result2 <= 0.05:
            hit += 1
        result3 = stats.ttest_ind(a=df.query('condition==0').data, b=df.query('condition==3').data)[1]
        if result3 <= 0.05:
            hit += 1
        result4 = stats.ttest_ind(a=df.query('condition==1').data, b=df.query('condition==2').data)[1]
        if result4 <= 0.05:
            hit += 1
        result5 = stats.ttest_ind(a=df.query('condition==1').data, b=df.query('condition==3').data)[1]
        if result5 <= 0.05:
            hit += 1
        result6 = stats.ttest_ind(a=df.query('condition==2').data, b=df.query('condition==3').data)[1]
        if result6 <= 0.05:
            hit += 1
        trial += 6
        numer.append(hit)
        denom.append(trial)
    denom_res[a,:], numer_res[a,:], prob_res[a,:] = denom, numer, [numer[z]/denom[z] for z in range(len(numer))]

# In[ ]:

t_df = pd.DataFrame([
    [np.mean(prob_res[:,i]) for i in range(max_man)],
    [np.mean(numer_res[:,i]) for i in range(max_man)],
    [np.mean(denom_res[:,i]) for i in range(max_man)]],
    index=['probability','hit_count','trial_count']).transpose()

# In[ ]:

with sns.axes_style('darkgrid'):
    plt.plot(t_df.probability)
    plt.plot(ANOVA_df.probability)

# In[ ]:

with sns.axes_style('darkgrid'):
    plt.plot(t_df.hit_count)
    plt.plot(ANOVA_df.hit_count)

# In[ ]:

with sns.axes_style('darkgrid'):
    plt.plot(t_df.trial_count)
    plt.plot(ANOVA_df.trial_count)

# In[ ]:

# import dataframes from additional continuous variables 
df_30r = pd.read_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\added_nobs30_rtest.csv')
df_200r = pd.read_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\added_nobs200_rtest.csv')
df_30t = pd.read_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\added_nobs30_ttest.csv')
df_200t = pd.read_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\added_nobs200_ttest.csv')

# In[ ]:

x = np.arange(1,21)
with sns.axes_style('darkgrid'):
    ax = sns.lineplot(x=x, y=df_30r.hit_count, color='darkturquoise', label='continuous variable, correlation test')
    sns.lineplot(x=x, y=df_30t.hit_count, color='darkcyan', label='continuous variable, t-test')
    sns.lineplot(x=x, y=ANOVA_df.hit_count, color='orangered', label='categorical variable, ANOVA')
    sns.lineplot(x=x, y=t_df.hit_count, color='firebrick', label='categorical variable, t-test')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set(xlim=(1,20),xlabel='number of independent variables',ylabel='number of positive findings')
plt.savefig(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\plots\chapter4\plot3')

# In[ ]:

with sns.axes_style('darkgrid'):
    ax = sns.lineplot(x=x, y=df_30r.probability, color='darkturquoise', label='continuouse variable, correlation test')
    sns.lineplot(x=x, y=df_30t.probability, color='darkcyan', label='continuous variable, t-test')
    sns.lineplot(x=x, y=ANOVA_df.probability, color='orangered', label='categorical variable ANOVA')
    sns.lineplot(x=x, y=t_df.probability, color='firebrick', label='categorical variable, t-test')    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set(xlim=(1,20),xlabel='number of independent variables',ylabel='probaiblity of positive finding')
