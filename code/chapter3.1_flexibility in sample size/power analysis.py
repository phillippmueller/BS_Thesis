# In[ ]:

from statsmodels.stats.power import normal_power 
import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import seaborn as sns

# In[ ]:

alpha = 0.05
mean_null = 3 # null hypothesis
sd = 3 # standard deviation in the population 
n = 10 # sample size 
se = sd/np.sqrt(n) # standard error of the sampling distribution
data = np.random.normal(loc=4.5, scale=3, size=n) # sample
d = (np.mean(data)-mean_null)/sd # effect size in cohens' d
z = (np.mean(data)-mean_null)/se # test statistic in one-sample z-test

# In[ ]:

c = stats.norm.isf(q=alpha) # identical to statsmodels computation
c_shifted = c - d*np.sqrt(n)/1 # in statsmodels: sigma=1, shifted in amount of z-score
c_null = stats.norm.isf(q=alpha, loc=mean_null, scale=se)
power = stats.norm.sf(c_shifted)

# In[ ]:

pow_ = normal_power(effect_size=d, nobs=n, alpha=alpha, alternative='larger')

# In[ ]:

# build plot
power_area = np.linspace(c_null, np.mean(data)+5*se, num=1000)
pow_area = np.linspace(c_shifted, 5, num=1000)

x = np.linspace(start=-4, stop=9, num=1000)
null_dist = stats.norm.pdf(x=x, loc=mean_null, scale=se) # sampling distribution under H0
alternative_dist = stats.norm.pdf(x=x, loc=np.mean(data), scale=se) # sampling distirbution under sample
sample_dist = stats.norm.pdf(x=x, loc=0, scale=1) # sample distribution in z-test
with sns.axes_style('darkgrid'):
    ax1 = sns.lineplot(x=x, y=sample_dist, color='seagreen',label='null distribution')
    ax1.fill_between(x=pow_area, y1=stats.norm.pdf(x=pow_area, loc=0, scale=1),
                     color='paleturquoise')
    ax2 = sns.lineplot(x=x, y=null_dist, color='orangered', label='probability under H0')
    ax3 = sns.lineplot(x=x, y=alternative_dist, color='lightseagreen', label='probability under H1')
    ax3.fill_between(x=power_area, y1=stats.norm.pdf(x=power_area, loc=np.mean(data), scale=se), 
                     color='paleturquoise', label='power')
    plt.axvline(c_null, color='orangered') # critical value under the null distribution 
    plt.axvline(c, color='seagreen') # critical value under the sample distribution
    plt.text(y=0.455, x=1.6, s='c',color='seagreen')
    plt.text(y=0.455, x=4.5, s='c_sample', color='orangered')
    plt.legend(bbox_to_anchor=(0.7,0.9))
    ax1.set(xlabel='test statistic', ylabel='density')
plt.savefig(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\plots\chapter3.2\plot3.png')

# In[ ]:

for i in range(x.size):
    if x[i] > c_null and x[i-1] <= c_null:
        index = i
print(index)
print(f'c={c_null}, x[index]={x[index]}')
print(f'diff={x[index]-c_null}')

# In[ ]:

power_by_integral = integrate.simps(y=alternative_dist[index:alternative_dist.size+1], x=x[index:x.size+1])

# In[ ]:

print(f'power = {power}, sigma=1')
print(f'pow_ = {pow_}, sigma=1')
print(f'integral = {power_by_integral}')
1-power
