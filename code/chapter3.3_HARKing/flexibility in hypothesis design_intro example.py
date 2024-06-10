# In[ ]:

import seaborn as sns
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

np.random.seed(0)

mu = 0
sigma = 1
size = 10000
data = np.random.normal(mu, sigma, size)

t = [((np.mean(data)-mu)/sigma)*np.sqrt(len(data))]

critical_two = [stats.norm.ppf(loc=mu, scale=sigma, q=0.025), stats.norm.ppf(loc=mu, scale=sigma, q=0.975)]
critical_one = [stats.norm.ppf(loc=mu, scale=sigma, q=0.05), stats.norm.ppf(loc=mu, scale=sigma, q=0.95)]

# In[ ]:

x = np.linspace(start=mu-2.5*sigma, stop=mu+2.5*sigma, num=len(data))
curve = stats.norm.pdf(x=x, loc=mu, scale=sigma)
with sns.axes_style('darkgrid'):
    g = sns.lineplot(x=x,y=curve, color='lightseagreen', label='null distribution')
    sns.scatterplot(x=critical_two ,y=0, label='two-tailed critical values', color='maroon', marker='s')
    sns.scatterplot(x=critical_one, y=0, label='one-tailed critical values', color='orangered', marker='s')
    sns.scatterplot(x=t, y=0, label='test statistic', color='navy', s=60)
    g.legend(loc='upper right', bbox_to_anchor=(1.2,1))
    g.set(xlabel='test statistic', ylabel='density')
plt.savefig(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\plots\chapter5\plot1.png')
