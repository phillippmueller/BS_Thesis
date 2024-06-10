# In[ ]:

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# In[ ]:

# null hypothesis: N(0,1)
mu = 0
sigma = 1
alpha = 0.05

# sample data
mean = 3
sd = 1
num=1000

# beta = 0.2
critical_value = stats.norm.ppf(0.975) # alpha = 5%, two-tailed
rejection_area = np.linspace(critical_value, mu+5*sigma, num=num)
beta_area = np.linspace(mu-1*sigma, critical_value, num=num)

x = np.linspace(mu-1*sigma, mu+5*sigma, num=num)
curve = stats.norm.pdf(x=x, loc=mu, scale=sigma)
sample_curve = stats.norm.pdf(x=x, loc=mean, scale=sd)

# In[ ]:

with sns.axes_style('darkgrid'):
    # plot null distribution
    ax = sns.lineplot(x=x, y=curve, color='orangered')
    # plot sample distribution 
    sns.lineplot(x=x, y=sample_curve, color='lightseagreen')
    #shade in error areas 
    ax.fill_between(x=beta_area, y1=stats.norm.pdf(x=beta_area, loc=mean, scale=sd),
                    color='paleturquoise')
    ax.fill_between(x=rejection_area, y1=stats.norm.pdf(x=rejection_area, loc=mu, scale=sigma), 
                    color='salmon')
    plt.axvline(x=critical_value, color='black')
    plt.text(s='critical \nvalue', x=1.2, y=0.3)
    # plot expected value under H0
    plt.axvline(x=mu, color='orangered')
    plt.text(s='H0:\nmean=0', x=0.1, y=0.15, color='orangered')
    plt.axvline(x=mean, color='lightseagreen')
    plt.text(s='H1:\nmean=3', x=3.1, y=0.15 ,color='lightseagreen')
    ax.set(xlabel='test statistic', ylabel='density')
    plt.title('alpha=5%')
plt.savefig(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\plots\chapter3.2\plot1')

# In[ ]:

power = stats.norm.sf(x=critical_value, loc=mean, scale=sd)
beta = stats.norm.cdf(x=critical_value, loc=mean, scale=sd)
print(power)
print(beta)
power + beta 

# In[ ]:

# increase alpha area to show trade-off
new_alpha = 0.2
new_crit = stats.norm.isf(new_alpha/2)
print(critical_value)
print(new_crit)

new_alpha_area = np.linspace(new_crit, mu+5*sigma, num=num)
new_beta_area = np.linspace(mu-1*sigma, new_crit, num=num)

# In[ ]:

with sns.axes_style('darkgrid'):
    ax = sns.lineplot(x=x, y=curve, color='orangered')
    sns.lineplot(x=x, y=sample_curve, color='lightseagreen')
    ax.fill_between(x=new_beta_area, y1=stats.norm.pdf(x=new_beta_area, loc=mean, scale=sd), color='paleturquoise') 
    ax.fill_between(x=new_alpha_area, y1=stats.norm.pdf(x=new_alpha_area, loc=mu, scale=sigma), color='salmon')
    plt.axvline(x=new_crit, color='black')
    plt.text(s='critical \nvalue', x=1.3, y=0.3)
    plt.axvline(x=mu, color='orangered')
    plt.text(s='H0:\nmean=0', x=0.1, y=0.15, color='orangered')
    plt.axvline(x=mean, color='lightseagreen')
    plt.text(s='H1:\nmean=3', x=3.1, y=0.15 ,color='lightseagreen')
    ax.set(xlabel='test statistic', ylabel='density')
    plt.title('alpha=20%')
plt.savefig(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\plots\chapter3.2\plot2')

# In[ ]:

new_power = stats.norm.sf(x=new_crit, loc=mean, scale=sd)
new_beta = stats.norm.cdf(x=new_crit, loc=mean, scale=sd)
print(new_power)
print(new_beta)
new_power + new_beta 
