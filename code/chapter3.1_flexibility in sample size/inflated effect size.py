# In[ ]:

import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
# from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.power import tt_ind_solve_power

# set parameters 
alpha = 0.05
power1 = 0.9
power2 = 0.3
mean1 = 0
mean2 = 1
sd1, sd2 = 1,1
effect_size = (mean2-mean1/sd1)

# In[ ]:

# create function to calculate cohens_d
def cohen(n1, n2, mean1, mean2, s1, s2 ):
#     pooled_sd = (((n1-1)*s1**2)+((n2-1)*s2**2)/(n1+n2-2))**0.5
    pooled_sd = np.sqrt((sd1**2+sd2**2)*0.5)
    cohen_d = ((mean2-mean1)/pooled_sd)
    return cohen_d

# In[ ]:

# compute sample size, according to predefined power levels 
# power_analysis = TTestIndPower
size1 = tt_ind_solve_power(effect_size=effect_size, power=power1, alpha=alpha, alternative='two-sided')
size2 = tt_ind_solve_power(effect_size=effect_size, power=power2, alpha=alpha, alternative='two-sided')
# size1 = power_analysis.solve_power(effect_size = effect_size, power = power1, alpha = alpha, alternative = 'two-sided',self)
# size2 = power_analysis.solve_power(effect_size = effect_size, power = power2, alpha = alpha, alternative = 'two-sided')
sample_size1 = np.round(size1,0).astype(int)
sample_size2 = np.round(size2,0).astype(int)
print(sample_size1)
print(sample_size2)
print(effect_size)

# In[ ]:

nsim = 10000 # 10000 simulations 
cohen_d1 = []
cohen_d2 = []
results1 = []
results2 = []
for i in range(0,nsim):
    # compute samples 
    sample1_1 = np.random.normal(loc=mean1, scale=sd1, size=sample_size1) 
    sample1_2 = np.random.normal(loc=mean2, scale=sd2, size=sample_size1)
    sample2_1 = np.random.normal(loc=mean1, scale=sd1, size=sample_size2)
    sample2_2 = np.random.normal(loc=mean2, scale=sd2, size=sample_size2)
    # measure standardized deviation: cohen_d
    d1 = cohen(mean1=np.mean(sample1_1), mean2=np.mean(sample1_2), n1=sample_size1, n2=sample_size1, 
                    s1=np.std(sample1_1), s2=np.std(sample1_2))
    d2 = cohen(mean1=np.mean(sample2_1), mean2=np.mean(sample2_2), n1=sample_size2, n2=sample_size2,
                    s1=np.std(sample2_1), s2=np.std(sample2_2))
    # perform two-tailed two-sample t-test and store p-values 
    result1 = stats.ttest_rel(a = sample1_2, b = sample1_1)[1]#, equal_var=True)[1]
    result2 = stats.ttest_rel(a = sample2_2, b = sample2_1)[1]#, equal_var=True)[1]
    # store values 
    cohen_d1.append(d1)
    cohen_d2.append(d2)
    results1.append(result1)
    results2.append(result2)

# In[ ]:

df1 = pd.DataFrame([cohen_d1, results1], index=['d1', 'results1']).transpose()
df2 = pd.DataFrame([cohen_d2, results2], index=['d2', 'results2']).transpose()

# In[ ]:

# search insignificant d
min1 = df1.query('results1 > 0.05').results1.min()
for i in range(len(df1)):
    if df1.iloc[i,1] == min1:
        print(df1.iloc[i,:])
        mind1 = df1.iloc[i,0]
min2 = df2.query('results2 > 0.05').results2.min()
for i in range(len(df2)):
    if df2.iloc[i,1] == min2:
        print(df2.iloc[i,:])
        mind2 = df2.iloc[i,0]

# In[ ]:

max1 = df1.query('results1 <= 0.05').results1.max()
for i in range(len(df1)):
    if df1.iloc[i,1] == max1:
        print(df1.iloc[i,:])
        maxd1 = df1.iloc[i,0]
max2 = df2.query('results2 <= 0.05').results2.max()
for i in range(len(df2)):
    if df2.iloc[i,1] == max2:
        print(df2.iloc[i,:])
        maxd2 = df2.iloc[i,0]

# In[ ]:

lower, upper = -1,3
with sns.axes_style('darkgrid'):
    _, bins, _ = plt.hist(x=df1.d1, density=1, bins=50, color='lightseagreen')
    mu, sigma = stats.norm.fit(df1.d1)
    curve = stats.norm.pdf(bins, mu, sigma)
    plt.plot(bins,curve, color='navy')
    plt.axvline(mind1,color='maroon')
    plt.text(s='smallest \nsignificant \neffect',x=0,y=1, color='maroon', fontsize='large')
    plt.xlim(lower,upper)
    plt.xlabel('cohen`s d')
    plt.ylabel('density')
    plt.text(s='power = 90%',x=1.8,y=1.2,fontsize='x-large')
plt.savefig(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\plots\chapter3.1\plot1')

# In[ ]:

lower, upper = -1,3
with sns.axes_style('darkgrid'):
    _, bins, _ = plt.hist(x=df2.d2, density=1, bins=50, color='lightseagreen')
    mu, sigma = stats.norm.fit(df2.d2)
    curve = stats.norm.pdf(bins, mu, sigma)
    plt.plot(bins,curve, color='navy')
    plt.axvline(mind2,color='maroon')
    plt.text(s='smallest \nsignificant \neffect',x=1.8,y=0.5, color='maroon', fontsize='large')
    plt.xlim(lower,upper)
    plt.xlabel('cohen`s d')
    plt.ylabel('density')
    plt.text(s='power = 30%',x=-.9,y=.6,fontsize='x-large')
plt.savefig(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\plots\chapter3.1\plot2')
