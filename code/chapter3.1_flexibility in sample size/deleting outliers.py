# In[ ]:

import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.power import normal_power 

# In[ ]:

df1 = pd.read_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\df1.csv').query('location==0.5 and scale==1')
df2 = pd.read_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\df2.csv').query('scale==1')
df0 = pd.read_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\df1.csv').query('location==0 and scale==1')

# In[ ]:

# remove latest added observation
data1 = df1.iloc[:,3].values.tolist()
data2 = df2.iloc[:,2].values.tolist()
t_result = [stats.ttest_ind(a=data1, b=data2)[0]]
p_result = [stats.ttest_ind(a=data1, b=data2)[1]]
# test
while len(data1) > 2:
    del data1[len(data1)-1]
    del data2[len(data2)-1]
    t = stats.ttest_ind(a=data1, b=data2)[0]
    p = stats.ttest_ind(a=data1, b=data2)[1]
    t_result.append(t), p_result.append(p)
# store results
df = pd.DataFrame(data=[t_result, p_result], index=['statistic', 'p_value'])
bac = df.transpose()
#restore data
data1 = df1.iloc[:,3].values.tolist()
data2 = df2.iloc[:,2].values.tolist()

# In[ ]:

# remove next outlier
data0 = df0.iloc[:,3].values.tolist()
data1 = df1.iloc[:,3].values.tolist()
data2 = df2.iloc[:,2].values.tolist()
t_result = [stats.ttest_ind(a=data1, b=data2)[0]]
t0_result = [stats.ttest_ind(a=data0, b=data2)[0]]
p_result = [stats.ttest_ind(a=data1, b=data2)[1]]
p0_result = [stats.ttest_ind(a=data0, b=data2)[1]]
upper_c = [stats.t.isf(q=0.025, loc=0, scale=1, df=100)]
lower_c = [stats.t.ppf(q=0.025, loc=0, scale=1, df=100)]
mean1, mean2, mean0 = np.mean(data1), np.mean(data2), np.mean(data0)
print(f'mean0={np.mean(data0)}')
print(f'mean1={np.mean(data1)}')
print(f'mean2={np.mean(data2)}')
# test
while len(data1) > 2:
    data1.remove(np.min(data1))
    data2.remove(np.max(data2))
    t = stats.ttest_ind(a=data1, b=data2)[0]
    p = stats.ttest_ind(a=data1, b=data2)[1]
    t_result.append(t), p_result.append(p)
    # compute critical value 
    upper_c.append(stats.t.isf(q=0.025, loc=0, scale=1, df=len(data1)))
    lower_c.append(stats.t.ppf(q=0.025, loc=0, scale=2, df=len(data1)))
data1 = df1.iloc[:,3].values.tolist()
data2 = df2.iloc[:,2].values.tolist()
data0 = df0.iloc[:,3].values.tolist()
while len(data0) > 2:
    data0.remove(np.min(data0))
    data2.remove(np.max(data2))
    t0 = stats.ttest_ind(a=data0, b=data2)[0]
    p0 = stats.ttest_ind(a=data0, b=data2)[1]
    t0_result.append(t0), p0_result.append(p0)
# store results
df = pd.DataFrame(data=[t_result, p_result, t0_result, p0_result], 
                  index=['statistic', 'p_value', 'statistic_neg', 'p_value_neg'])
out = df.transpose()
#restore data
data0 = df0.iloc[:,3].values.tolist()
data1 = df1.iloc[:,3].values.tolist()
data2 = df2.iloc[:,2].values.tolist()

# In[ ]:

effect = mean1-mean2
std1, std2 = np.std(data1), np.std(data2)

d = np.round(effect / np.sqrt((std1**2+std2**2) / 2), 4)
print(f'H0=false: d={d}')

# In[ ]:

x = np.arange(0,98)
with sns.axes_style('darkgrid'):
    g = sns.lineplot(x=x, y=bac.statistic.iloc[0:98], color='crimson', label='delete latest oservation, H0=false')
    sns.lineplot(x=x, y=out.statistic.iloc[0:98], color='orangered', label='delete next outlier, H0=false')
    sns.lineplot(x=x, y=out.statistic_neg.iloc[0:98], color='lightseagreen', label='delete next outlier, H0=true')
    sns.lineplot(x=x, y=upper_c[0:98], color='lime', label='upper ciritical value at alpha=5%')
    g.set(xlabel='number of removed observations', ylabel='test statistic')
plt.savefig(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\plots\chapter3.3\plot9')

# In[ ]:

x = np.arange(0,100)
with sns.axes_style('darkgrid'):
    g = sns.lineplot(x=x, y=bac.p_value, color='crimson', label='delete latest oservation, H0=false')
    sns.lineplot(x=x, y=out.p_value, color='orangered', label='delete next outlier, H0=false')
    sns.lineplot(x=x, y=out.p_value_neg, color='lightseagreen', label='delete next outlier, H0=true')
    sns.lineplot(x=x, y=0.05, color='lime', label='alpha = 5%')
    g.set(xlabel='number of removed observations', ylabel='p-value')   

# In[ ]:

bac_max_p = np.max(bac.p_value)
bac_min_p = np.min(bac.p_value)
out_max_p = np.max(out.p_value)
out_min_p = np.min(out.p_value)

print(f'maximum p in backwards: p={bac_max_p}')
print(f'maximum p in outliers: p={out_max_p}')
print(f'delta in maximum p = {bac_max_p-out_max_p}')
print(f'minimum p in backwards: p={bac_min_p}')
print(f'minimum p in outliers: p={out_min_p}')
print(f'(bac/out)-ratio in minumum p = {bac_min_p/out_min_p}')

# In[ ]:

a,b = [],[]
for i in range(2):
    a.append(max(data1))
    b.append(min(data2))
stats.ttest_ind(a=a, b=b)

# In[ ]:

plt.plot(bac.p_value-out.p_value)
(bac.p_value-out.p_value)[60:80]

# In[ ]:

# varying effects
dfa = pd.read_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\df1.csv')
dfb = pd.read_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\df2.csv')

# In[ ]:

# create vectors to capture power
alpha = 0.05
power0, power1, power2, power3, power4, power5, power6 = [],[],[],[],[],[],[]

# In[ ]:

sample1 = dfa.query('location==0.75 and scale==1').iloc[:,3].values.tolist()
sample2 = dfa.query('location==0.5 and scale==1').iloc[:,3].values.tolist()
sample3 = dfa.query('location==0.75 and scale==2').iloc[:,3].values.tolist()
sample4 = dfa.query('location==0.5 and scale==2').iloc[:,3].values.tolist()
sample5 = dfa.query('location==0.25 and scale==2').iloc[:,3].values.tolist()
sample6 = dfa.query('location==0.25 and scale==3').iloc[:,3].values.tolist()
sample = dfa.query('location==0 and scale==3')
sample0 = sample.iloc[:,3].values.tolist()
baseline1 = dfb.query('scale==1').iloc[:,2].values.tolist()
baseline2 = dfb.query('scale==2').iloc[:,2].values.tolist()
baseline3 = dfb.query('scale==3').iloc[:,2].values.tolist()

d1 = np.round((np.mean(sample1)-np.mean(baseline1)) / np.sqrt((np.std(sample1)**2+np.std(baseline1)**2)/2),4)
d2 = np.round((np.mean(sample2)-np.mean(baseline1)) / np.sqrt((np.std(sample2)**2+np.std(baseline1)**2)/2),4)
d3 = np.round((np.mean(sample3)-np.mean(baseline2)) / np.sqrt((np.std(sample3)**2+np.std(baseline2)**2)/2),4)
d4 = np.round((np.mean(sample4)-np.mean(baseline2)) / np.sqrt((np.std(sample4)**2+np.std(baseline2)**2)/2),4)
d5 = np.round((np.mean(sample5)-np.mean(baseline2)) / np.sqrt((np.std(sample5)**2+np.std(baseline2)**2)/2),4)
d6 = np.round((np.mean(sample6)-np.mean(baseline3)) / np.sqrt((np.std(sample6)**2+np.std(baseline3)**2)/2),4)
d0 = np.round((np.mean(sample0)-np.mean(baseline3)) / np.sqrt((np.std(sample0)**2+np.std(baseline3)**2)/2),4)
print(f'{d1,d2,d3,d4,d5,d6, d0}')

p1 = [stats.ttest_ind(a=sample1, b=baseline1)[1]]
p2 = [stats.ttest_ind(a=sample2, b=baseline1)[1]]
p3 = [stats.ttest_ind(a=sample3, b=baseline2)[1]]
p4 = [stats.ttest_ind(a=sample4, b=baseline2)[1]]
p5 = [stats.ttest_ind(a=sample5, b=baseline2)[1]]
p6 = [stats.ttest_ind(a=sample6, b=baseline3)[1]]
p0 = [stats.ttest_ind(a=sample0, b=baseline3)[1]]

# In[ ]:

while len(sample1) > 2:
    sample1.remove(min(sample1))
    sample2.remove(min(sample2))
    baseline1.remove(max(baseline1))
    p1.append(stats.ttest_ind(a=sample1, b=baseline1)[1])
    p2.append(stats.ttest_ind(a=sample2, b=baseline1)[1])
    sample3.remove(min(sample3))
    sample4.remove(min(sample4))
    sample5.remove(min(sample5))
    baseline2.remove(max(baseline2))
    p3.append(stats.ttest_ind(a=sample3, b=baseline2)[1])
    p4.append(stats.ttest_ind(a=sample4, b=baseline2)[1])
    p5.append(stats.ttest_ind(a=sample5, b=baseline2)[1])
    sample6.remove(min(sample6))
    sample0.remove(min(sample0))
    baseline3.remove(max(baseline3))
    p6.append(stats.ttest_ind(a=sample6, b=baseline3)[1])
    p0.append(stats.ttest_ind(a=sample0, b=baseline3)[1])
    # compute power 
    D1 = (np.mean(sample1)-np.mean(baseline1)) / np.sqrt((np.std(sample1)**2+np.std(baseline1)**2)/2)
    D2 = (np.mean(sample2)-np.mean(baseline1)) / np.sqrt((np.std(sample2)**2+np.std(baseline1)**2)/2)
    D3 = (np.mean(sample3)-np.mean(baseline2)) / np.sqrt((np.std(sample3)**2+np.std(baseline2)**2)/2)
    D4 = (np.mean(sample4)-np.mean(baseline2)) / np.sqrt((np.std(sample4)**2+np.std(baseline2)**2)/2)
    D5 = (np.mean(sample5)-np.mean(baseline2)) / np.sqrt((np.std(sample5)**2+np.std(baseline2)**2)/2)
    D6 = (np.mean(sample6)-np.mean(baseline3)) / np.sqrt((np.std(sample6)**2+np.std(baseline3)**2)/2)
    D0 = (np.mean(sample0)-np.mean(baseline3)) / np.sqrt((np.std(sample0)**2+np.std(baseline3)**2)/2)
    power1.append(normal_power(effect_size=D1, nobs=len(sample1), alpha=alpha, alternative='two-sided'))
    power2.append(normal_power(effect_size=D2, nobs=len(sample1), alpha=alpha, alternative='two-sided'))
    power3.append(normal_power(effect_size=D3, nobs=len(sample1), alpha=alpha, alternative='two-sided'))
    power4.append(normal_power(effect_size=D4, nobs=len(sample1), alpha=alpha, alternative='two-sided'))
    power5.append(normal_power(effect_size=D5, nobs=len(sample1), alpha=alpha, alternative='two-sided'))
    power6.append(normal_power(effect_size=D6, nobs=len(sample1), alpha=alpha, alternative='two-sided'))
    power0.append(normal_power(effect_size=D0, nobs=len(sample1), alpha=alpha, alternative='two-sided'))

# In[ ]:

with sns.axes_style('darkgrid'):
    ax = sns.lineplot(x=x[:15], y=p6[:15], color='lightblue', label=f'p at d={d6}')
    sns.lineplot(x=x[:15], y=p5[:15], color='dodgerblue', label=f'p at d={d5}')
    sns.lineplot(x=x[:15], y=p4[:15], color='royalblue', label=f'p at d={d4}')
    sns.lineplot(x=x[:15], y=0.05, color='lime', label='alpha = 5%')
    ax.set(xlabel='number of removed observations', ylabel='p-value')
plt.savefig(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\plots\chapter3.3\plot10')

# In[ ]:

d = 0.08333
c = stats.norm.isf(q=alpha/2) # identical to statsmodels computation
c_shifted = c - d*np.sqrt(95)/1 # in statsmodels: sigma=1, shifted in amount of z-score
power = stats.norm.sf(c_shifted)
print(power)
pow_ = normal_power(effect_size=d, alpha=alpha, nobs=95, alternative='two-sided')
pow_
