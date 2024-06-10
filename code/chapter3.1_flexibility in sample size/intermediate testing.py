# In[ ]:

import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# In[ ]:

# generate a sample with H0=true
# generate a sample with H0=false
# perform paired t-test
# measure: p, test statistic = t, false positives = fp, false negatives = fn
# increase sample size by one observation 
# repeat test

# !!! use generated data to know truth value of test decision !!!

# In[ ]:

df_pos = pd.read_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\df1.csv').query('location==0.5 and scale==1')
df_neg = pd.read_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\df1.csv').query('location==0 and scale==1')
df_base = pd.read_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\df2.csv').query('scale==1')

# In[ ]:

# truth = H1 = positive
# H0: equal means 
# possible outcomes in the population: false negative, true positive 
sample1_pos = df_pos.iloc[:,3].values.tolist()
sample2_pos = df_base.iloc[:,2].values.tolist()
effect = (np.mean(sample1_pos)-np.mean(sample2_pos))
d = np.round(effect/np.sqrt((np.std(sample1_pos)**2+np.std(sample2_pos)**2)/2), 4)
print(f'd = {d}')
# truth = H0 = negative
# H0: equal means
# possible outcomes in the population: false positive, true negative

sample1_neg = df_neg.iloc[:,3].values.tolist()
sample2_neg = sample2_pos

# In[ ]:

group1_pos, group2_pos = [], []
t_pos = np.array([0.0]*len(sample1_pos))
p_pos = np.array([0.0]*len(sample1_pos))
fn = np.array([0.0]*len(sample1_pos))

group1_neg, group2_neg = [], []
t_neg = np.array([0.0]*len(sample1_pos))
p_neg = np.array([0.0]*len(sample1_pos))
fp = np.array([0.0]*len(sample1_pos))

for i in range(len(df_base)):
    # H0 false
    group1_pos.append(sample1_pos[i])
    group2_pos.append(sample2_pos[i])
    statistic_pos = stats.ttest_ind(a=group1_pos,b=group2_pos)
    t_pos[i], p_pos[i] = statistic_pos[0], statistic_pos[1]
    # H0 true
    group1_neg.append(sample1_neg[i])
    group2_neg.append(sample2_neg[i])
    statistic_neg = stats.ttest_ind(a=group1_neg, b=group2_neg)
    t_neg[i], p_neg[i] = statistic_neg[0], statistic_neg[1]

# In[ ]:

x = np.arange(0,101,).astype(int)
df = pd.DataFrame(data=[t_pos, p_pos, t_neg, p_neg], index=['t_pos', 'p_pos', 't_neg', 'p_neg'], columns=x)
results = df.dropna(axis=1)
results

# In[ ]:

x = np.linspace(2,len(sample1_pos),len(sample1_pos)-1).astype(int)
alpha = 0.05
with sns.axes_style('darkgrid'):
    ax = sns.lineplot(x=x, y=results.iloc[1,:], color='orangered', label='H0=false')
    sns.lineplot(x=x, y=results.iloc[3,:], color='lightseagreen', label='H0=true')
    sns.lineplot(x=x, y=alpha, color='lime', label='alpha=5%')
    ax.set(xlabel='sample size', ylabel='p-value')
    plt.legend(bbox_to_anchor=(0.7,0.2))
plt.savefig(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\plots\chapter3.3\plot8')

# In[ ]:

with sns.axes_style('darkgrid'):
    g = sns.lineplot(x=x, y=results.iloc[0,:], color='orangered', label='H0=false')
    sns.lineplot(x=x, y=results.iloc[2,:], color='lightseagreen', label='H0=true')
    g.set(xlabel='sample size', ylabel='test statistic')

# In[ ]:

# varying effects
df1 = pd.read_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\df1.csv')
df2 = pd.read_csv(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\data\df2.csv')

# In[ ]:

sample1 = df1.query('location==0.75 and scale==1').iloc[:,3].values.tolist()
sample2 = df1.query('location==0.5 and scale==1').iloc[:,3].values.tolist()
sample3 = df1.query('location==0.75 and scale==2').iloc[:,3].values.tolist()
sample4 = df1.query('location==0.5 and scale==2').iloc[:,3].values.tolist()
sample5 = df1.query('location==0.25 and scale==2').iloc[:,3].values.tolist()
sample6 = df1.query('location==0.25 and scale==3').iloc[:,3].values.tolist()
baseline1 = df2.query('scale==1').iloc[:,2].values.tolist()
baseline2 = df2.query('scale==2').iloc[:,2].values.tolist()
baseline3 = df2.query('scale==3').iloc[:,2].values.tolist()

# In[ ]:

p1,p2,p3 = np.array([0.0]*(len(sample1)-1)), np.array([0.0]*(len(sample1)-1)), np.array([0.0]*(len(sample1)-1))
p4,p5,p6 = np.array([0.0]*(len(sample1)-1)), np.array([0.0]*(len(sample1)-1)), np.array([0.0]*(len(sample1)-1))

cond1_1, cond1_2, cond1_3 = [sample1[0]],[sample2[0]],[sample3[0]]
cond1_4, cond1_5, cond1_6 = [sample4[0]],[sample5[0]],[sample6[0]]

cond2_1, cond2_2, cond2_3 = [baseline1[0]], [baseline2[0]], [baseline3[0]]
for i in range(1,len(sample1)):
    cond1_1.append(sample1[i])
    cond1_2.append(sample2[i])
    cond1_3.append(sample3[i])
    cond1_4.append(sample4[i])
    cond1_5.append(sample5[i])
    cond1_6.append(sample6[i])
    cond2_1.append(baseline1[i])
    cond2_2.append(baseline2[i])
    cond2_3.append(baseline3[i])
    p1[i-1] = stats.ttest_ind(a=cond1_1,b=cond2_1)[1]
    p2[i-1] = stats.ttest_ind(a=cond1_2, b=cond2_1)[1]
    p3[i-1] = stats.ttest_ind(a=cond1_3,b=cond2_2)[1] 
    p4[i-1] = stats.ttest_ind(a=cond1_4,b=cond2_2)[1]
    p5[i-1] = stats.ttest_ind(a=cond1_5,b=cond2_2)[1]
    p6[i-1] = stats.ttest_ind(a=cond1_6,b=cond2_3)[1]

# In[ ]:

d1 = np.round((np.mean(sample1)-np.mean(baseline1)) / np.sqrt((np.std(sample1)**2+np.std(baseline1)**2)/2),4)
d2 = np.round((np.mean(sample2)-np.mean(baseline1)) / np.sqrt((np.std(sample2)**2+np.std(baseline1)**2)/2),4)
d3 = np.round((np.mean(sample3)-np.mean(baseline2)) / np.sqrt((np.std(sample3)**2+np.std(baseline2)**2)/2),4)
d4 = np.round((np.mean(sample4)-np.mean(baseline2)) / np.sqrt((np.std(sample4)**2+np.std(baseline2)**2)/2),4)
d5 = np.round((np.mean(sample5)-np.mean(baseline2)) / np.sqrt((np.std(sample5)**2+np.std(baseline2)**2)/2),4)
d6 = np.round((np.mean(sample6)-np.mean(baseline3)) / np.sqrt((np.std(sample6)**2+np.std(baseline3)**2)/2),4)
print(f'{d1,d2,d3,d4,d5,d6}')

# In[ ]:

print(np.mean(sample1)-np.mean(baseline1))
print(np.mean(sample2)-np.mean(baseline1))
print(np.mean(sample3)-np.mean(baseline2))
print(np.mean(sample4)-np.mean(baseline2))
print(np.mean(sample5)-np.mean(baseline2))
print(np.mean(sample6)-np.mean(baseline3))

# In[ ]:

print(np.mean(baseline3))

# In[ ]:

with sns.axes_style('darkgrid'):
    ax = sns.lineplot(x=x, y=p6, color='lightblue', label=f'p at d={d6}')
    sns.lineplot(x=x, y=p5, color='dodgerblue', label=f'p at d={d5}')
    sns.lineplot(x=x, y=p4, color='royalblue', label=f'p at d={d4}')
    sns.lineplot(x=x, y=p3, color='darkblue', label=f'p at d={d3}') 
    sns.lineplot(x=x, y=0.05, color='lime', label='alpha = 5%')
    ax.set(xlabel='sample size', ylabel='p-value')
    plt.legend(bbox_to_anchor=(.18,1))
plt.savefig(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\plots\chapter3.3\plot7')
