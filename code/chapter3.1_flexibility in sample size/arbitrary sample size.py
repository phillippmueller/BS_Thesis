# In[ ]:

import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# In[ ]:

size = 101
x = np.linspace(2,size-1,size-2).astype(int)
data = [np.random.normal(loc=0, scale=1, size=i) for i in range(2,size)]
# truth = H1 = positive
# H0: mean = 1
statistic_pos = [stats.ttest_1samp(a=data[i], popmean=0.5)[0] for i in range(size-2)]
p_pos = [stats.ttest_1samp(a=data[i], popmean=0.5)[1] for i in range(size-2)]
# truth = H0 = negative
# H0: mean = 0
statistic_neg = [stats.ttest_1samp(a=data[i], popmean=0)[0] for i in range(size-2)]
p_neg = [stats.ttest_1samp(a=data[i], popmean=0)[1] for i in range(size-2)]

df = pd.DataFrame(data=[statistic_pos, p_pos, statistic_neg, p_neg], index=['statistic_pos', 'p_pos', 'statistic_neg', 'p_neg'])
df.columns=x
results = df.transpose()

# In[ ]:

# calculate measured standardized effect sizes 
mean = [np.mean(data[i]) for i in range(len(data))]
std = [np.std(data[i]) for i in range(len(data))]
effect_pos = [np.abs(mean[i]-0.5) for i in range(len(data))]
effect_neg = [np.abs(mean[i]-0) for i in range(len(data))]
d_pos = [effect_pos/(np.sqrt(std[i]**2+1)/2) for i in range(len(data))]
d_neg = [effect_neg/(np.sqrt(std[i]**2+1)/2) for i in range(len(data))]
np.mean(d_pos),np.mean(d_neg)

# In[ ]:

with sns.axes_style('darkgrid'):
    g = sns.lineplot(x=x, y=results.p_pos, color='orangered', label='H0=false')
    sns.lineplot(x=x, y=results.p_neg, color='lightseagreen', label='H0=true')
    sns.lineplot(x=x, y=0.05, color='lime', label='alpha=5%')
    g.set(xlabel='sample size', ylabel='p-value')
    plt.legend(loc='upper right')
plt.savefig(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\plots\chapter3.3\plot2')

# In[ ]:

with sns.axes_style('darkgrid'):
    g = sns.lineplot(x=x, y=results.statistic_pos,color='orangered', label='H0=false')
    sns.lineplot(x=x, y=results.statistic_neg, color='lightseagreen', label='H0=true')
    g.set(xlabel='sample size', ylabel='test statistic')
plt.savefig(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\plots\chapter3.3\plot1')

# In[ ]:

# repetition
nsim=1000
size = 101
x = np.linspace(2,size-1,size-2).astype(int)
result_p_pos, result_p_neg = np.array([[0.0]*nsim]*(size-2)), np.array([[0.0]*nsim]*(size-2))

for i in range(nsim):
    data = [np.random.normal(loc=0, scale=1, size=i) for i in range(2,size)]
    statistic_pos = [stats.ttest_1samp(a=data[i], popmean=0.5)[0] for i in range(size-2)]
    statistic_neg = [stats.ttest_1samp(a=data[i], popmean=0)[0] for i in range(size-2)]
    p_pos = [stats.ttest_1samp(a=data[i], popmean=0.5)[1] for i in range(size-2)]
    p_neg = [stats.ttest_1samp(a=data[i], popmean=0)[1] for i in range(size-2)]
    result_p_pos[:,i] = p_pos
    result_p_neg[:,i] = p_neg

# In[ ]:

sns.heatmap(result_p_pos)

# In[ ]:

sns.heatmap(result_p_neg)

# In[ ]:

fn, fp = np.array([[0.0]*nsim]*(size-2)), np.array([[0.0]*nsim]*(size-2))
for i in range(size-2):
    for j in range(nsim):
        if result_p_pos[i,j] < 0.05:
            fn[i,j] = 0 #true positive
        else:
            fn[i,j] = 1 #false negative 
        if result_p_neg[i,j] < 0.05:
            fp[i,j] = 1 #false positive 
        else:
            fp[i,j] = 0 #true negative

# In[ ]:

mean_p_pos = [np.mean(result_p_pos[i,:]) for i in range(size-2)]
mean_p_neg = [np.mean(result_p_neg[i,:]) for i in range(size-2)]
fn_rate = [np.mean(fn[i,:]) for i in range(size-2)]
fp_rate = [np.mean(fp[i,:]) for i in range(size-2)]
tp_rate = [1-fn_rate[i] for i in range(size-2)]
tn_rate = [1-fp_rate[i] for i in range(size-2)]

FDR = [fp_rate[i]/(tp_rate[i]+fp_rate[i]) for i in range(size-2)]

# In[ ]:

std_p_neg = [np.std(result_p_neg[i,:]) for i in range(size-2)]
std_p_pos = [np.std(result_p_pos[i,:]) for i in range(size-2)]
print(f' std in p_neg: {np.mean(std_p_neg)}')
print(f' std in p_pos: {np.mean(std_p_pos)}')
print(f' ratio={np.mean(std_p_pos)/np.mean(std_p_neg)}')

# In[ ]:

with sns.axes_style('darkgrid'):
    g = sns.lineplot(x=x, y=mean_p_pos, color='orangered', label='H0=false')
    sns.lineplot(x=x, y=mean_p_neg, color='lightseagreen', label='H0=true')
    sns.lineplot(x=x, y=0.05, color='lime', label='alpha=5%')
    g.set(xlabel='sample size', ylabel='p-value')
plt.savefig(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\plots\chapter3.3\plot4')

# In[ ]:

with sns.axes_style('darkgrid'):
    g = sns.lineplot(x=x, y=fn_rate, color='orangered', label='false negative rate')
    sns.lineplot(x=x, y=fp_rate, color='lightseagreen', label='false positive rate')
    sns.lineplot(x=x, y=FDR, color='maroon', label='false discovery rate')
    g.set(xlabel='sample size', ylabel='probability')
plt.savefig(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\plots\chapter3.3\plot3')

# In[ ]:

np.mean(fp_rate)
np.mean(fn_rate[:30])
np.mean(fn_rate[30:])

# In[ ]:

np.mean(fn_rate[30:])

# In[ ]:

# varying effect sizes 
nsim=1000
new_size = 201
new_x = np.linspace(2,new_size-1,new_size-2).astype(int)
result1, result2, result3 = np.array([[0.0]*nsim]*(new_size-2)), np.array([[0.0]*nsim]*(new_size-2)), np.array([[0.0]*nsim]*(new_size-2))

for i in range(nsim):
    data = [np.random.normal(loc=0, scale=1, size=i) for i in range(2,new_size)]
    p1 = [stats.ttest_1samp(a=data[i], popmean=-0.2)[1] for i in range(new_size-2)]
    p2 = [stats.ttest_1samp(a=data[i], popmean=-0.5)[1] for i in range(new_size-2)]
    p3 = [stats.ttest_1samp(a=data[i], popmean=-0.8)[1] for i in range(new_size-2)] 
    result1[:,i] = p1
    result2[:,i] = p2
    result3[:,i] = p3

# In[ ]:

fn1, fn2, fn3 = np.array([[0.0]*nsim]*(new_size-2)), np.array([[0.0]*nsim]*(new_size-2)), np.array([[0.0]*nsim]*(new_size-2))
for i in range(new_size-2):
    for j in range(nsim):
        if result1[i,j] < 0.05:
            fn1[i,j] = 0 #true positive
        else:
            fn1[i,j] = 1
        if result2[i,j] < 0.05:
            fn2[i,j] = 0
        else:
            fn2[i,j] = 1
        if result3[i,j] < 0.05:
            fn3[i,j] = 0
        else:
            fn3[i,j] = 1

# In[ ]:

FNR1 = [np.mean(fn1[i,:]) for i in range(new_size-2)]
FNR2 = [np.mean(fn2[i,:]) for i in range(new_size-2)]
FNR3 = [np.mean(fn3[i,:]) for i in range(new_size-2)]

p_val1 = [np.mean(result1[i,:]) for i in range(new_size-2)]
p_val2 = [np.mean(result2[i,:]) for i in range(new_size-2)]
p_val3 = [np.mean(result3[i,:]) for i in range(new_size-2)]

# In[ ]:

with sns.axes_style('darkgrid'):
    g = sns.lineplot(x=new_x, y=FNR1, color='salmon', label='FNR at d=0.2')
    sns.lineplot(x=new_x, y=FNR2, color='indianred', label='FNR at d=0.5')
    sns.lineplot(x=new_x, y=FNR3, color='darkred', label='FNR at d=0.8')
    g.set(xlabel='sample size', ylabel='probability')
plt.savefig(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\plots\chapter3.3\plot5')

# In[ ]:

with sns.axes_style('darkgrid'):
    g = sns.lineplot(x=new_x, y=p_val1, color='skyblue', label='p at d=0.2')
    sns.lineplot(x=new_x, y=p_val2, color='royalblue', label='p at d=0.5')
    sns.lineplot(x=new_x, y=p_val3, color='navy', label='p at d=0.8')
    sns.lineplot(x=new_x, y=0.05, color='lime', label='alpha=5%')
    g.set(xlabel='sample size', ylabel='p-value')
plt.savefig(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\plots\chapter3.3\plot6')
