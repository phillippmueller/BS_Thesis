# In[ ]:

import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# In[ ]:

### plot power on effect size axis ###
p, alpha = 0.0013, 0.05
c_p = stats.norm.isf(p)
c_alpha = stats.norm.isf(alpha)

N = 66
mean_reported = 0.426
z_reported = stats.norm.isf(q = p/2) # assuming two-tailed test was performed 
se_reported = mean_reported/z_reported
d = np.linspace(0,1,1000)

# based on estimates
c_est_p, c_est_alpha = np.array([0.0]*len(d)), np.array([0.0]*len(d))
pow_est_p, pow_est_alpha = np.array([0.0]*len(d)),np.array([0.0]*len(d))
# based on reported se    
d_se = d/(se_reported*np.sqrt(N))
c_se_p, c_se_alpha = np.array([0.0]*len(d)), np.array([0.0]*len(d))
pow_se_p, pow_se_alpha = np.array([0.0]*len(d)), np.array([0.0]*len(d))

for i in range(len(d)):
    c_est_p[i], c_est_alpha[i] = c_p-d[i]*np.sqrt(N), c_alpha-d[i]*np.sqrt(N)
    pow_est_p[i], pow_est_alpha[i] = stats.norm.sf(c_est_p[i]), stats.norm.sf(c_est_alpha[i])
    c_se_p[i], c_se_alpha[i] = c_p-d_se[i]*np.sqrt(N), c_alpha-d_se[i]*np.sqrt(N)
    pow_se_p[i], pow_se_alpha[i] = stats.norm.sf(c_se_p[i]), stats.norm.sf(c_se_alpha[i])
    
with sns.axes_style('darkgrid'):
    g = sns.lineplot(x=d, y=pow_est_p, color='firebrick', label='estimated sd, p=0.13%')
    sns.lineplot(x=d, y=pow_est_alpha, color='orangered', label='estimated sd, alpha=5%')
    sns.lineplot(x=d, y=pow_se_p, color='darkcyan', label='reported sd, p=0.13%')
    sns.lineplot(x=d, y=pow_se_alpha, color='darkturquoise', label='reported sd, alpha=5%')
    g.set(xlabel='cohen`s d', ylabel='power', ylim=(-0.05,1.05))
plt.savefig(r'C:\Users\phili\OneDrive\Dokumente_One Drive\KIT\Bachelorarbeit\plots\chapter3.4\plot')
